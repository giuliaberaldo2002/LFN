"""
------------------
Manual urbanity tuning via constrained random search.

Assumes the graph already has (node) attributes computed:
  - degree
  - clustering_coeff
  - avg_edge_len
  - freq_residential
  - freq_motorway
  - (optional) avg_maxspeed

It then:
  1) builds a feature matrix per node with stable transforms (log1p, winsorization)
  2) z-scores (StandardScaler)
  3) runs a random search over weight vectors with fixed sign constraints
  4) picks the best weights according to an interpretable objective (separation of anchors)
  5) saves the chosen weights + metadata to JSON

Usage:
  python urbanity_tuning.py \
      --input bremen_processed_graph.pkl \
      --output urbanity_weights.json \
      --trials 800 --sample 80000 --seed 0
"""

import argparse
import json
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("urbanity_tuning")


# -----------------------------
# Utilities
# -----------------------------

def _as_float_array(x) -> np.ndarray:
    a = np.array(x, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def _cap_percentile(x: np.ndarray, p: float = 99.5) -> np.ndarray:
    """Winsorize upper tail to reduce outlier dominance."""
    x = x.copy()
    mask = np.isfinite(x)
    if not np.any(mask):
        return np.zeros_like(x)
    hi = np.percentile(x[mask], p)
    x[~mask] = 0.0
    return np.clip(x, None, hi)


@dataclass
class TuningConfig:
    trials: int = 800
    sample_n: int = 80000
    q: float = 0.2
    seed: int = 0
    cap_p: float = 99.5
    include_maxspeed: bool = True

    # Ranges for random weights (magnitudes), signs are fixed by range sign.
    # You can tweak these later if needed.
    w_ranges: Dict[str, Tuple[float, float]] = None

    def __post_init__(self):
        if self.w_ranges is None:
            self.w_ranges = {
                # Urban-positive
                "log_degree": (0.6, 1.8),
                "clustering_coeff": (0.1, 1.0),
                "freq_residential": (0.2, 1.6),
                # Urban-negative
                "avg_edge_len": (-1.8, -0.2),
                "freq_motorway": (-1.6, -0.1),
                # Optional negative
                "avg_maxspeed": (-1.8, -0.2),
            }


def load_graph_pkg(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path, "rb") as f:
        pkg = pickle.load(f)
    if "graph" not in pkg:
        raise ValueError("Pickle does not contain key 'graph'.")
    return pkg


def require_node_attrs(g, attrs: List[str]) -> None:
    missing = [a for a in attrs if a not in g.vs.attribute_names()]
    if missing:
        raise ValueError(
            "Missing required node attributes: "
            + ", ".join(missing)
            + ".\nCompute features first (degree, clustering_coeff, avg_edge_len, freq_*, avg_maxspeed)."
        )


# -----------------------------
# Feature matrix builder
# -----------------------------

def build_feature_matrix(
    g,
    cap_p: float = 99.5,
    include_maxspeed: bool = True,
) -> Tuple[np.ndarray, List[str], Dict]:
    """
    Build X (N x d) with stable transforms and minimal compositional issues.

    We intentionally DO NOT include freq_other / freq_service to avoid
    compositional redundancy dominating the signal.
    """
    base_attrs = ["degree", "clustering_coeff", "avg_edge_len", "freq_residential", "freq_motorway"]
    if include_maxspeed:
        base_attrs.append("avg_maxspeed")

    require_node_attrs(g, base_attrs)

    deg = _as_float_array(g.vs["degree"])
    clu = _as_float_array(g.vs["clustering_coeff"])
    elen = _as_float_array(g.vs["avg_edge_len"])
    res = _as_float_array(g.vs["freq_residential"])
    mw = _as_float_array(g.vs["freq_motorway"])

    # transforms
    deg = np.log1p(deg)
    elen = _cap_percentile(elen, cap_p)

    cols = [deg, clu, res, elen, mw]
    names = ["log_degree", "clustering_coeff", "freq_residential", "avg_edge_len", "freq_motorway"]

    transforms = {
        "log_degree": True,
        "cap_percentile": cap_p,
        "zscore": True,
        "dropped_freqs": ["freq_other", "freq_service", "freq_primary"],  # document intent
    }

    if include_maxspeed and "avg_maxspeed" in g.vs.attribute_names():
        ms = _as_float_array(g.vs["avg_maxspeed"])
        ms = _cap_percentile(ms, cap_p)
        cols.append(ms)
        names.append("avg_maxspeed")
        transforms["include_maxspeed"] = True
    else:
        transforms["include_maxspeed"] = False

    X = np.column_stack(cols).astype(np.float32, copy=False)
    return X, names, transforms


# -----------------------------
# Tuning objective
# -----------------------------

def objective_separation(
    Zs: np.ndarray,
    scores: np.ndarray,
    names: List[str],
    q: float,
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate how well scores separate anchor features between top and bottom q-quantiles.

    Since Zs is z-scored, deltas are in SD units and comparable.
    """
    lo = np.quantile(scores, q)
    hi = np.quantile(scores, 1.0 - q)
    bot = scores <= lo
    top = scores >= hi

    def delta(feat: str) -> float:
        j = names.index(feat)
        return float(Zs[top, j].mean() - Zs[bot, j].mean())

    d_deg = delta("log_degree")
    d_res = delta("freq_residential")
    d_clu = delta("clustering_coeff")
    d_len = delta("avg_edge_len")
    d_mw  = delta("freq_motorway")
    d_ms  = delta("avg_maxspeed") if "avg_maxspeed" in names else 0.0

    # We want:
    #  top should have higher deg/res/clu (positive deltas)
    #  top should have lower edge_len/motorway/maxspeed (negative deltas)
    J = (
        1.00 * d_deg +
        0.80 * d_res +
        0.25 * d_clu
        - 0.80 * d_len
        - 0.60 * d_mw
        - 0.50 * d_ms
    )

    # Hard penalties if direction is “wrong”
    if d_deg < 0: J -= 5.0
    if d_res < 0: J -= 3.0
    if d_len > 0: J -= 3.0
    if d_mw  > 0: J -= 3.0
    if ("avg_maxspeed" in names) and (d_ms > 0): J -= 2.0

    details = {
        "d_log_degree": d_deg,
        "d_freq_residential": d_res,
        "d_clustering_coeff": d_clu,
        "d_avg_edge_len": d_len,
        "d_freq_motorway": d_mw,
        "d_avg_maxspeed": d_ms,
    }
    return float(J), details


def tune_weights(
    Z: np.ndarray,
    names: List[str],
    cfg: TuningConfig,
) -> Tuple[np.ndarray, Dict]:
    """
    Random search with sign constraints encoded by the ranges in cfg.w_ranges.
    """
    rng = np.random.default_rng(cfg.seed)
    N, d = Z.shape

    # Sample for speed
    if N > cfg.sample_n:
        idx = rng.choice(N, size=cfg.sample_n, replace=False)
        Zs = Z[idx]
    else:
        idx = None
        Zs = Z

    # Precompute index mapping
    name_to_j = {n: j for j, n in enumerate(names)}

    # Build per-dimension ranges aligned to Z columns
    ranges = [(-0.01, 0.01)] * d  # default tiny noise for unused dims (shouldn’t happen)
    for feat, (a, b) in cfg.w_ranges.items():
        if feat in name_to_j:
            ranges[name_to_j[feat]] = (a, b)

    best_w = None
    best_J = -1e18
    best_details = None

    for t in range(cfg.trials):
        w = np.zeros(d, dtype=np.float32)
        for j in range(d):
            a, b = ranges[j]
            # If a==b, deterministic; otherwise random
            w[j] = rng.uniform(a, b) if a != b else np.float32(a)

        # Normalize scale for stability (doesn't change ranking in z-scored space)
        w = w / (np.linalg.norm(w) + 1e-8)

        s = (Zs @ w).astype(np.float32, copy=False)
        J, details = objective_separation(Zs, s, names, cfg.q)

        if J > best_J:
            best_J = J
            best_w = w
            best_details = details

        if (t + 1) % max(50, cfg.trials // 10) == 0:
            logger.info(f"Tuning progress: {t+1}/{cfg.trials} best_J={best_J:.4f}")

    info = {
        "best_objective": float(best_J),
        "best_details": best_details,
        "q": cfg.q,
        "sample_n": min(cfg.sample_n, N),
        "seed": cfg.seed,
    }
    return best_w, info


# -----------------------------
# Main (CLI)
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input pickle containing {'graph': ig.Graph, ...}")
    ap.add_argument("--output", required=True, help="Output JSON path for tuned weights")
    ap.add_argument("--trials", type=int, default=800)
    ap.add_argument("--sample", type=int, default=80000)
    ap.add_argument("--q", type=float, default=0.2, help="Quantile for top/bottom separation (e.g., 0.2)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cap", type=float, default=99.5, help="Upper percentile cap (winsorize) for heavy tails")
    ap.add_argument("--no-maxspeed", action="store_true", help="Exclude avg_maxspeed from tuning/features")
    args = ap.parse_args()

    pkg = load_graph_pkg(args.input)
    g = pkg["graph"]

    cfg = TuningConfig(
        trials=args.trials,
        sample_n=args.sample,
        q=args.q,
        seed=args.seed,
        cap_p=args.cap,
        include_maxspeed=(not args.no_maxspeed),
    )

    logger.info(f"Loaded graph: nodes={g.vcount():,} edges={g.ecount():,}")

    # Build features
    X, names, transforms = build_feature_matrix(
        g,
        cap_p=cfg.cap_p,
        include_maxspeed=cfg.include_maxspeed,
    )
    logger.info(f"Feature set ({len(names)}): {names}")

    # Standardize
    scaler = StandardScaler()
    Z = scaler.fit_transform(X).astype(np.float32, copy=False)

    # Tune weights
    best_w, info = tune_weights(Z, names, cfg)

    logger.info(f"Best objective: {info['best_objective']:.4f}")
    logger.info("Best deltas (top - bottom on z-scored features):")
    for k, v in info["best_details"].items():
        logger.info(f"  {k:<22s}: {v:+.4f}")

    logger.info("Chosen weights (unit-norm):")
    for n, w in zip(names, best_w):
        logger.info(f"  {n:<18s}: {float(w):+.4f}")

    # Save
    out = {
        "feature_names": names,
        "weights": [float(x) for x in best_w],
        "transforms": transforms,
        "tuning_info": info,
        "notes": {
            "score_definition": "urbanity_score = zscore(transformed_features) @ weights",
            "urban_high_score": True,
            "recommended_split": "GMM(2) on urbanity_score, keep component with higher mean",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    logger.info(f"Saved tuned weights to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
