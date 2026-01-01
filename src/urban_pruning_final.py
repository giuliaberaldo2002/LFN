"""
Urbanity Pruning (Manual Tuning + GMM)
--------------------------------------
Computes features for an OSM road network, applies a manually tuned urbanity score
(learned by constrained random search in urbanity_tuning.py), and prunes rural nodes
by fitting a 2-component GMM on the 1D score distribution.

"""

import math
import os
import sys
import json
import pickle
import logging
from collections import Counter
from typing import Any, Optional, Tuple, Dict

import numpy as np
import igraph as ig
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -----------------------
# Constants
# -----------------------
INPUT_PKL = "bremen_processed_graph.pkl"
OUTPUT_PRUNED_PKL = "bremen_pruned_graph.pkl"
WEIGHTS_JSON = "urbanity_weights.json"

MAIN_HIGHWAY_TYPES = {"residential", "primary", "motorway", "service"}


# -----------------------
# I/O
# -----------------------
def load_graph_data(filepath: str) -> Tuple[ig.Graph, Dict[int, int]]:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found. Run graph_init_optimized.py first.")
    logger.info(f"Loading graph from {filepath}...")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["graph"], data.get("osmid_map", {})

def load_tuned_weights(json_path: str) -> Tuple[list[str], np.ndarray, dict]:
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} not found. Run urbanity_tuning.py first.")
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    names = obj["feature_names"]
    w = np.array(obj["weights"], dtype=np.float32)
    meta = obj.get("transforms", {})
    return names, w, meta


# -----------------------
# Feature computation
# -----------------------
def parse_maxspeed(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.lower().strip()
        if val in {"none", "signals", "variable", "walk"}:
            return None
        import re
        m = re.search(r"(\d+(\.\d+)?)", val)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None

def add_tag_based_features(g: ig.Graph) -> None:
    logger.info("Computing tag-based features...")
    es_attrs = g.es.attribute_names()
    has_highway = "highway" in es_attrs
    has_maxspeed = "maxspeed" in es_attrs

    freq_results = {k: np.zeros(g.vcount(), dtype=np.float32)
                    for k in ["residential", "primary", "motorway", "service", "other"]}
    avg_speeds = np.zeros(g.vcount(), dtype=np.float32)

    for v in range(g.vcount()):
        inc_edges = g.incident(v, mode="ALL")
        if not inc_edges:
            continue

        edge_objs = g.es[inc_edges]
        hw_types = []
        speeds = []

        for e in edge_objs:
            if has_highway:
                hw = e["highway"]
                if isinstance(hw, list):
                    hw = hw[0]
                hw_types.append(str(hw) if hw else "other")
            else:
                hw_types.append("other")

            if has_maxspeed:
                sp = parse_maxspeed(e["maxspeed"])
                if sp is not None:
                    speeds.append(sp)

        total = len(hw_types)
        counts = Counter(hw_types)

        freq_results["residential"][v] = counts.get("residential", 0) / total
        freq_results["primary"][v] = counts.get("primary", 0) / total
        freq_results["motorway"][v] = counts.get("motorway", 0) / total
        freq_results["service"][v] = counts.get("service", 0) / total

        other_count = sum(c for k, c in counts.items() if k not in MAIN_HIGHWAY_TYPES)
        freq_results["other"][v] = other_count / total

        if speeds:
            avg_speeds[v] = float(np.mean(speeds))

    for k, vals in freq_results.items():
        g.vs[f"freq_{k}"] = vals.tolist()
    g.vs["avg_maxspeed"] = avg_speeds.tolist()

def compute_topology_features(g: ig.Graph) -> None:
    """
    NOTE: betweenness removed (too slow and not needed for pruning).
    """
    logger.info("Computing topological features...")
    g.vs["degree"] = g.degree()

    clust = g.transitivity_local_undirected()
    g.vs["clustering_coeff"] = [c if not math.isnan(c) else 0.0 for c in clust]

    # Avg edge length (uses edge attr "length" if present)
    weights = g.es["length"] if "length" in g.es.attribute_names() else None
    if weights is None:
        g.vs["avg_edge_len"] = [0.0] * g.vcount()
        return

    avg_lens = [0.0] * g.vcount()
    for v in range(g.vcount()):
        edges = g.incident(v)
        if edges:
            lens = [weights[e] for e in edges]
            avg_lens[v] = float(np.mean(lens))
    g.vs["avg_edge_len"] = avg_lens


# -----------------------
# Manual-score feature matrix builder (MUST match tuning)
# -----------------------
def _cap_percentile(x: np.ndarray, p: float) -> np.ndarray:
    x = x.copy()
    mask = np.isfinite(x)
    if not np.any(mask):
        return np.zeros_like(x)
    hi = np.percentile(x[mask], p)
    x[~mask] = 0.0
    return np.clip(x, None, hi)

def build_X_from_names(g: ig.Graph, feature_names: list[str], cap_p: float) -> np.ndarray:
    """
    Builds X columns in the same order as the tuned feature_names.
    Supported names (from tuner):
      - log_degree
      - clustering_coeff
      - freq_residential
      - avg_edge_len
      - freq_motorway
      - avg_maxspeed (optional)
    """
    cols = []
    for name in feature_names:
        if name == "log_degree":
            deg = np.array(g.vs["degree"], dtype=np.float64)
            deg = np.nan_to_num(deg, nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(np.log1p(deg))
        elif name == "clustering_coeff":
            x = np.array(g.vs["clustering_coeff"], dtype=np.float64)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(x)
        elif name == "freq_residential":
            x = np.array(g.vs["freq_residential"], dtype=np.float64)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(x)
        elif name == "avg_edge_len":
            x = np.array(g.vs["avg_edge_len"], dtype=np.float64)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(_cap_percentile(x, cap_p))
        elif name == "freq_motorway":
            x = np.array(g.vs["freq_motorway"], dtype=np.float64)
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            cols.append(x)
        elif name == "avg_maxspeed":
            if "avg_maxspeed" not in g.vs.attribute_names():
                cols.append(np.zeros(g.vcount(), dtype=np.float64))
            else:
                x = np.array(g.vs["avg_maxspeed"], dtype=np.float64)
                x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                cols.append(_cap_percentile(x, cap_p))
        else:
            raise ValueError(f"Unsupported tuned feature name: {name}")
    X = np.column_stack(cols).astype(np.float32, copy=False)
    return X


def gmm_keep_indices(scores: np.ndarray, seed: int = 0, prob_threshold: float = 0.5) -> np.ndarray:
    S = scores.reshape(-1, 1).astype(np.float32, copy=False)
    gmm = GaussianMixture(n_components=2, random_state=seed).fit(S)
    means = gmm.means_.ravel()
    urban_label = int(np.argmax(means))
    proba_urban = gmm.predict_proba(S)[:, urban_label]
    keep = np.where(proba_urban >= prob_threshold)[0]
    logger.info(f"GMM means={means}, urban_label={urban_label}, keep={len(keep):,}/{len(scores):,}")
    return keep


# -----------------------
# Main
# -----------------------
def main():
    logger.info("--- START JOB: Urban Pruning (Manual Tuning + GMM) ---")

    g, osmid_map = load_graph_data(INPUT_PKL)
    logger.info(f"Nodes: {g.vcount():,}, Edges: {g.ecount():,}")

    # 1) compute features (same as before)
    add_tag_based_features(g)
    compute_topology_features(g)

    # 2) load tuned weights
    tuned_names, w, meta = load_tuned_weights(WEIGHTS_JSON)
    cap_p = float(meta.get("cap_percentile", 99.5))
    logger.info(f"Loaded tuned weights from {WEIGHTS_JSON}")
    logger.info(f"Feature order: {tuned_names}")

    # 3) build X and standardize
    X = bu
