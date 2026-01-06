
#!/usr/bin/env python3
"""
compare_algorithms.py
--------------------
Compare sampling methods using MidTerm metrics (from sampling.py) on a PRUNED graph.

We compare 3 methods:
  - Dummy 1: Full Random (sample from full graph; evaluate only the hits in pruned)
  - Dummy 2: Pruned Random (random within pruned)
  - Pipeline (Smart): Round-robin (community-based) OR FFT (graph-distance k-center-ish)

Metrics (Midterm-aligned):
  - efficiency (urban hit rate for full-random; for pruned methods it's 1.0)
  - community coverage
  - global coverage: R_max, mean, median, p90 of d(x,S)
  - diversity: min_sep, mean, median, p10, p90 of pairwise d(si,sj)
  - balance: cv, entropy_norm
  - optional: ARI/NMI (sanity check)

Outputs:
  - results_raw.csv (one row per (k, algo, repeat))
  - results_agg.csv (mean/std aggregated over repeats)
  - plots/*.png (metric vs k with error bars)

Requires:
  - full graph pickle: {'graph': ig.Graph, ...} with vertex 'osmid' or implicit ids
  - pruned graph pickle: {'graph': ig.Graph, ...} WITH vertex 'community' (+ 'osmid' recommended)
"""

import argparse
import os
import pickle
import logging
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sampling import (
    sample_round_robin,
    fft_sample_graph,
    hybrid_city_ordered_round_robin,
    build_csr_adjacency,
    evaluate_midterm,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("compare")


# -------------------------
# IO helpers
# -------------------------

def load_graph_pkg(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "rb") as f:
        pkg = pickle.load(f)
    if "graph" not in pkg:
        raise ValueError(f"{path} does not contain key 'graph'")
    return pkg

def get_osmid_array(g) -> np.ndarray:
    if "osmid" in g.vs.attribute_names():
        return np.asarray(g.vs["osmid"])
    # fallback: internal ids
    return np.arange(g.vcount(), dtype=np.int64)

def build_osmid_to_index(g) -> Dict[int, int]:
    osmids = get_osmid_array(g)
    return {int(osmids[i]): int(i) for i in range(len(osmids))}


# -------------------------
# Algorithms
# -------------------------

def algo_full_random(full_g, k: int, seed: int) -> List[int]:
    """Return a list of OSMIDs sampled uniformly from FULL graph."""
    rng = np.random.default_rng(seed)
    n = full_g.vcount()
    if n == 0 or k <= 0:
        return []
    idx = rng.choice(n, size=min(k, n), replace=False)
    osmids = get_osmid_array(full_g)
    return [int(osmids[i]) for i in idx]

def algo_pruned_random(pruned_g, k: int, seed: int) -> List[int]:
    """Return a list of OSMIDs sampled uniformly from PRUNED graph."""
    rng = np.random.default_rng(seed)
    n = pruned_g.vcount()
    if n == 0 or k <= 0:
        return []
    idx = rng.choice(n, size=min(k, n), replace=False)
    osmids = get_osmid_array(pruned_g)
    return [int(osmids[i]) for i in idx]

def algo_pipeline_smart(pruned_g, k: int, seed: int, method: str, weight_attr: str) -> List[int]:
    """
    Smart sampling on PRUNED graph.
    method: 'round_robin' or 'fft'
    Returns OSMIDs.
    """
    if k <= 0 or pruned_g.vcount() == 0:
        return []

    if method == "round_robin":
        if "community" not in pruned_g.vs.attribute_names():
            raise ValueError("round_robin requires vertex attribute 'community' in pruned graph.")
        idx = sample_round_robin(pruned_g, k, seed=seed)  # no duplicates :contentReference[oaicite:5]{index=5}
    elif method == "fft":
        # deterministic seed selection is inside fft_sample_graph unless you set seed_idx
        idx = fft_sample_graph(pruned_g, k, weight_attr=weight_attr, seed_idx=None)
    elif method == "hybrid":
        idx = hybrid_city_ordered_round_robin(pruned_g, k, seed=seed, weight_attr=weight_attr)
    else:
        raise ValueError(f"Unknown smart method: {method}")

    osmids = get_osmid_array(pruned_g)
    return [int(osmids[i]) for i in idx]


# -------------------------
# Evaluation wrapper (Midterm metrics)
# -------------------------

def flatten_metrics(mid: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten evaluate_midterm() dict into scalar columns."""
    cov = mid["community_coverage"]
    bal = mid["balance"]
    glob = mid["global_coverage"]
    div = mid["diversity"]
    agree = mid.get("clustering_agreement", {})

    out = {
        "cov": float(cov["coverage"]),
        "n_comm_full": int(cov["n_communities_full"]),
        "n_comm_repr": int(cov["n_communities_represented"]),
        "balance_cv": float(bal["cv"]),
        "balance_entropy": float(bal["entropy_norm"]),
        "global_R_max": float(glob["R_max"]),
        "global_mean": float(glob["mean"]),
        "global_median": float(glob["median"]),
        "global_p90": float(glob["p90"]),
        "div_min_sep": float(div["min_sep"]),
        "div_mean": float(div["mean"]),
        "div_median": float(div["median"]),
        "div_p10": float(div["p10"]),
        "div_p90": float(div["p90"]),
        "ari": (None if agree.get("ari") is None else float(agree["ari"])),
        "nmi": (None if agree.get("nmi") is None else float(agree["nmi"])),
    }
    return out


def evaluate_on_pruned(
    pruned_g,
    pruned_osmid_to_idx: Dict[int, int],
    sample_osmids: List[int],
    labels_full: np.ndarray,
    indptr, indices, data,
    weight_attr: str,
    eval_subset: np.ndarray | None,
) -> Tuple[Dict[str, Any], int, float]:
    """
    Map OSMIDs to PRUNED indices; compute:
      - metrics via evaluate_midterm() on PRUNED graph
      - k_eff = number of valid (urban) samples used
      - efficiency = k_eff / k_requested
    """
    k_req = len(sample_osmids)
    valid_idx = [pruned_osmid_to_idx[o] for o in sample_osmids if o in pruned_osmid_to_idx]
    k_eff = len(valid_idx)
    efficiency = (k_eff / k_req) if k_req > 0 else 0.0

    if k_eff == 0:
        # return NaNs for midterm metrics if nothing hits the pruned set
        nan = float("nan")
        mid = {
            "community_coverage": {"coverage": 0.0, "n_communities_full": int(len(np.unique(labels_full))),
                                  "n_communities_represented": 0, "counts_per_community": {}},
            "balance": {"cv": nan, "entropy_norm": nan},
            "global_coverage": {"R_max": nan, "mean": nan, "median": nan, "p90": nan},
            "diversity": {"min_sep": nan, "mean": nan, "median": nan, "p10": nan, "p90": nan},
            "clustering_agreement": {"ari": None, "nmi": None},
        }
        return mid, k_eff, efficiency

    mid = evaluate_midterm(
        pruned_g,
        labels_full=labels_full,
        sampled_nodes=valid_idx,
        indptr=indptr,
        indices=indices,
        data=data,
        eval_subset=eval_subset,
        weight_attr=weight_attr,
    )  # metrics aligned to midterm 
    return mid, k_eff, efficiency


# -------------------------
# Plotting
# -------------------------

def plot_metric_with_errorbars(df_agg: pd.DataFrame, metric: str, ylabel: str, title: str, out_png: str):
    plt.figure(figsize=(10, 6))
    for algo in df_agg["algo"].unique():
        sub = df_agg[df_agg["algo"] == algo].sort_values("k")
        plt.errorbar(
            sub["k"], sub[f"{metric}_mean"],
            yerr=sub[f"{metric}_std"],
            marker="o", capsize=3, linewidth=2, label=algo
        )
    plt.xlabel("k (Number of samples)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", required=True, help="Full graph pickle")
    ap.add_argument("--pruned", required=True, help="Pruned graph pickle (must include 'community')")
    ap.add_argument("--outdir", default="comparison_results", help="Output directory")
    ap.add_argument("--ks", nargs="+", type=int, default=[10, 20, 50, 100, 200, 500])
    ap.add_argument("--repeats", type=int, default=5, help="How many runs with different seeds")
    ap.add_argument("--seed", type=int, default=0, help="Base seed")
    ap.add_argument("--smart", choices=["round_robin", "fft"], default="round_robin")
    ap.add_argument("--weight_attr", default="length")
    ap.add_argument("--eval_subset", type=int, default=0,
                    help="If >0, approximate global coverage on random subset of this size (faster).")

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    plot_dir = os.path.join(args.outdir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    full_pkg = load_graph_pkg(args.full)
    pruned_pkg = load_graph_pkg(args.pruned)
    full_g = full_pkg["graph"]
    pruned_g = pruned_pkg["graph"]

    if "community" not in pruned_g.vs.attribute_names():
        raise ValueError("Pruned graph must have vertex attribute 'community' (run Leiden first).")

    pruned_osmid_to_idx = build_osmid_to_index(pruned_g)
    labels = np.asarray(pruned_g.vs["community"], dtype=int)

    logger.info(f"Full graph:   n={full_g.vcount():,}, m={full_g.ecount():,}")
    logger.info(f"Pruned graph: n={pruned_g.vcount():,}, m={pruned_g.ecount():,}")

    # Precompute CSR once (major speedup) :contentReference[oaicite:7]{index=7}
    indptr, indices, data = build_csr_adjacency(pruned_g, weight_attr=args.weight_attr)

    # Optional eval subset
    eval_subset = None
    if args.eval_subset and 0 < args.eval_subset < pruned_g.vcount():
        rng0 = np.random.default_rng(args.seed)
        eval_subset = rng0.choice(pruned_g.vcount(), size=args.eval_subset, replace=False)
        logger.info(f"Using eval_subset of size {len(eval_subset):,} for global coverage approx.")

    rows = []

    for r in range(args.repeats):
        run_seed = args.seed + r
        logger.info(f"=== Repeat {r+1}/{args.repeats} seed={run_seed} ===")

        for k in args.ks:
            # 1) Dummy 1: FULL random (efficiency may be < 1)
            s_full = algo_full_random(full_g, k, run_seed)
            mid_full, k_eff_full, eff_full = evaluate_on_pruned(
                pruned_g, pruned_osmid_to_idx, s_full, labels,
                indptr, indices, data,
                weight_attr=args.weight_attr,
                eval_subset=eval_subset,
            )
            flat_full = flatten_metrics(mid_full)
            rows.append({
                "algo": "Dummy 1 (Full Random)",
                "k": int(k),
                "repeat": int(r),
                "seed": int(run_seed),
                "k_eff": int(k_eff_full),
                "efficiency": float(eff_full),
                **flat_full,
            })

            # 2) Dummy 2: PRUNED random (efficiency = 1)
            s_pr = algo_pruned_random(pruned_g, k, run_seed)
            mid_pr, k_eff_pr, eff_pr = evaluate_on_pruned(
                pruned_g, pruned_osmid_to_idx, s_pr, labels,
                indptr, indices, data,
                weight_attr=args.weight_attr,
                eval_subset=eval_subset,
            )
            flat_pr = flatten_metrics(mid_pr)
            rows.append({
                "algo": "Dummy 2 (Pruned Random)",
                "k": int(k),
                "repeat": int(r),
                "seed": int(run_seed),
                "k_eff": int(k_eff_pr),
                "efficiency": float(1.0),
                **flat_pr,
            })

            # 3) Pipeline smart on PRUNED (efficiency = 1)
            s_sm = algo_pipeline_smart(pruned_g, k, run_seed, method=args.smart, weight_attr=args.weight_attr)
            mid_sm, k_eff_sm, eff_sm = evaluate_on_pruned(
                pruned_g, pruned_osmid_to_idx, s_sm, labels,
                indptr, indices, data,
                weight_attr=args.weight_attr,
                eval_subset=eval_subset,
            )
            flat_sm = flatten_metrics(mid_sm)
            rows.append({
                "algo": f"Pipeline (Smart: {args.smart})",
                "k": int(k),
                "repeat": int(r),
                "seed": int(run_seed),
                "k_eff": int(k_eff_sm),
                "efficiency": float(1.0),
                **flat_sm,
            })

            logger.info(
                f"k={k:4d} | eff(full)={eff_full:.2f} "
                f"| cov(smart)={flat_sm['cov']:.3f} "
                f"| Rmax(smart)={flat_sm['global_R_max']:.1f} "
                f"| minsep(smart)={flat_sm['div_min_sep']:.1f}"
            )

    df = pd.DataFrame(rows)
    raw_csv = os.path.join(args.outdir, "results_raw.csv")
    df.to_csv(raw_csv, index=False)
    logger.info(f"Saved raw results: {raw_csv}")

    # Aggregate mean/std over repeats
    metrics = [
        "efficiency", "cov", "balance_cv", "balance_entropy",
        "global_R_max", "global_mean", "global_median", "global_p90",
        "div_min_sep", "div_mean", "div_median", "div_p10", "div_p90",
    ]
    grp = df.groupby(["algo", "k"], as_index=False)
    agg_mean = grp[metrics].mean().rename(columns={m: f"{m}_mean" for m in metrics})
    agg_std  = grp[metrics].std(ddof=0).rename(columns={m: f"{m}_std" for m in metrics})

    df_agg = pd.merge(agg_mean, agg_std, on=["algo", "k"], how="inner")
    agg_csv = os.path.join(args.outdir, "results_agg.csv")
    df_agg.to_csv(agg_csv, index=False)
    logger.info(f"Saved aggregated results: {agg_csv}")

    # Plots (midterm metrics)
    plot_metric_with_errorbars(df_agg, "efficiency", "Urban Efficiency (0-1)", "Urban Efficiency vs k",
                              os.path.join(plot_dir, "efficiency.png"))
    plot_metric_with_errorbars(df_agg, "cov", "Community Coverage (0-1)", "Community Coverage vs k",
                              os.path.join(plot_dir, "coverage.png"))
    plot_metric_with_errorbars(df_agg, "global_R_max", "R_max (graph distance)", "Global Coverage (R_max) vs k (Lower is Better)",
                              os.path.join(plot_dir, "global_R_max.png"))
    plot_metric_with_errorbars(df_agg, "global_p90", "p90 d(x,S) (graph distance)", "Global Coverage (p90) vs k (Lower is Better)",
                              os.path.join(plot_dir, "global_p90.png"))
    plot_metric_with_errorbars(df_agg, "div_min_sep", "Min separation (graph distance)", "Diversity (Min Sep) vs k (Higher is Better)",
                              os.path.join(plot_dir, "div_min_sep.png"))
    plot_metric_with_errorbars(df_agg, "balance_cv", "CV over community counts", "Balance (CV) vs k (Lower is Better)",
                              os.path.join(plot_dir, "balance_cv.png"))
    plot_metric_with_errorbars(df_agg, "balance_entropy", "Normalized entropy (0-1)", "Balance (Entropy) vs k (Higher is Better)",
                              os.path.join(plot_dir, "balance_entropy.png"))

    logger.info(f"Saved plots in: {plot_dir}")


if __name__ == "__main__":
    main()
