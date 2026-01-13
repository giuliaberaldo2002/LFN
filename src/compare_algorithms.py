
#!/usr/bin/env python3
"""
compare_algorithms.py
--------------------
Compare sampling methods using MidTerm metrics (from sampling.py) on a PRUNED graph.

We compare:
  - Dummy 1: Full Random (sample from full graph; evaluate only the hits in pruned)
  - Dummy 2: Pruned Random (random within pruned)
  - Pipeline (Smart): round_robin OR fft OR hybrid (or all)

Metrics (Midterm-aligned):
  - efficiency (urban hit-rate for full-random; for pruned methods it's 1.0 but we also report k_eff)
  - community coverage
  - global coverage: R_max, mean, median, p90 of d(x,S)
  - diversity: min_sep, mean, median, p10, p90 of pairwise d(si,sj)
  - balance: cv, entropy_norm
  - optional: ARI/NMI

Outputs:
  - results_raw.csv (one row per (k, algo, repeat))
  - results_agg.csv (mean/std aggregated over repeats)
  - plots/*.png (metric vs k with error bars)
"""

import argparse
import os
import pickle
import logging
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sampling import (
    sample_round_robin,
    fft_sample_graph,
    hybrid_city_ordered_round_robin,
    compute_fft_city_order_optimized,
    evaluate_midterm,
)
from fix_connectivity import connect_graph_components

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
# Algorithms (return OSMIDs)
# -------------------------

def algo_full_random(full_g, k: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    n = full_g.vcount()
    if n == 0 or k <= 0:
        return []
    idx = rng.choice(n, size=min(k, n), replace=False)
    osmids = get_osmid_array(full_g)
    return [int(osmids[i]) for i in idx]

def algo_pruned_random(pruned_g, k: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    n = pruned_g.vcount()
    if n == 0 or k <= 0:
        return []
    idx = rng.choice(n, size=min(k, n), replace=False)
    osmids = get_osmid_array(pruned_g)
    return [int(osmids[i]) for i in idx]

def algo_pipeline_smart(pruned_g, k: int, seed: int, method: str, weight_attr: str, cached_city_order: Optional[List[int]] = None) -> List[int]:
    """
    Smart sampling on PRUNED graph.
    method: 'round_robin' | 'fft' | 'hybrid'
    """
    if k <= 0 or pruned_g.vcount() == 0:
        return []

    if method == "round_robin":
        if "community" not in pruned_g.vs.attribute_names():
            raise ValueError("round_robin requires vertex attribute 'community' in pruned graph.")
        idx = sample_round_robin(pruned_g, k, seed=seed)  # must be no-duplicates
    elif method == "fft":
        idx = fft_sample_graph(pruned_g, k, weight_attr=weight_attr, seed_idx=None)
    elif method == "hybrid":
        idx = hybrid_city_ordered_round_robin(
            pruned_g, k, seed=seed, weight_attr=weight_attr,
            cached_city_order=cached_city_order
        )
    else:
        raise ValueError(f"Unknown smart method: {method}")

    osmids = get_osmid_array(pruned_g)
    return [int(osmids[i]) for i in idx]


# -------------------------
# Evaluation wrapper (Midterm metrics)
# -------------------------

def map_osmids_to_pruned_indices(
    pruned_osmid_to_idx: Dict[int, int],
    sampled_osmids: List[int],
) -> List[int]:
    """
    Map sampled OSMIDs to pruned internal indices, removing duplicates while preserving order.
    """
    out = []
    seen = set()
    for oid in sampled_osmids:
        if oid in pruned_osmid_to_idx:
            j = pruned_osmid_to_idx[oid]
            if j not in seen:
                seen.add(j)
                out.append(int(j))
    return out

def flatten_metrics(mid: Dict[str, Any]) -> Dict[str, Any]:
    cov = mid["community_coverage"]
    bal = mid["balance"]
    glob = mid["global_coverage"]
    div = mid["diversity"]
    agree = mid.get("clustering_agreement", {})

    return {
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

def evaluate_on_pruned(
    pruned_g,
    pruned_osmid_to_idx: Dict[int, int],
    sampled_osmids: List[int],
    labels_full: np.ndarray,
    weight_attr: str,
    eval_subset: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, Any], int, float]:
    """
    Evaluate a sampled set (OSMIDs) on the pruned graph with midterm metrics.
    Returns:
      - midterm_metrics (dict)
      - k_eff (# of valid unique hits in pruned)
      - efficiency (k_eff / k_requested)
    """
    k_req = len(sampled_osmids)
    valid_idx = map_osmids_to_pruned_indices(pruned_osmid_to_idx, sampled_osmids)
    k_eff = len(valid_idx)
    eff = (k_eff / k_req) if k_req > 0 else 0.0

    if k_eff == 0:
        # safe empty metrics
        empty = {
            "community_coverage": {"coverage": 0.0, "n_communities_full": int(len(np.unique(labels_full))), "n_communities_represented": 0, "counts_per_community": {}},
            "balance": {"cv": float("nan"), "entropy_norm": float("nan")},
            "global_coverage": {"R_max": float("inf"), "mean": float("inf"), "median": float("inf"), "p90": float("inf")},
            "diversity": {"min_sep": 0.0, "mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0},
            "clustering_agreement": {"ari": None, "nmi": None},
        }
        return empty, k_eff, eff

    mid = evaluate_midterm(
        pruned_g,
        labels_full,
        valid_idx,
        eval_subset=eval_subset,
        weight_attr=weight_attr,
    )
    return mid, k_eff, eff


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
    ap.add_argument("--smart", choices=["round_robin", "fft", "hybrid", "all"], default="all",
                    help="Which smart sampler(s) to run on pruned graph")
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

    # FIX: Ensure graph is connected for metric calculation (virtual edges)
    pruned_g = connect_graph_components(pruned_g, weight_attr=args.weight_attr)

    if "community" not in pruned_g.vs.attribute_names():
        raise ValueError("Pruned graph must have vertex attribute 'community' (run Leiden first).")

    pruned_osmid_to_idx = build_osmid_to_index(pruned_g)
    labels = np.asarray(pruned_g.vs["community"], dtype=int)

    logger.info(f"Full graph:   n={full_g.vcount():,}, m={full_g.ecount():,}")
    logger.info(f"Pruned graph: n={pruned_g.vcount():,}, m={pruned_g.ecount():,}")

    # Precompute CSR once (major speedup) -> Replaced by igraph
    # indptr, indices, data = build_csr_adjacency(pruned_g, weight_attr=args.weight_attr)

    # Optional eval subset
    eval_subset = None
    if args.eval_subset and 0 < args.eval_subset < pruned_g.vcount():
        rng0 = np.random.default_rng(args.seed)
        eval_subset = rng0.choice(pruned_g.vcount(), size=args.eval_subset, replace=False)
        logger.info(f"Using eval_subset of size {len(eval_subset):,} for global coverage approx.")

    smart_list = ["round_robin", "fft", "hybrid"] if args.smart == "all" else [args.smart]

    rows = []

    for r in range(args.repeats):
        run_seed = args.seed + r
        logger.info(f"=== Repeat {r+1}/{args.repeats} seed={run_seed} ===")
        
        # Precompute city order for 'hybrid' if needed
        cached_order = None
        if "hybrid" in smart_list:
            logger.info("  Precomputing community FFT order (optimized)...")
            cached_order = compute_fft_city_order_optimized(pruned_g, weight_attr=args.weight_attr, seed=run_seed)

        for k in args.ks:
            # 1) Dummy 1: FULL random
            logger.info(f"  k={k}: Running Dummy 1 (Full Random)")  ### NEW LOG
            s_full = algo_full_random(full_g, k, run_seed)
            mid_full, k_eff_full, eff_full = evaluate_on_pruned(
                pruned_g, pruned_osmid_to_idx, s_full, labels,
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

            # 2) Dummy 2: PRUNED random
            logger.info(f"  k={k}: Running Dummy 2 (Pruned Random)") ### NEW LOG
            s_pr = algo_pruned_random(pruned_g, k, run_seed)
            mid_pr, k_eff_pr, eff_pr = evaluate_on_pruned(
                pruned_g, pruned_osmid_to_idx, s_pr, labels,
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

            # 3) Pipeline smart(s) on PRUNED
            for sm in smart_list:
                logger.info(f"    ... running Pipeline Smart ({sm}) k={k}") ### NEW LOG
                
                # Se 'sm' è 'hybrid', qui potrebbe impiegarci vari minuti per l'inizializzazione!
                s_sm = algo_pipeline_smart(
                    pruned_g, k, run_seed, method=sm,
                    weight_attr=args.weight_attr,
                    cached_city_order=cached_order
                )
                
                mid_sm, k_eff_sm, eff_sm = evaluate_on_pruned(
                    pruned_g, pruned_osmid_to_idx, s_sm, labels,
                    weight_attr=args.weight_attr,
                    eval_subset=eval_subset,
                )
                flat_sm = flatten_metrics(mid_sm)
                rows.append({
                    "algo": f"Pipeline (Smart: {sm})",
                    "k": int(k),
                    "repeat": int(r),
                    "seed": int(run_seed),
                    "k_eff": int(k_eff_sm),
                    "efficiency": float(1.0),
                    **flat_sm,
                })

            logger.info(
                f"k={k:4d} DONE | eff(full)={eff_full:.2f} | cov(pr)={flat_pr['cov']:.3f}"
            )

    df = pd.DataFrame(rows)
    raw_csv = os.path.join(args.outdir, "results_raw.csv")
    df.to_csv(raw_csv, index=False)
    logger.info(f"Saved raw results: {raw_csv}")

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
    plot_metric_with_errorbars(df_agg, "div_p10", "p10 pairwise distance", "Diversity (p10) vs k (Higher is Better)",
                              os.path.join(plot_dir, "div_p10.png"))
    plot_metric_with_errorbars(df_agg, "balance_cv", "CV over community counts", "Balance (CV) vs k (Lower is Better)",
                              os.path.join(plot_dir, "balance_cv.png"))
    plot_metric_with_errorbars(df_agg, "balance_entropy", "Normalized entropy (0-1)", "Balance (Entropy) vs k (Higher is Better)",
                              os.path.join(plot_dir, "balance_entropy.png"))

    logger.info(f"Saved plots in: {plot_dir}")


if __name__ == "__main__":
    main()
