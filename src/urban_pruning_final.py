"""
urban_pruning_final.py
----------------------
Step 3 in the pipeline (after feature computation).
Computes urbanity scores for all nodes and prunes the graph to retain only "urban" areas.

Logic:
  1. Loads the graph with features (from compute_features.py).
  2. Loads the tuned feature weights (from urbanity_tuning.py / urbanity_weights.json).
  3. Constructs the feature matrix X for valid features (log_degree, clustering, freq_residential, etc.).
  4. Standardizes X and computes the linear score: score = Z @ w.
  5. Fits a 2-component Gaussian Mixture Model (GMM) to the score distribution to find the "urban" cluster.
  6. Filters nodes where P(urban | score) > threshold (default 0.5).
  7. Saves the pruned graph.

Inputs:
  - --input: Pickle of graph with vertex features.
  - --weights: JSON containing feature names and weights.
  - --prob_threshold: Minimum probability to classify a node as urban (default 0.5).

Outputs:
  - Saves the pruned graph to --output.
"""

import os
import sys
import argparse
import json
import pickle
import logging
from typing import Tuple, Dict

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
# Constants / Defaults
# -----------------------
DEFAULT_INPUT_PKL = "bremen_graph_with_features.pkl"
DEFAULT_OUTPUT_PRUNED = "bremen_pruned_graph.pkl"
DEFAULT_WEIGHTS = "urbanity_weights.json"

MAIN_HIGHWAY_TYPES = {"residential", "primary", "motorway", "service"}


# -----------------------
# I/O Helper Functions
# -----------------------

def load_graph_data(filepath: str) -> Tuple[ig.Graph, Dict[int, int]]:
    """Loads feature-enriched graph from pickle."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found. Run graph_init_optimized.py first.")
    logger.info(f"Loading graph from {filepath}...")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["graph"], data.get("osmid_map", {})

def load_tuned_weights(json_path: str) -> Tuple[list[str], np.ndarray, dict]:
    """Loads feature weights metdata from JSON."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"{json_path} not found. Run urbanity_tuning.py first.")
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    names = obj["feature_names"]
    w = np.array(obj["weights"], dtype=np.float32)
    meta = obj.get("transforms", {})
    return names, w, meta


# -----------------------
# Feature Matrix Construction
# -----------------------

def _cap_percentile(x: np.ndarray, p: float) -> np.ndarray:
    """Outlier capping at percentile p."""
    x = x.copy()
    mask = np.isfinite(x)
    if not np.any(mask):
        return np.zeros_like(x)
    hi = np.percentile(x[mask], p)
    x[~mask] = 0.0
    return np.clip(x, None, hi)

def build_X_from_names(g: ig.Graph, feature_names: list[str], cap_p: float) -> np.ndarray:
    """
    Builds the feature matrix X columns in the order defined by the tuned weights.
    Applies necessary transforms (log, capping) to match the tuning phase.
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


# -----------------------
# GMM Scoring & Filtering
# -----------------------

def gmm_keep_indices(scores: np.ndarray, seed: int = 0, prob_threshold: float = 0.5) -> np.ndarray:
    """
    Fits a 2-component GMM on 1D scores and returns indices of nodes
    that belong to the 'Urban' cluster (classification by probability).
    """
    S = scores.reshape(-1, 1).astype(np.float32, copy=False)
    gmm = GaussianMixture(n_components=2, random_state=seed).fit(S)
    
    # Identify which component mean is higher (assumed to be Urban)
    means = gmm.means_.ravel()
    urban_label = int(np.argmax(means))
    
    proba_urban = gmm.predict_proba(S)[:, urban_label]
    keep = np.where(proba_urban >= prob_threshold)[0]
    
    logger.info(f"GMM means={means}, urban_label={urban_label}, keep={len(keep):,}/{len(scores):,}")
    return keep


# -----------------------
# Main Execution
# -----------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_PKL, help="Input graph (with features) pickle")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PRUNED, help="Output pruned graph pickle")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Input weights JSON")
    parser.add_argument("--prob_threshold", type=float, default=0.5, help="GMM probability threshold")
    args = parser.parse_args()

    logger.info("--- START JOB: Urban Pruning (Manual Tuning + GMM) ---")

    g, osmid_map = load_graph_data(args.input)
    logger.info(f"Nodes: {g.vcount():,}, Edges: {g.ecount():,}")

    # 1. Load Tuned Weights
    tuned_names, w, meta = load_tuned_weights(args.weights)
    cap_p = float(meta.get("cap_percentile", 99.5))
    logger.info(f"Loaded tuned weights from {args.weights}")
    logger.info(f"Feature order: {tuned_names}")

    # 2. Build Feature Matrix X and Standardize
    # Matches the exact preprocessing used during tuning
    X = build_X_from_names(g, tuned_names, cap_p=cap_p)
    Z = StandardScaler().fit_transform(X).astype(np.float32, copy=False)

    # 3. Compute Linear Score
    scores = (Z @ w).astype(np.float32, copy=False)
    g.vs["urbanity_score"] = scores.tolist() # Save score to graph for reference

    # 4. Prune using GMM
    # Finds the high-score GMM component and keeps nodes belonging to it
    keep_indices = gmm_keep_indices(scores, seed=0, prob_threshold=args.prob_threshold)

    g_pruned = g.subgraph(keep_indices.tolist())
    new_osmid_map = {new_i: osmid_map[old_i] for new_i, old_i in enumerate(keep_indices) if old_i in osmid_map}

    logger.info(f"Original: {g.vcount():,}, Pruned: {g_pruned.vcount():,}")
    logger.info(f"Nodes removed: {g.vcount() - g_pruned.vcount():,}")

    package = {
        "graph": g_pruned,
        "osmid_map": new_osmid_map,
        "crs": "epsg:4326",
    }
    with open(args.output, "wb") as f:
        pickle.dump(package, f)

    logger.info(f"Saved pruned graph to {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
