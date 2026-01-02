"""
Urbanity Pruning (Manual Tuning + GMM)
--------------------------------------
Computes features for an OSM road network, applies a manually tuned urbanity score
(learned by constrained random search in urbanity_tuning.py), and prunes rural nodes
by fitting a 2-component GMM on the 1D score distribution.

Author: Giulia 
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
# Constants
# -----------------------
DEFAULT_INPUT_PKL = "bremen_graph_with_features.pkl"
DEFAULT_OUTPUT_PRUNED = "bremen_pruned_graph.pkl"
DEFAULT_WEIGHTS = "urbanity_weights.json"

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


# Features are now computed in compute_features.py using graph_features.py
# and loaded directly from bremen_graph_with_features.pkl


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT_PKL, help="Input graph (with features) pickle")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PRUNED, help="Output pruned graph pickle")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Input weights JSON")
    parser.add_argument("--prob_threshold", type=float, default=0.5, help="GMM probability threshold")
    args = parser.parse_args()

    logger.info("--- START JOB: Urban Pruning (Manual Tuning + GMM) ---")

    g, osmid_map = load_graph_data(args.input)
    logger.info(f"Nodes: {g.vcount():,}, Edges: {g.ecount():,}")

    # 1) compute features (already computed in compute_features.py)
    # add_tag_based_features(g)
    # compute_topology_features(g)

    # 2) load tuned weights
    tuned_names, w, meta = load_tuned_weights(args.weights)
    cap_p = float(meta.get("cap_percentile", 99.5))
    logger.info(f"Loaded tuned weights from {args.weights}")
    logger.info(f"Feature order: {tuned_names}")

    # 3) build X and standardize
    X = build_X_from_names(g, tuned_names, cap_p=cap_p)
    Z = StandardScaler().fit_transform(X).astype(np.float32, copy=False)

    # 4) score
    scores = (Z @ w).astype(np.float32, copy=False)
    g.vs["urbanity_score"] = scores.tolist()

    # 5) prune using GMM split on 1D scores
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
