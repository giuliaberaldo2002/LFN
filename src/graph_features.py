"""
graph_features.py
-----------------
Shared module for computing node-level features used in urbanity scoring.
This library provides functions to extract both tag-based and topological features
from an igraph object.

Features Computed:
  1. Tag-based (from 'highway', 'maxspeed' edge attributes):
     - Frequency distributions of highway types (residential, primary, motorway, service, other).
     - Average maxspeed of incident edges.
  2. Topological:
     - Degree.
     - Clustering coefficient (local transitivity).
     - Average edge length of incident edges.

Usage:
  - Imported by `compute_features.py`.
  - Functions modify the graph in-place by adding vertex attributes.
"""

import math
import logging
from collections import Counter
from typing import Any, Optional

import numpy as np
import igraph as ig

logger = logging.getLogger(__name__)

# -----------------------
# Constants
# -----------------------
MAIN_HIGHWAY_TYPES = {"residential", "primary", "motorway", "service"}

# -----------------------
# Helpers
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

# -----------------------
# Core Functions
# -----------------------
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
