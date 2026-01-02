"""
Graph Features Computation (Vectorized)
---------------------------------------
Shared logic to compute node features for urbanity scoring.
Extracts:
- Tag-based features (highway frequencies, avg_maxspeed)
- Topological features (degree, clustering coefficient, avg edge length)

Optimized with NumPy to avoid Python loops over nodes/edges.
"""

import math
import logging
import numpy as np
import igraph as ig

logger = logging.getLogger(__name__)

# -----------------------
# Constants
# -----------------------
MAIN_HIGHWAY_TYPES = ["residential", "primary", "motorway", "service"]


# -----------------------
# Helpers
# -----------------------
def parse_maxspeed_vectorized(values: list) -> np.ndarray:
    """
    Parses a list of maxspeed values (strings/ints/floats/None) into a float array.
    Invalid or missing values become NaN.
    """
    import re
    
    # Pre-compile regex
    num_re = re.compile(r"(\d+(\.\d+)?)")
    
    out = np.full(len(values), np.nan, dtype=np.float32)
    
    for i, val in enumerate(values):
        if val is None:
            continue
        if isinstance(val, (int, float)):
            out[i] = float(val)
        elif isinstance(val, str):
            v_str = val.lower().strip()
            if v_str in {"none", "signals", "variable", "walk"}:
                continue
            m = num_re.search(v_str)
            if m:
                try:
                    out[i] = float(m.group(1))
                except ValueError:
                    pass
    return out


# -----------------------
# Core Functions
# -----------------------
def add_tag_based_features(g: ig.Graph) -> None:
    """
    Computes highway type frequencies and avg maxspeed for each node.
    Uses vectorized operations for performance.
    """
    logger.info("Computing tag-based features (Vectorized)...")
    
    num_nodes = g.vcount()
    num_edges = g.ecount()
    
    if num_edges == 0:
        for k in MAIN_HIGHWAY_TYPES + ["other"]:
            g.vs[f"freq_{k}"] = [0.0] * num_nodes
        g.vs["avg_maxspeed"] = [0.0] * num_nodes
        return

    # Get edge sources and targets (undirected/incident logic)
    # iGraph stores edges as (source, target). 
    # For 'incident' (ALL), we care about both endpoints.
    edges = np.array(g.get_edgelist(), dtype=np.int32)
    u = edges[:, 0]
    v = edges[:, 1]
    
    # --- 1. Highway Frequencies ---
    
    # Get highway attributes (list of strings/lists/None)
    # We normalize them first
    raw_hw = g.es["highway"]
    hw_normalized = []
    for h in raw_hw:
        if isinstance(h, list):
            hw_normalized.append(str(h[0]) if h else "other")
        elif h is None:
            hw_normalized.append("other")
        else:
            hw_normalized.append(str(h))
            
    hw_arr = np.array(hw_normalized)

    # Initialize counters for each type
    # counts[type][node_idx]
    counts = {k: np.zeros(num_nodes, dtype=np.float32) for k in MAIN_HIGHWAY_TYPES}
    counts["other"] = np.zeros(num_nodes, dtype=np.float32)
    
    # Accumulate
    # We do two add.at calls per type: one for u, one for v (incident edges)
    for k in MAIN_HIGHWAY_TYPES:
        is_type = (hw_arr == k)
        # If 'residential', add 1 to u and v indices where edge is residential
        # Optimization: boolean array `is_type` acts as mask, but `add.at` needs indices.
        # Faster: `np.add.at(counts[k], u, is_type.astype(float))`
        # Because we want to add 1 where true, 0 where false.
        val_to_add = is_type.astype(np.float32)
        np.add.at(counts[k], u, val_to_add)
        np.add.at(counts[k], v, val_to_add)
        
    # For 'other', it's anything not in MAIN
    is_main = np.isin(hw_arr, list(MAIN_HIGHWAY_TYPES))
    is_other = ~is_main
    val_other = is_other.astype(np.float32)
    np.add.at(counts["other"], u, val_other)
    np.add.at(counts["other"], v, val_other)
    
    # Normalize by degree (total incident edges)
    degree = np.zeros(num_nodes, dtype=np.float32)
    # Just sum all counts to get degree (considering only edges with highway attr? 
    # Usually all edges. Let's use actual degree to be safe or sum of computed counts.
    # Sum of computed counts IS the degree in terms of classified edges)
    total_counts = sum(counts.values())
    
    # Avoid div by zero
    total_counts[total_counts == 0] = 1.0
    
    for k in counts:
        freq = counts[k] / total_counts
        g.vs[f"freq_{k}"] = freq.tolist()
        
    # --- 2. Avg Maxspeed ---
    
    if "maxspeed" in g.es.attribute_names():
        raw_speeds = g.es["maxspeed"]
        speed_vals = parse_maxspeed_vectorized(raw_speeds)
        
        # We only care about edges with valid speeds
        mask_valid = ~np.isnan(speed_vals)
        
        if np.any(mask_valid):
            valid_speeds = speed_vals[mask_valid]
            valid_u = u[mask_valid]
            valid_v = v[mask_valid]
            
            sum_speeds = np.zeros(num_nodes, dtype=np.float32)
            count_speeds = np.zeros(num_nodes, dtype=np.float32)
            
            np.add.at(sum_speeds, valid_u, valid_speeds)
            np.add.at(sum_speeds, valid_v, valid_speeds)
            
            np.add.at(count_speeds, valid_u, 1)
            np.add.at(count_speeds, valid_v, 1)
            
            # Avoid div 0
            count_speeds[count_speeds == 0] = 1.0
            avg_speeds = sum_speeds / count_speeds
            # Reset 0-count nodes to 0.0 (or whatever default)
            avg_speeds[sum_speeds == 0] = 0.0
            
            g.vs["avg_maxspeed"] = avg_speeds.tolist()
        else:
             g.vs["avg_maxspeed"] = [0.0] * num_nodes
    else:
        g.vs["avg_maxspeed"] = [0.0] * num_nodes


def compute_topology_features(g: ig.Graph) -> None:
    """
    Computes degree, clustering coefficient, and avg edge length.
    Vectorized where possible.
    """
    logger.info("Computing topological features (Vectorized)...")
    
    # 1. Degree
    g.vs["degree"] = g.degree()

    # 2. Clustering Coefficient (Local Transitivity)
    # iGraph's C implementation is already optimized
    clust = g.transitivity_local_undirected()
    # Handle NaNs (isolated nodes or degree < 2)
    g.vs["clustering_coeff"] = np.nan_to_num(clust, nan=0.0).tolist()

    # 3. Avg Edge Length
    if "length" not in g.es.attribute_names():
        g.vs["avg_edge_len"] = [0.0] * g.vcount()
        return

    lengths = np.array(g.es["length"], dtype=np.float32)
    # Replace any NaNs with 0
    lengths = np.nan_to_num(lengths, nan=0.0)
    
    edges = np.array(g.get_edgelist(), dtype=np.int32)
    u = edges[:, 0]
    v = edges[:, 1]
    
    num_nodes = g.vcount()
    sum_len = np.zeros(num_nodes, dtype=np.float32)
    
    # Accumulate lengths to both endpoints
    np.add.at(sum_len, u, lengths)
    np.add.at(sum_len, v, lengths)
    
    # Divide by degree
    deg = np.array(g.degree(), dtype=np.float32)
    deg[deg == 0] = 1.0
    
    avg_len = sum_len / deg
    g.vs["avg_edge_len"] = avg_len.tolist()
