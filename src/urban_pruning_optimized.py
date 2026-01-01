"""
Urbanity Pruning & Scoring Module
---------------------------------
Computes topological and tag-based features for an OSM road network,
applies PCA to learn an urbanity score, and uses DBSCAN with automated parameter 
selection ('Elbow Method') to prune rural nodes.

Optimizations:
- Smart DBSCAN: Automated 'eps' detection using NearestNeighbors (Elbow Method).
- PCA Sign Correction: Ensures 'High Score' correlates with 'Degree' (Urban).
- Memory-aware feature computation.

Author: [Assistant]
"""

import math
import os
import pickle
import sys
import logging
from collections import Counter
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import igraph as ig
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
# Try to import kneed for elbow detection, else fallback to simple method
try:
    from kneed import KneeLocator
    HAS_KNEED = True
except ImportError:
    HAS_KNEED = False

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
INPUT_PKL = "bremen_processed_graph.pkl"
OUTPUT_PRUNED_PKL = "bremen_pruned_graph.pkl"
MAIN_HIGHWAY_TYPES = {"residential", "primary", "motorway", "service"}

def load_graph_data(filepath: str) -> Tuple[ig.Graph, Dict[int, int]]:
    """Loads the igraph object and ID map from pickle."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"{filepath} not found. Run graph_init_optimized.py first.")
        
    logger.info(f"Loading graph from {filepath}...")
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data["graph"], data.get("osmid_map", {})

def parse_maxspeed(val: Any) -> Optional[float]:
    """Parses maxspeed tag values into float or None."""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.lower().strip()
        if val in {"none", "signals", "variable", "walk"}:
            return None
        # Extract first numeric sequence
        import re
        m = re.search(r"(\d+(\.\d+)?)", val)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
    return None

def add_tag_based_features(g: ig.Graph) -> None:
    """Computes and adds tag-based features (highway freq, avg speed) to nodes."""
    logger.info("Computing tag-based features...")
    
    es_attrs = g.es.attribute_names()
    has_highway = "highway" in es_attrs
    has_maxspeed = "maxspeed" in es_attrs
    
    # Pre-allocate lists for speed
    freq_results = {k: np.zeros(g.vcount()) for k in ["residential", "primary", "motorway", "service", "other"]}
    avg_speeds = np.zeros(g.vcount())
    
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
                # Convert list to single string if necessary (though we flattened in init, let's be safe)
                if isinstance(hw, list):
                    hw = hw[0]
                hw_types.append(str(hw) if hw else "other")
            else:
                hw_types.append("other")
                
            if has_maxspeed:
                sp = parse_maxspeed(e["maxspeed"])
                if sp is not None:
                    speeds.append(sp)
        
        # Frequencies
        total = len(hw_types)
        counts = Counter(hw_types)
        
        freq_results["residential"][v] = counts.get("residential", 0) / total
        freq_results["primary"][v] = counts.get("primary", 0) / total
        freq_results["motorway"][v] = counts.get("motorway", 0) / total
        freq_results["service"][v] = counts.get("service", 0) / total
        
        other_count = sum(c for k, c in counts.items() if k not in MAIN_HIGHWAY_TYPES)
        freq_results["other"][v] = other_count / total
        
        # Speed
        if speeds:
            avg_speeds[v] = np.mean(speeds)
            
    # Assign attributes in bulk
    for k, vals in freq_results.items():
        g.vs[f"freq_{k}"] = vals.tolist()
    g.vs["avg_maxspeed"] = avg_speeds.tolist()

def compute_topology_features(g: ig.Graph) -> None:
    """Computes topological features: Degree, Clustering, Betweenness, Avg Edge Length."""
    logger.info("Computing topological features...")
    
    # 1. Degree
    g.vs["degree"] = g.degree()
    
    # 2. Clustering (Local Transitivity)
    # Handle NaNs (isolated nodes return NaN usually)
    clust = g.transitivity_local_undirected()
    g.vs["clustering_coeff"] = [c if not math.isnan(c) else 0.0 for c in clust]
    
    # 3. Betweenness (Expensive!)
    # Using 'length' as weight if available, else None (unweighted)
    weights = g.es["length"] if "length" in g.es.attribute_names() else None
    g.vs["betweenness"] = g.betweenness(weights=weights)
    
    # 4. Avg Edge Length
    # Vectorized approach usually hard on jagged arrays, loop is fine for millions
    avg_lens = []
    if weights:
        for v in range(g.vcount()):
            edges = g.incident(v)
            if edges:
                # get attribute values corresponding to edge indices
                # igraph sequence indexing is fast
                lens = [weights[e] for e in edges]
                avg_lens.append(np.mean(lens))
            else:
                avg_lens.append(0.0)
    else:
        avg_lens = [0.0] * g.vcount()
        
    g.vs["avg_edge_len"] = avg_lens

def find_optimal_eps(X: np.ndarray, k: int = 10) -> float:
    """
    Finds optimal eps for DBSCAN using k-distance graph elbow method.
    Returns: recommended eps value.
    """
    logger.info("Estimating optimal DBSCAN eps using K-Distance Elbow...")
    
    # Limit sample size for NN if dataset is huge (e.g. > 100k) to save time
    # For full accuracy we use all, but memory might be tight.
    # We'll use all for now as user has 2TB RAM (cluster).
    
    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(X)
    distances, _ = nbrs.kneighbors(X)
    
    # 1. Sort distances to the k-th nearest neighbor
    k_distances = np.sort(distances[:, k-1])
    
    # 2. Find Elbow
    if HAS_KNEED:
        # Use KneeLocator if available
        x_axis = range(len(k_distances))
        kneedle = KneeLocator(x_axis, k_distances, curve="convex", direction="increasing")
        optimal_eps = kneedle.knee_y
        if optimal_eps is None:
             optimal_eps = np.percentile(k_distances, 90) # Fallback
        logger.info(f"Kneelocator found eps={optimal_eps:.4f}")
    else:
        # Simple derivative-based fallback or percentile
        # We assume the "elbow" is where curvature is high. 
        # A robust crude heuristic is the 90th-95th percentile for pruning tasks
        # but let's try a simple geometry method: distance from line connecting start-end.
        logger.warning("'kneed' library not found. Using geometric distance method.")
        
        # Normalize curve
        y = k_distances
        x = np.arange(len(y))
        
        # Line from (0, y[0]) to (N, y[-1])
        p1 = np.array([0, y[0]])
        p2 = np.array([len(y)-1, y[-1]])
        
        # Distance of each point to the line
        denom = np.linalg.norm(p2 - p1)
        if denom == 0:
            optimal_eps = 0.5
        else:
            numer = np.abs(np.cross(p2-p1, p1 - np.c_[x, y]))
            dist_to_line = numer / denom
            idx_max = np.argmax(dist_to_line)
            optimal_eps = y[idx_max]
            
        logger.info(f"Geometric method found eps={optimal_eps:.4f}")
        
    return float(optimal_eps)

def main():
    logger.info("--- START JOB: Urban Pruning Optimized ---")
    
    # 1. LOAD DATA
    try:
        g, osmid_map = load_graph_data(INPUT_PKL)
    except Exception as e:
        logger.error(e)
        return
        
    logger.info(f"Nodes: {g.vcount():,}, Edges: {g.ecount():,}")

    # 2. COMPUTE FEATURES
    add_tag_based_features(g)
    compute_topology_features(g)
    
    feature_names = [
        "degree", "avg_edge_len", "betweenness", "clustering_coeff",
        "freq_residential", "freq_primary", "freq_motorway", 
        "freq_service", "freq_other", "avg_maxspeed"
    ]
    
    # Extract X matrix
    data_matrix = []
    for feat in feature_names:
        col = np.array(g.vs[feat])
        col[np.isnan(col)] = 0.0 # Safety fill
        data_matrix.append(col)
        
    X = np.column_stack(data_matrix)
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 3. PCA & WEIGHT LEARNING
    logger.info("Running PCA (1 component)...")
    pca = PCA(n_components=1)
    pca.fit(X_scaled)
    
    weights = pca.components_[0]
    
    # Check correlation with 'degree' (Must be positive for Urban)
    deg_idx = feature_names.index("degree")
    if weights[deg_idx] < 0:
        logger.info("PCA component inverted relative to Degree. Flipping signs.")
        weights = -weights
        
    logger.info("Learned Weights:")
    for name, w in zip(feature_names, weights):
        logger.info(f"   {name:<20}: {w:.4f}")
        
    # Calculate Score
    scores = X_scaled @ weights
    g.vs["urbanity_score"] = scores
    
    # 4. CLUSTERING & PRUNING (SMART DBSCAN)
    # Using 'min_samples' ~ ln(N) or fixed small number. 
    # For large graphs, min_samples=10-20 is reasonable to avoid noise.
    optimal_eps = find_optimal_eps(X_scaled, k=10)
    
    logger.info(f"Running DBSCAN with eps={optimal_eps:.4f}, min_samples=10...")
    db = DBSCAN(eps=optimal_eps, min_samples=10, n_jobs=-1).fit(X_scaled)
    labels = db.labels_
    g.vs["cluster_label"] = labels
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1) # Remove noise
        
    if not unique_labels:
        logger.warning("DBSCAN found only noise. Falling back to percentile pruning.")
        cutoff = np.percentile(scores, 25)
        keep_indices = [i for i, s in enumerate(scores) if s >= cutoff]
    else:
        # Evaluate clusters
        cluster_avgs = {}
        for L in unique_labels:
            mask = (labels == L)
            cluster_avgs[L] = np.mean(scores[mask])
            
        # Identify Rural Cluster (Lowest Average Score)
        rural_label = min(cluster_avgs, key=cluster_avgs.get)
        logger.info(f"Clusters found: {len(unique_labels)}. Rural Cluster Label: {rural_label} (Score: {cluster_avgs[rural_label]:.2f})")
        
        # Keep everything NOT in rural cluster (and decided what to do with noise -1)
        # Usually noise in DBSCAN means low density. 
        # Depending on context, noise could be 'super rural' or 'outliers'.
        # Safest: Prune rural cluster only.
        keep_indices = [i for i, L in enumerate(labels) if L != rural_label]
        
    # Create Subgraph
    g_pruned = g.subgraph(keep_indices)
    
    # Update Map
    new_osmid_map = {new_i: osmid_map[old_i] for new_i, old_i in enumerate(keep_indices) if old_i in osmid_map}
    
    logger.info(f"Original: {g.vcount():,}, Pruned: {g_pruned.vcount():,}")
    logger.info(f"Nodes removed: {g.vcount() - g_pruned.vcount():,}")
    
    # 5. SAVE
    package = {
        "graph": g_pruned,
        "osmid_map": new_osmid_map,
        "crs": "epsg:4326"
    }
    with open(OUTPUT_PRUNED_PKL, "wb") as f:
        pickle.dump(package, f)
        
    logger.info(f"Saved pruned graph to {os.path.abspath(OUTPUT_PRUNED_PKL)}")

if __name__ == "__main__":
    main()
