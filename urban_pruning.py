import igraph as ig
import numpy as np
import pandas as pd
import math
import pickle
import time
import os
import re
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN

import sys

# ---------------- CONFIGURATION ------------------------
INPUT_PKL = "nord_est_processed_graph.pkl"
OUTPUT_PKL = "nord_est_pruned_graph.pkl"

# ---------------- HELPER FUNCTIONS ------------------------

def parse_maxspeed_vectorized(series):
    """
    Parses a pandas Series of maxspeed values (strings/numbers/nans).
    Returns a numpy array of floats (with NaN for missing/invalid).
    """
    def _parse_single(val):
        if pd.isna(val) or val is None:
            return np.nan
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            val_lower = val.lower()
            if val_lower in {"none", "signals", "variable", "implicit"}:
                return np.nan
            # Extract first number found
            m = re.search(r"(\d+(\.\d+)?)", val)
            if m:
                try:
                    return float(m.group(1))
                except ValueError:
                    return np.nan
        return np.nan

    # Apply the parsing
    return series.apply(_parse_single)

def find_knee_point(sorted_values):
    """
    Find the maximum curvature point in the k-distances curve.
    Based on the maximum distance of the line connecting the extremes.
    """
    n_points = len(sorted_values)
    if n_points < 2:
        return 0.1 # Fallback
        
    all_coords = np.vstack((range(n_points), sorted_values)).T
    first_point = all_coords[0]
    line_vec = all_coords[-1] - all_coords[0]
    norm_val = np.sum(line_vec**2)
    if norm_val == 0:
        return sorted_values[0]
        
    line_vec_norm = line_vec / np.sqrt(norm_val)
    vec_from_first = all_coords - first_point
    scalar_product = np.sum(vec_from_first * line_vec_norm, axis=1)
    vec_from_first_parallel = np.outer(scalar_product, line_vec_norm)
    vec_to_line = vec_from_first - vec_from_first_parallel
    dist_to_line = np.sqrt(np.sum(vec_to_line ** 2, axis=1))
    idx_of_best_point = np.argmax(dist_to_line)
    return sorted_values[idx_of_best_point]


# ---------------- MAIN EXECUTION ------------------------

def main():
    # Force flushing of stdout for real-time logging
    sys.stdout.reconfigure(line_buffering=True)

    print(f"--- START JOB: Urban Pruning (Optimized) ---")
    start_global = time.time()

    # 1. Load Graph
    print(f"\n[1/5] Loading Graph from {INPUT_PKL}...")
    if not os.path.exists(INPUT_PKL):
        print(f"Error: Could not find {INPUT_PKL}. Please run graph_init_corrected.py first.")
        return

    with open(INPUT_PKL, "rb") as f:
        data_package = pickle.load(f)
    
    g_ig = data_package["graph"]
    osmid_map = data_package.get("osmid_map", {})
    print(f"   Graph loaded: {g_ig.vcount():,} nodes, {g_ig.ecount():,} edges")

    # 2. Compute Basic Features (iGraph native)
    print("\n[2/5] Computing Basic Node Features...")
    t0 = time.time()
    
    # Degree
    g_ig.vs["degree"] = g_ig.degree()
    
    # Clustering Coefficient
    # handle None/NaNs by fillna(0)
    clustering = g_ig.transitivity_local_undirected()
    g_ig.vs["clustering_coeff"] = [0.0 if (c is None or np.isnan(c)) else c for c in clustering]

    # Betweenness - WARNING: Very slow for >100k nodes. 
    # For very large graphs, estimation or skipping is recommended.
    # Set to 0.0 effectively disabling it to save time as requested in previous logic.
    print("   Skipping exact betweenness calculation for performance (setting to 0.0).")
    g_ig.vs["betweenness"] = np.zeros(g_ig.vcount())
    
    print(f"   Basic features computed in {time.time()-t0:.2f}s")


    # 3. Compute Vectorized Tag-Based Features
    print("\n[3/5] Computing Tag-Based Features (Vectorized Aggregation)...")
    t0 = time.time()

    # Extract edge attributes to Pandas DataFrame
    edge_attrs = {}
    for attr in ["highway", "maxspeed", "length"]:
        if attr in g_ig.es.attribute_names():
            edge_attrs[attr] = g_ig.es[attr]
        else:
            edge_attrs[attr] = [None] * g_ig.ecount()
            print(f"   Warning: '{attr}' attribute missing, filling with None.")

    # Get edge endpoints (indices)
    edges = g_ig.get_edgelist()
    sources = [e[0] for e in edges]
    targets = [e[1] for e in edges]

    df_edges = pd.DataFrame(edge_attrs)
    df_edges["u"] = sources
    df_edges["v"] = targets

    # Normalize 'highway' column: list -> string, string -> string, None -> 'other'
    def normalize_highway(x):
        if isinstance(x, list):
            return x[0] if x else "other"
        if pd.isna(x):
            return "other"
        return str(x)
    
    df_edges["highway"] = df_edges["highway"].apply(normalize_highway)

    # Normalize 'maxspeed': string/number -> float
    df_edges["maxspeed_parsed"] = parse_maxspeed_vectorized(df_edges["maxspeed"])

    # To aggregate per node (undirected sense), we duplicate edges: (u,v) + (v,u)
    # This allows a simple groupby on the source node.
    df_rev = df_edges.copy()
    df_rev["u"] = df_edges["v"]
    df_rev["v"] = df_edges["u"]
    
    # Concatenate forward and backward edges
    df_all_inc = pd.concat([df_edges, df_rev], ignore_index=True)

    # Group by Node 'u'
    grouped = df_all_inc.groupby("u")

    # A. Aggregations
    # 1. Average Edge Length
    avg_len_series = grouped["length"].mean()

    # 2. Average Maxspeed
    avg_speed_series = grouped["maxspeed_parsed"].mean().fillna(0.0)

    # 3. Highway Type Frequencies
    # One-hot encode highway types then taking the mean gives the frequency
    # We only care about: residential, primary, motorway, service. Others -> 'other'
    desired_types = ["residential", "primary", "motorway", "service"]
    
    # Map all non-desired types to 'other' in a temp column for counting
    df_all_inc["highway_cat"] = df_all_inc["highway"].apply(lambda x: x if x in desired_types else "other")
    
    # Crosstab or pivot is cleaner, but one-hot encoding manually is memory efficient for large dfs
    # Better: groupby + value_counts(normalize=True), then unstack
    #   u   | highway_cat | proportion
    #   0   | residential | 0.5
    #   0   | other       | 0.5
    freqs = df_all_inc.groupby(["u", "highway_cat"]).size().unstack(fill_value=0)
    # Normalize by row sum to get frequencies
    freqs = freqs.div(freqs.sum(axis=1), axis=0)

    # Ensure all columns exist
    for col in desired_types + ["other"]:
        if col not in freqs.columns:
            freqs[col] = 0.0

    # Align with graph nodes (some nodes might be isolated and missing from df_all_inc)
    # Create a base DataFrame for all nodes
    all_nodes_df = pd.DataFrame(index=range(g_ig.vcount()))
    
    # Merge results
    all_nodes_df["avg_edge_len"] = avg_len_series
    all_nodes_df["avg_maxspeed"] = avg_speed_series
    
    # Merge frequencies
    all_nodes_df = all_nodes_df.join(freqs[desired_types + ["other"]]).fillna(0.0)

    # Assign back to iGraph
    g_ig.vs["avg_edge_len"] = all_nodes_df["avg_edge_len"].values
    g_ig.vs["avg_maxspeed"] = all_nodes_df["avg_maxspeed"].values
    g_ig.vs["freq_residential"] = all_nodes_df["residential"].values
    g_ig.vs["freq_primary"] = all_nodes_df["primary"].values
    g_ig.vs["freq_motorway"] = all_nodes_df["motorway"].values
    g_ig.vs["freq_service"] = all_nodes_df["service"].values
    g_ig.vs["freq_other"] = all_nodes_df["other"].values

    print(f"   Vectorized features computed in {time.time()-t0:.2f}s")


    # 4. Preparing X for PCA
    features_to_use = [
        "degree", "avg_edge_len", "betweenness", "clustering_coeff",
        "freq_residential", "freq_primary", "freq_motorway", 
        "freq_service", "freq_other", "avg_maxspeed"
    ]
    
    # Extract feature matrix X
    # Handle NaNs finally just in case
    X = np.column_stack([
        np.nan_to_num(g_ig.vs[feat], nan=0.0) for feat in features_to_use
    ])
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. PCA & Scoring
    print("\n[4/5] PCA & Urbanity Scoring...")
    pca = PCA(n_components=1)
    pca.fit(X_scaled)
    
    weights = pca.components_[0]
    
    # Check consistency: Degree should correlate positively with Urbanity scores
    # If weight of degree is negative, flip all weights
    idx_degree = features_to_use.index("degree")
    if weights[idx_degree] < 0:
        print("   Inverting PCA signs to match Urban conceptualization.")
        weights = -weights
        
    print("   Learned Feature Weights:")
    for name, w in zip(features_to_use, weights):
        print(f"      {name:<20}: {w:.4f}")

    g_ig.vs["urbanity_score"] = X_scaled @ weights

    # 6. DBSCAN & Pruning
    print("\n[5/5] Clustering & Pruning...")
    
    # Optimizing Epsilon using a sample
    n_nodes = X_scaled.shape[0]
    sample_size = min(50000, n_nodes)
    if n_nodes > sample_size:
        indices = np.random.choice(n_nodes, sample_size, replace=False)
        X_sample = X_scaled[indices]
    else:
        X_sample = X_scaled

    min_samples = 10
    print(f"   Estimating Epsilon on {sample_size} samples...")
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_sample)
    distances, _ = nbrs.kneighbors(X_sample)
    k_distances = np.sort(distances[:, min_samples-1])
    optimal_eps = find_knee_point(k_distances)
    print(f"   Optimal Epsilon: {optimal_eps:.4f}")

    print(f"   Running DBSCAN on full dataset ({n_nodes} nodes)...")
    # DBSCAN can be memory hungry. eps is usually small, so sparse matrix internally helps.
    db = DBSCAN(eps=optimal_eps, min_samples=min_samples, n_jobs=-1)
    labels = db.fit_predict(X_scaled)
    g_ig.vs["cluster_label"] = labels

    # Identify Rural Cluster
    # Approach: Cluster with lowest average Urbanity Score (excluding noise -1)
    unique_labels = set(labels)
    if -1 in unique_labels: 
        unique_labels.remove(-1)

    if not unique_labels:
        print("   Warning: DBSCAN found only noise. Pruning by percentile threshold (Bottom 25%).")
        tau = np.percentile(g_ig.vs["urbanity_score"], 25)
        keep_indices = [i for i, s in enumerate(g_ig.vs["urbanity_score"]) if s >= tau]
    else:
        cluster_avgs = {}
        for lab in unique_labels:
            # We can use numpy mask for speed
            mask = (labels == lab)
            scores = np.array(g_ig.vs["urbanity_score"])[mask]
            cluster_avgs[lab] = np.mean(scores)
        
        rural_label = min(cluster_avgs, key=cluster_avgs.get)
        print(f"   Identified Rural Cluster: Label {rural_label} (Avg Score: {cluster_avgs[rural_label]:.2f})")
        
        # Keep nodes NOT in rural cluster
        keep_indices = [i for i, lab in enumerate(labels) if lab != rural_label]

    # Subgraph
    G_prime = g_ig.subgraph(keep_indices)
    
    # Remap IDs
    new_osmid_map = {new_i: osmid_map[old_i] for new_i, old_i in enumerate(keep_indices)}

    print(f"   Original: {g_ig.vcount():,} nodes -> Pruned: {G_prime.vcount():,} nodes")
    print(f"   Removed: {g_ig.vcount() - G_prime.vcount():,} nodes")

    # Save
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump({
            "graph": G_prime,
            "osmid_map": new_osmid_map,
            "crs": "epsg:4326"
        }, f)
    
    print(f"\n--- DONE in {(time.time()-start_global)/60:.2f} minutes ---")
    print(f"Output saved to: {os.path.abspath(OUTPUT_PKL)}")

if __name__ == "__main__":
    main()
