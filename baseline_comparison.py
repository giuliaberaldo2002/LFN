import igraph as ig
import numpy as np
import pandas as pd
import pickle
import random
import sys
import os
import time

# Import the existing sampling module for metrics
import sampling

# ---------------- CONFIGURATION ------------------------
FULL_GRAPH_PKL = "bremen_processed_graph.pkl"
PRUNED_GRAPH_PKL = "bremen_pruned_graph.pkl"
K_SAMPLES = 50  # Number of points to sample
SEED = 42

# ---------------- BASELINE STRATEGIES ------------------

def random_sampling(g, k, seed=None):
    """
    Baseline 1: Pure Random Sampling.
    Randomly selects k nodes from the graph.
    """
    if seed is not None:
        random.seed(seed)
    
    n_nodes = g.vcount()
    if n_nodes < k:
        return list(range(n_nodes))
        
    return random.sample(range(n_nodes), k)

def load_graph(path):
    """Helper to safely load graph pickle"""
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}")
        return None, None
        
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    return data["graph"], data.get("osmid_map", {})

def print_metrics(name, metrics):
    """Pretty prints the evaluation metrics"""
    print(f"\n--- Results for: {name} ---")
    
    # 1. Global Coverage
    gc = metrics["global_coverage"]
    print(f"  Max Distance to Nearest Sample: {gc['max_nn_distance']:.2f} km")
    print(f"  Mean Distance to Nearest Sample: {gc['mean_nn_distance']:.2f} km")
    
    # 2. Diversity
    div = metrics["diversity"]
    print(f"  Mean Pairwise Distance:         {div['mean_pairwise_distance']:.2f} km")
    print(f"  Min Pairwise Distance:          {div['min_pairwise_distance']:.2f} km")
    
    # 3. Community Coverage
    cc = metrics["community_coverage"]
    print(f"  Community Coverage Rate:        {cc['coverage_rate']:.2%}")
    print(f"  Communities Represented:        {cc['n_communities_represented']} / {cc['n_communities_full']}")

# ---------------- MAIN EXECUTION ------------------------

def main():
    sys.stdout.reconfigure(line_buffering=True)
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"--- COMPARISON OF SAMPLING STRATEGIES (k={K_SAMPLES}) ---")

    # 1. Load Data
    print(f"\n[1/3] Loading Graphs...")
    
    # Full Graph
    g_full, _ = load_graph(FULL_GRAPH_PKL)
    if g_full is None:
        return
    print(f"  Full Graph loaded: {g_full.vcount():,} nodes.")

    # Pruned Graph
    g_pruned, map_pruned_to_orig = load_graph(PRUNED_GRAPH_PKL)
    if g_pruned is None:
        print("  [WARNING] Pruned graph not found. Skipping 'Pruned Random' baseline.")
    else:
        print(f"  Pruned Graph loaded: {g_pruned.vcount():,} nodes.")

    # Prepare Data for Evaluation (Coords + Labels)
    # We assume 'lat' and 'lon' exist. If not, we might need to recover them or use dummy.
    if "lat" in g_full.vs.attribute_names() and "lon" in g_full.vs.attribute_names():
        coords_full = np.column_stack((g_full.vs["lat"], g_full.vs["lon"]))
    else:
        print("  [WARNING] 'lat'/'lon' attributes missing in full graph. Metrics needing coords will fail.")
        coords_full = np.zeros((g_full.vcount(), 2))

    # Ideally we have 'community' labels. If not, we create dummy ones or perform quick clustering.
    if "community" in g_full.vs.attribute_names():
        labels_full = np.array(g_full.vs["community"])
    else:
        print("  [INFO] 'community' attribute missing. Running fast Label Propagation for metric calculation...")
        # Use simple label propagation for baseline metrics
        comm = g_full.community_label_propagation()
        labels_full = np.array(comm.membership)


    # -----------------------------------------------
    # STRATEGY 1: PURE RANDOM (Stupid Algorithm A)
    # -----------------------------------------------
    print(f"\n[2/3] Running Strategy A: Pure Random on Full Graph...")
    t0 = time.time()
    
    sample_indices_A = random_sampling(g_full, K_SAMPLES, seed=SEED)
    
    metrics_A = sampling.evaluate_sampling(coords_full, labels_full, sample_indices_A)
    print_metrics("Pure Random (Full Graph)", metrics_A)
    print(f"  Time: {time.time()-t0:.4f}s")


    # -----------------------------------------------
    # STRATEGY 2: PRUNED RANDOM (Stupid Algorithm B)
    # -----------------------------------------------
    if g_pruned:
        print(f"\n[3/3] Running Strategy B: Random on Pruned Graph...")
        t0 = time.time()
        
        # 1. Sample indices LOCAL to g_pruned
        local_indices_B = random_sampling(g_pruned, K_SAMPLES, seed=SEED)
        
        # 2. Map back to GLOBAL indices (for evaluation against full graph ground truth)
        # Note: 'osmid_map' maps local_idx -> OSMID.
        # But we need local_idx -> full_graph_index.
        # We need a way to map Pruned Nodes -> Full Graph Nodes.
        # Best way: Match OSM IDs.
        
        # Build OSMID -> Index map for Full Graph
        # g_full.vs["_nx_name"] usually holds OSMID if converted from NX.
        # OR we rely on the osmid_map saved in the pickle.
        
        # Let's try to infer the mapping.
        # Ideally, urban_pruning.py should have saved a map OLD_IDX -> NEW_IDX or similar.
        # Currently it saves "osmid_map" which is Index -> OSMID.
        
        full_osmid_to_idx = {osmid: i for i, osmid in enumerate(g_full.vs["_nx_name"])} 
        
        # Get OSMIDs of sampled pruned nodes
        pruned_osmids = [g_pruned.vs["_nx_name"][i] for i in local_indices_B]
        
        # Find their index in Full Graph
        sample_indices_B = []
        for osmid in pruned_osmids:
            if osmid in full_osmid_to_idx:
                sample_indices_B.append(full_osmid_to_idx[osmid])
        
        metrics_B = sampling.evaluate_sampling(coords_full, labels_full, sample_indices_B)
        print_metrics("Random (Pruned Graph)", metrics_B)
        print(f"  Time: {time.time()-t0:.4f}s")

    print("\n--- COMPARISON COMPLETE ---")

if __name__ == "__main__":
    main()
