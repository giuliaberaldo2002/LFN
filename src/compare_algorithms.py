
import argparse
import os
import pickle
import random
import json
import logging
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter

# Import sampling functions from your existing module
# Ensure sampling.py is in the Python path or same directory
from sampling import (
    sample_round_robin,
    fft_sample_graph,
    build_csr_adjacency,
    evaluate_midterm
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def load_graph_with_osmid(path: str):
    """
    Loads a pickle file. Expects {'graph': ig.Graph, 'osmid_map': dict, ...}
    or just {'graph': ig.Graph} where nodes satisfy osmid logic.
    Returns (graph, osmid_map).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    
    with open(path, "rb") as f:
        data = pickle.load(f)
    
    g = data["graph"]
    # If osmid_map is present, it maps internal_id -> real_osmid
    # If not, we assume g.vs["osmid"] exists or we use internal IDs as proxies if necessary.
    osmid_map = data.get("osmid_map", {})
    
    # If map is empty but graph has "osmid", we build the map
    if not osmid_map and "osmid" in g.vs.attribute_names():
        osmid_map = {i: v["osmid"] for i, v in enumerate(g.vs)}
    
    return g, osmid_map

def get_osmid_set(g):
    """Returns a set of all OSMIDs in the graph."""
    if "osmid" in g.vs.attribute_names():
        return set(g.vs["osmid"])
    # Fallback if no osmid attr, though typical for this project
    return set(range(g.vcount()))

# ----------------------------------------------------
# 1. ALGORITHMS
# ----------------------------------------------------

def algo_dummy_1_full_random(full_g, k, seed):
    """
    Dummy 1: Randomly picks k nodes from the WHOLE graph.
    Returns: list of osmids
    """
    random.seed(seed)
    n = full_g.vcount()
    if n == 0:
        return []
    
    indices = random.sample(range(n), min(k, n))
    
    # Retrieve true OSMIDs
    if "osmid" in full_g.vs.attribute_names():
        return [full_g.vs[i]["osmid"] for i in indices]
    return indices

def algo_dummy_2_pruned_random(pruned_g, k, seed):
    """
    Dummy 2: Randomly picks k nodes from the PRUNED graph.
    Returns: list of osmids
    """
    random.seed(seed)
    n = pruned_g.vcount()
    if n == 0:
        return []
    
    indices = random.sample(range(n), min(k, n))
    
    if "osmid" in pruned_g.vs.attribute_names():
        return [pruned_g.vs[i]["osmid"] for i in indices]
    return indices

def algo_pipeline_smart(pruned_g, k, seed, weight_attr="length"):
    """
    Pipeline: Uses Smart Sampling (Round Robin or FFT) on the PRUNED graph.
    We'll use FFT (Farthest First Traversal) as the 'Smart' default here, 
    or Round Robin if 'community' is highly prioritized.
    Let's use FFT for spatial coverage strength, or Round Robin for balance.
    Based on requests, let's try FFT here.
    """
    # For determinism
    # fft_sample_graph internally uses argmax, but initialization might be robust.
    # If we want Round Robin:
    if "community" in pruned_g.vs.attribute_names():
         indices = sample_round_robin(pruned_g, k, seed=seed)
    else:
         # Fallback to FFT if communities missing
         indices = fft_sample_graph(pruned_g, k, weight_attr=weight_attr, seed_idx=None)
    
    if "osmid" in pruned_g.vs.attribute_names():
        return [pruned_g.vs[i]["osmid"] for i in indices]
    return indices

# ----------------------------------------------------
# 2. METRICS WRAPPER
# ----------------------------------------------------

def evaluate_sample_set(name, k, sample_osmids, full_g, pruned_g, 
                        pruned_indptr, pruned_indices, pruned_data, weight_attr):
    """
    Evaluates a set of sampled OSMIDs against the 'Ground Truth' (Pruned Graph).
    
    Metrics:
    - Efficiency (Urban Hit Rate): % of samples that are in the Pruned Graph.
    - Coverage: Calculated ONLY on the valid Urban samples within the Pruned Graph.
    """
    
    # Map OSMIDs back to Pruned Graph Internal Indices
    # We need a reverse map for the pruned graph: osmid -> index
    if "osmid" in pruned_g.vs.attribute_names():
        pruned_osmid_to_idx = {val: i for i, val in enumerate(pruned_g.vs["osmid"])}
    else:
        # Fallback if testing without osmids
        pruned_osmid_to_idx = {i: i for i in range(pruned_g.vcount())}
        
    valid_indices = []
    rural_count = 0
    
    for oid in sample_osmids:
        if oid in pruned_osmid_to_idx:
            valid_indices.append(pruned_osmid_to_idx[oid])
        else:
            rural_count += 1
            
    efficiency = len(valid_indices) / k if k > 0 else 0
    
    # If we have valid urban nodes, calculate detailed metrics on the pruned graph
    metrics = {
        "k": k,
        "algo": name,
        "efficiency": efficiency,
        "rural_count": rural_count,
        "valid_urban_count": len(valid_indices)
    }
    
    # We use the existing 'evaluate_midterm' function for graph metrics.
    # Note: evaluate_midterm expects indices relative to the graph passed to it.
    # We pass 'pruned_g' as the environment.
    
    if len(valid_indices) > 0:
        labels_full = np.array(pruned_g.vs["community"], dtype=int) if "community" in pruned_g.vs.attribute_names() else np.zeros(pruned_g.vcount())
        
        midterm_results = evaluate_midterm(
            pruned_g, 
            labels_full, 
            valid_indices, 
            pruned_indptr, 
            pruned_indices, 
            pruned_data, 
            eval_subset=None, 
            weight_attr=weight_attr
        )
        
        # Flatten structure slightly for easier plotting
        metrics["cov_ratio"] = midterm_results["community_coverage"]["coverage"]
        metrics["global_R_max"] = midterm_results["global_coverage"]["R_max"]
        metrics["div_min_sep"] = midterm_results["diversity"]["min_sep"]
        metrics["balance_cv"] = midterm_results["balance"]["cv"]
        
    else:
        # Defaults if no valid urban nodes were found
        metrics["cov_ratio"] = 0.0
        metrics["global_R_max"] = np.inf
        metrics["div_min_sep"] = 0.0
        metrics["balance_cv"] = np.nan

    return metrics

# ----------------------------------------------------
# 3. PLOTTING
# ----------------------------------------------------

def plot_comparison(df, output_dir):
    """
    Generates comparison plots from the results DataFrame.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    methods = df['algo'].unique()
    ks = sorted(df['k'].unique())
    
    # Define metrics to plot
    # (Metric Key, Y-Label, Title, Filename)
    plots_config = [
        ("efficiency", "Urban Efficiency (0-1)", "Urban Efficiency vs k", "efficiency.png"),
        ("cov_ratio", "Community Coverage (0-1)", "Community Coverage vs k", "coverage.png"),
        ("global_R_max", "Max Distance to Nearest Sample (m)", "Global Coverage (R_max) vs k (Lower is Better)", "global_coverage.png"),
        ("div_min_sep", "Min Separation (m)", "Diversity (Min Sep) vs k (Higher is Better)", "diversity.png")
    ]
    
    for metric, ylabel, title, fname in plots_config:
        plt.figure(figsize=(10, 6))
        
        for method in methods:
            subset = df[df['algo'] == method].sort_values('k')
            plt.plot(subset['k'], subset[metric], marker='o', label=method, linewidth=2)
            
        plt.xlabel("k (Number of Samples)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(os.path.join(output_dir, fname))
        plt.close()

# ----------------------------------------------------
# 4. CITY ANALYSIS (Optional)
# ----------------------------------------------------
def analyze_cities(g, indices):
    """
    If 'addr:city' or 'city' tag exists, count how many unique cities are covered.
    """
    cities = []
    # Check potential attributes
    attr = None
    if "addr:city" in g.vs.attribute_names():
        attr = "addr:city"
    elif "city" in g.vs.attribute_names():
        attr = "city"
        
    if not attr:
        return 0
        
    for i in indices:
        val = g.vs[i][attr]
        if val:
            cities.append(val)
            
    return len(set(cities))

# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare Sampling Algorithms")
    parser.add_argument("--full", required=True, help="Path to Full Graph (pickle)")
    parser.add_argument("--pruned", required=True, help="Path to Pruned/Community Graph (pickle)")
    parser.add_argument("--output", default="comparison_results", help="Output directory")
    parser.add_argument("--ks", nargs="+", type=int, default=[10, 20, 50, 100], help="List of k values")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # 1. Load Graphs
    logger.info("Loading graphs...")
    full_g, _ = load_graph_with_osmid(args.full)
    pruned_g, _ = load_graph_with_osmid(args.pruned)
    
    logger.info(f"Full Graph: {full_g.vcount()} nodes")
    logger.info(f"Pruned Graph: {pruned_g.vcount()} nodes")
    
    # Pre-compute Dijkstra structures for the Pruned Graph (Efficiency)
    indptr, indices, data = build_csr_adjacency(pruned_g, weight_attr="length")
    
    results = []
    
    # 2. Run Experiments
    for k in args.ks:
        logger.info(f"--- Running for k={k} ---")
        
        # A) Dummy 1: Full Random
        # Algorithm
        s1_osmids = algo_dummy_1_full_random(full_g, k, args.seed)
        # Evaluate
        m1 = evaluate_sample_set("Dummy 1 (Full Random)", k, s1_osmids, full_g, pruned_g, 
                                 indptr, indices, data, "length")
        results.append(m1)
        
        # B) Dummy 2: Pruned Random
        # Algorithm
        s2_osmids = algo_dummy_2_pruned_random(pruned_g, k, args.seed)
        # Evaluate
        m2 = evaluate_sample_set("Dummy 2 (Pruned Random)", k, s2_osmids, full_g, pruned_g, 
                                 indptr, indices, data, "length")
        results.append(m2)
        
        # C) Pipeline: Smart
        # Algorithm
        s3_osmids = algo_pipeline_smart(pruned_g, k, args.seed, "length")
        # Evaluate
        m3 = evaluate_sample_set("Pipeline (Smart)", k, s3_osmids, full_g, pruned_g, 
                                 indptr, indices, data, "length")
        results.append(m3)
        
    # 3. Save Results
    df = pd.DataFrame(results)
    
    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "metrics.json")
    df.to_json(json_path, orient="records", indent=2)
    logger.info(f"Saved metrics to {json_path}")
    
    # 4. Plot
    plot_comparison(df, args.output)
    logger.info(f"Saved plots to {args.output}/")
    
    # Print Summary Table
    print("\nSummary of Results (Efficiency & Coverage):")
    print(df[["k", "algo", "efficiency", "cov_ratio", "global_R_max"]].to_string(index=False))

if __name__ == "__main__":
    main()
