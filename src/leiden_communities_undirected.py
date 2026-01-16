"""
leiden_communities_undirected.py
--------------------------------
Performs Leiden community detection on the pruned graph.
This script ensures the graph is treated as undirected during the clustering process
to avoid directionality bias in community formation.

Logic:
  1. Loads the pruned graph.
  2. Converts to undirected (if directed), collapsing edges with min-weight strategy.
  3. Prepares edge weights (inverse of length) for the optimization objective.
  4. Runs a parameter sweep on the resolution parameter (gamma) for Leiden (CPM or Modularity).
  5. Selects the partition with the highest quality score.
  6. Filters to keep only the top N communities (by size), setting others to -1.
  7. Saves the graph with a new 'community' vertex attribute.

Inputs:
  - --input: Pruned graph pickle.
  - --top_n: Number of largest communities to retain (default: 300).
  - --objective: Optimization function ('CPM' or 'modularity').

Outputs:
  - Saves the graph with updated 'community' labels to the output path.
"""

import argparse
import igraph as ig
import pickle
import numpy as np
import sys
import os

# -----------------------
# Constants & Defaults
# -----------------------
DEFAULT_INPUT = "nord_est_pruned_graph.pkl"
DEFAULT_OUTPUT = "nord_est_pruned_with_communities.pkl"


# -----------------------
# Helper Functions
# -----------------------

def summarize_partition(part):
    """Computes summary statistics for a given partition."""
    sizes = np.array(part.sizes())
    return {
        "n_communities": len(part),
        "quality": float(part.q),
        "size_min": int(sizes.min()),
        "size_median": float(np.median(sizes)),
        "size_max": int(sizes.max()),
    }

def run_leiden_tuning(g, gammas, objective="CPM", weights=None, n_iterations=10):
    """
    Runs Leiden optimization for a list of gamma values.
    Returns a list of results containing the partition and its stats.
    """
    results = []
    for gamma in gammas:
        print(f"\n=== Gamma = {gamma} ===")
        part = g.community_leiden(
            objective_function=objective,
            weights=weights,
            resolution=gamma,
            n_iterations=n_iterations,
        )
        stats = summarize_partition(part)
        results.append({"gamma": gamma, "partition": part, "stats": stats})
        print(f" communities: {stats['n_communities']} | quality (q) = {stats['quality']:.4f}")
    return results


# -----------------------
# Main Execution
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Run Leiden Community Detection on Pruned Graph")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input pruned graph pickle")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output graph with community labels pickle")
    parser.add_argument("--objective", default="CPM", choices=["CPM", "modularity"], help="Leiden objective function")
    parser.add_argument("--top_n", type=int, default=300, help="Number of top communities to keep")
    args = parser.parse_args()

    # 1. Load Graph
    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found.")
        sys.exit(1)

    print(f"Loading graph from {args.input}...")
    with open(args.input, "rb") as f:
        data_pkg = pickle.load(f)
    Gp = data_pkg["graph"]
    print(f"Loaded pruned graph. Nodes: {Gp.vcount():,}, Edges: {Gp.ecount():,}")

    # 2. Prepare Undirected Graph for Clustering
    # Leiden works best on undirected graphs for spatial clustering.
    # We collapse multi-edges by taking the minimum length (or weight).
    if Gp.is_directed():
        print("Graph is directed. Creating undirected copy for Leiden execution...")
        Gp_run = Gp.copy()
        Gp_run.to_undirected(mode="collapse", combine_edges={"length": "min", "weight": "min"})
    else:
        Gp_run = Gp

    # 3. Define Edge Weights
    # We invert 'length' to get 'strength': closer nodes = higher edge weight
    weights = None
    edge_attr_names = Gp_run.es.attribute_names()

    if "length" in edge_attr_names:
        lengths = np.array(Gp_run.es["length"], dtype=float)
        strengths = 1.0 / (lengths + 1e-6)
        weights = strengths.tolist()
        print("Using custom weights: 1 / length")
    elif "weight" in edge_attr_names:
        weights = "weight"
        print("Using edge weights attribute: 'weight'")
    else:
        print("No length/weight attribute found. Using unweighted Leiden.")

    # 4. Run Gamma Sweep (Hyperparameter Tuning)
    # Different gammas yield different granularities.
    GAMMAS = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05] if args.objective == "CPM" else [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]

    results = run_leiden_tuning(Gp_run, gammas=GAMMAS, objective=args.objective, weights=weights)
    
    # Pick best result by quality score
    best = max(results, key=lambda r: r["stats"]["quality"])
    best_part = best["partition"]
    print(f"\nBest Gamma: {best['gamma']} (Quality: {best['stats']['quality']:.4f})")

    # 5. Filter Top N Communities
    # We only care about the largest urban clusters. Small fragments are discarded (-1).
    print(f"\nFiltering top {args.top_n} communities by size...")
    
    comm_sizes = best_part.sizes()
    
    # Sort communities by size desc
    # sorted_communities is list of (community_id, size)
    sorted_communities = sorted(enumerate(comm_sizes), key=lambda x: x[1], reverse=True)
    
    # Identify top N community IDs
    top_n_indices = [info[0] for info in sorted_communities[:args.top_n]]
    top_n_set = set(top_n_indices)
    
    # Create new membership vector: keep ID if in top N, else -1
    original_membership = best_part.membership
    filtered_membership = [m if m in top_n_set else -1 for m in original_membership]
    
    # 6. Apply & Save
    # Apply the filtered membership back to the original (possibly directed) graph object
    Gp.vs["community"] = filtered_membership
    
    nodes_in_top = sum(1 for m in filtered_membership if m != -1)
    
    print(f"Filtering complete.")
    print(f"Total nodes: {Gp.vcount()}")
    print(f"Nodes assigned to top {args.top_n} communities: {nodes_in_top} ({(nodes_in_top/Gp.vcount())*100:.2f}%)")
    print(f"Nodes set to -1 (discarded): {Gp.vcount() - nodes_in_top}")

    out_pkg = data_pkg.copy()
    out_pkg["graph"] = Gp

    with open(args.output, "wb") as f:
        pickle.dump(out_pkg, f)

    print(f"Saved graph with community labels to: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
