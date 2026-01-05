"""
leiden_communities_undirected.py

Run Leiden community detection on a (possibly directed) pruned road graph and
save the graph with a vertex-level "community" label.

Input:
- A pickle containing either:
  - an igraph.Graph, or
  - a dict with key "graph" -> igraph.Graph.

Processing:
- If the graph is directed, create an undirected working copy for Leiden via
  edge collapse (combining attributes like length/weight by mean).
- Choose edge weights if available ("length" preferred, otherwise "weight";
  otherwise unweighted Leiden).
- Run a small resolution (gamma) sweep depending on objective:
  - CPM: small resolution values
  - modularity: values around 1.0
- Select the partition with the best reported quality (q) and write
  membership to the ORIGINAL graph as:  Gp.vs["community"] = membership

Output:
- A pickle saved to --output. If the input was a dict package, the same dict
  is copied and its "graph" entry is replaced with the updated graph.

CLI:
  --input, --output, --objective {CPM, modularity}
"""

import argparse
import igraph as ig
import pickle
import numpy as np
import sys
import os

# arguments
DEFAULT_INPUT = "nord_est_pruned_graph.pkl"
DEFAULT_OUTPUT = "nord_est_pruned_with_communities.pkl"

"""
    Parse CLI args, run Leiden (optionally weighted), attach ``community`` labels, and save the result.
"""
def main():
    parser = argparse.ArgumentParser(description="Run Leiden Community Detection on Pruned Graph")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input pruned graph pickle")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output graph with community labels pickle")
    parser.add_argument("--objective", default="CPM", choices=["CPM", "modularity"], help="Leiden objective function")
    args = parser.parse_args()

    # --- Load graph ---
    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found.")
        sys.exit(1)

    print(f"Loading graph from {args.input}...")
    with open(args.input, "rb") as f:
        data_pkg = pickle.load(f)
    Gp = data_pkg["graph"]
    print(f"Loaded pruned graph. Nodes: {Gp.vcount():,}, Edges: {Gp.ecount():,}")

    # ----------------- CHOOSE WEIGHTS (ONE VARIABLE) -----------------
    # Create a working copy for Leiden (must be undirected)
    if Gp.is_directed():
        print("Graph is directed. Creating undirected copy for Leiden execution...")
        Gp_run = Gp.copy()
        # Collapse edges and average lengths. 
        # combining 'length' is critical for the weights.
        Gp_run.to_undirected(mode="collapse", combine_edges={"length": "mean", "weight": "mean"})
    else:
        Gp_run = Gp

    weights = None
    edge_attr_names = Gp_run.es.attribute_names()

    if "length" in edge_attr_names:
        # Option B: use 1 / length as strength (shorter roads = stronger connection)
        lengths = np.array(Gp_run.es["length"], dtype=float)
        strengths = 1.0 / (lengths + 1e-6)
        weights = strengths.tolist()
        print("Using custom weights: 1 / length")
    elif "weight" in edge_attr_names:
        # Fallback: use existing 'weight' attribute by name
        weights = "weight"
        print("Using edge weights attribute: 'weight'")
    else:
        print("No 'length' or 'weight' attribute found, running unweighted Leiden.")

    # ----------------- HELPER FUNCTIONS -----------------
    def summarize_partition(part):
        sizes = np.array(part.sizes())
        return {
            "n_communities": len(part),
            "quality": float(part.q),
            "size_min": int(sizes.min()),
            "size_median": float(np.median(sizes)),
            "size_max": int(sizes.max()),
        }

    def run_leiden_tuning(g, gammas, objective="CPM", weights=None, n_iterations=10):
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
            print(
                f" communities: {stats['n_communities']}"
                f" | size[min/med/max] = {stats['size_min']}/"
                f"{stats['size_median']}/{stats['size_max']}"
                f" | quality (q) = {stats['quality']:.4f}"
            )
        return results

    # ----------------- GAMMA SWEEP -----------------
    if args.objective == "CPM":
        GAMMAS = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
    else:
        # For Modularity, resolution around 1.0 is standard. Lower -> larger communities.
        GAMMAS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

    print(f"\nRunning Leiden (community_leiden) with {args.objective} for different gamma values...")
    results = run_leiden_tuning(
        Gp_run,
        gammas=GAMMAS,
        objective=args.objective,
        weights=weights,
        n_iterations=10,
    )

    best = max(results, key=lambda r: r["stats"]["quality"])
    best_gamma = best["gamma"]
    best_part = best["partition"]
    best_stats = best["stats"]

    print("\nSelected gamma (max q):", best_gamma)
    print(
        f" communities: {best_stats['n_communities']}"
        f" | size[min/med/max] = {best_stats['size_min']}/"
        f"{best_stats['size_median']}/{best_stats['size_max']}"
        f" | quality (q) = {best_stats['quality']:.4f}"
    )

    # ----------------- USE BEST PARTITION -----------------
    part = best_part
    
    # Assign membership back to the ORIGINAL (potentially directed) graph
    # Node indices are preserved during to_undirected(mode='collapse')
    Gp.vs["community"] = part.membership
    num_comms = len(part)
    print(f"\nLeiden finished. Found {num_comms} communities with gamma = {best_gamma}.")

    sizes = sorted(part.sizes(), reverse=True)
    print("Top 10 community sizes:", sizes[:10])

    # Save ONLY the graph in the package (or preserve other keys if they existed in input? Input structure was unclear, usually we keep everything)
    # The original script just saved {"graph": Gp}. We should stick to that or improve it.
    # Ideally we should assume the input package structure and just update the graph.
    # But for now, let's stick to returning what the previous script returned: just the graph dict.
    # Actually, verify data_pkg keys.
    # To be safe, let's copy the input package and update 'graph'.
    
    out_pkg = data_pkg.copy()
    out_pkg["graph"] = Gp
    
    with open(args.output, "wb") as f:
        pickle.dump(out_pkg, f)

    print(f"Saved graph with community labels to: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()
