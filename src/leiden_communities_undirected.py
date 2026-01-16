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
from collections import Counter, defaultdict


DEFAULT_INPUT = "nord_est_pruned_graph.pkl"
DEFAULT_OUTPUT = "nord_est_pruned_with_communities.pkl"


def summarize_partition(part: ig.clustering.VertexClustering):
    sizes = np.array(part.sizes(), dtype=int)
    return {
        "n_communities": int(len(part)),
        "quality": float(part.q),
        "size_min": int(sizes.min()),
        "size_median": float(np.median(sizes)),
        "size_p10": float(np.percentile(sizes, 10)),
        "size_p90": float(np.percentile(sizes, 90)),
        "size_max": int(sizes.max()),
        "frac_singletons": float(np.mean(sizes == 1)),
        "frac_leq2": float(np.mean(sizes <= 2)),
    }


def run_leiden_once(g: ig.Graph, gamma: float, objective: str, weights, n_iterations: int, seed: int):
    # igraph versions may differ on resolution parameter name.
    try:
        return g.community_leiden(
            objective_function=objective,
            weights=weights,
            resolution_parameter=gamma,
            n_iterations=n_iterations,
            seed=seed,
        )
    except TypeError:
        # fallback for older igraph
        return g.community_leiden(
            objective_function=objective,
            weights=weights,
            resolution=gamma,
            n_iterations=n_iterations,
            seed=seed,
        )


def run_leiden_tuning(g: ig.Graph, gammas, objective="CPM", weights=None, n_iterations=10, seed=0):
    results = []
    for gamma in gammas:
        print(f"\n=== Gamma = {gamma:.2e} ===")
        part = run_leiden_once(g, gamma, objective, weights, n_iterations, seed)
        stats = summarize_partition(part)

        results.append({"gamma": gamma, "partition": part, "stats": stats})

        print(
            f" n_comm={stats['n_communities']:,}"
            f" | size[min/med/max]={stats['size_min']}/{stats['size_median']:.1f}/{stats['size_max']}"
            f" | p10/p90={stats['size_p10']:.1f}/{stats['size_p90']:.1f}"
            f" | singletons={100*stats['frac_singletons']:.1f}%"
            f" | q={stats['quality']:.4f}"
        )
    return results


def pick_best_gamma(results, max_singletons=0.30, min_median_size=10.0):
    # Filter partitions that look “city-like”
    candidates = [
        r for r in results
        if (r["stats"]["frac_singletons"] <= max_singletons)
        and (r["stats"]["size_median"] >= min_median_size)
    ]

    if not candidates:
        print("\nWARNING: no gamma satisfies constraints; falling back to max q.")
        return max(results, key=lambda r: r["stats"]["quality"])

    # tie-break by quality among feasible candidates
    return max(candidates, key=lambda r: r["stats"]["quality"])


def merge_tiny_communities(g: ig.Graph, labels: list[int], s_min: int = 10, passes: int = 2) -> list[int]:
    """
    Merge communities with size < s_min into the most frequent neighboring community.
    Works on the current graph topology (undirected recommended).
    """
    labels = list(map(int, labels))
    n = g.vcount()

    for _ in range(passes):
        # build size table
        sizes = Counter(labels)

        # group nodes by community
        comm_nodes = defaultdict(list)
        for v, c in enumerate(labels):
            comm_nodes[c].append(v)

        # identify tiny communities
        tiny = [c for c, sz in sizes.items() if sz < s_min]
        if not tiny:
            break

        changed = 0

        for c in tiny:
            nodes = comm_nodes[c]
            if not nodes:
                continue

            neighbor_comms = Counter()

            # look at boundary neighbors
            for v in nodes:
                for u in g.neighbors(v):
                    cu = labels[u]
                    if cu != c:
                        neighbor_comms[cu] += 1

            if not neighbor_comms:
                # isolated tiny component: skip (or you could assign to closest by coord if you want)
                continue

            # pick most frequent neighboring community
            target = neighbor_comms.most_common(1)[0][0]

            # reassign all nodes in c
            for v in nodes:
                labels[v] = target
            changed += 1

        if changed == 0:
            break

    return labels



def main():
    parser = argparse.ArgumentParser(description="Run Leiden (CPM or modularity) on pruned graph + tune gamma.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input pruned graph pickle")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output graph with community labels pickle")
    parser.add_argument("--objective", default="CPM", choices=["CPM", "modularity"], help="Leiden objective function")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_singletons", type=float, default=0.30)
    parser.add_argument("--min_median_size", type=float, default=10.0)
    args = parser.parse_args()

    # --- Load graph package ---
    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found.")
        sys.exit(1)

    print(f"Loading graph from {args.input}...")
    with open(args.input, "rb") as f:
        data_pkg = pickle.load(f)
    if isinstance(data_pkg, dict) and "graph" in data_pkg:
        Gp = data_pkg["graph"]
    else:
        # allow pickled igraph directly
        Gp = data_pkg

    print(f"Loaded pruned graph. Nodes: {Gp.vcount():,}, Edges: {Gp.ecount():,}")

    # --- Undirected copy for Leiden ---
    if Gp.is_directed():
        print("Graph is directed. Creating undirected copy (collapse) for Leiden...")
        Gp_run = Gp.copy()
        Gp_run.to_undirected(mode="collapse", combine_edges={"length": "mean", "weight": "mean"})
    else:
        Gp_run = Gp

    # --- Choose weights ---
    weights = None
    edge_attr_names = Gp_run.es.attribute_names()

    if "length" in edge_attr_names:
        lengths = np.asarray(Gp_run.es["length"], dtype=float)
        # Use "strength" as similarity weight (bigger for shorter edges)
        Gp_run.es["strength"] = (1.0 / (lengths + 1e-6)).astype(float).tolist()
        weights = "strength"  # pass attribute name
        print("Using weights='strength' where strength=1/length")
    elif "weight" in edge_attr_names:
        weights = "weight"
        print("Using weights='weight'")
    else:
        print("No 'length' or 'weight' edge attribute found: running unweighted Leiden.")

    # --- Gamma grid (hardcoded, plausible) ---
    if args.objective == "CPM":
        GAMMAS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
    else:
        # keep the values smaller 
        GAMMAS = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

    print(f"\nRunning Leiden objective={args.objective} over {len(GAMMAS)} gamma values...")
    results = run_leiden_tuning(
        Gp_run,
        gammas=GAMMAS,
        objective=args.objective,
        weights=weights,
        n_iterations=10,
        seed=args.seed,
    )

    best = pick_best_gamma(results, max_singletons=args.max_singletons, min_median_size=args.min_median_size)
    best_gamma = best["gamma"]
    part = best["partition"]
    stats = best["stats"]

    labels = list(map(int, part.membership))
    
    # Merge micro-communities
    labels = merge_tiny_communities(Gp_run, labels, s_min=10, passes=2)
    # Assign to the original graph
    Gp.vs["community"] = labels


    print("\nSelected gamma:", best_gamma)
    print(
        f" n_comm={stats['n_communities']:,}"
        f" | median={stats['size_median']:.1f}"
        f" | singletons={100*stats['frac_singletons']:.1f}%"
        f" | q={stats['quality']:.4f}"
    )

   
    # Gp.vs["community"] = list(map(int, part.membership))

    # --- Save (preserve package if dict) ---
    if isinstance(data_pkg, dict):
        out_pkg = data_pkg.copy()
        out_pkg["graph"] = Gp
        out_pkg["best_gamma"] = float(best_gamma)
        out_pkg["best_stats"] = stats
    else:
        out_pkg = {"graph": Gp, "best_gamma": float(best_gamma), "best_stats": stats}

    with open(args.output, "wb") as f:
        pickle.dump(out_pkg, f)

    print(f"Saved graph with community labels to: {os.path.abspath(args.output)}")


if __name__ == "__main__":
    main()
