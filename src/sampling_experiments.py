import pickle
import igraph as ig
import numpy as np

from sampling import (
    sample_by_community,
    fft_sample,
    mean_pairwise_distance_geo,
    mean_pairwise_distance_graph,
    community_coverage_gini,
)

INPUT_PKL = "bremen_pruned_with_communities.pkl"
K_VALUES = [5, 10, 20] # TODO: enlarge the possible values that k assumes (for instance k = 5, 10, 20, 30 ,...) 


def load_graph(path: str) -> ig.Graph:
    with open(path, "rb") as f:
        data_pkg = pickle.load(f)
    g = data_pkg["graph"]
    if "community" not in g.vs.attribute_names():
        raise RuntimeError("Graph has no vertex attribute 'community'. Run Leiden first.")
    print(f"Loaded graph from {path}: {g.vcount():,} nodes, {g.ecount():,} edges.")
    return g


def main():
    g = load_graph(INPUT_PKL)

    for k in K_VALUES:
        print(f"\n=== k = {k} ===")

        # 1) Community-based
        comm_nodes = sample_by_community(g, k, seed=42)
        mean_graph_comm = mean_pairwise_distance_graph(g, comm_nodes, weight_attr="length")
        # You can comment this out if you don't care about Haversine:
        mean_geo_comm = mean_pairwise_distance_geo(g, comm_nodes)
        gini_comm = community_coverage_gini(g, comm_nodes)

        print("[Community-based sampling]")
        print(f"  Sampled nodes: {comm_nodes}")
        print(f"  Mean graph distance (length): {mean_graph_comm:.2f}")
        print(f"  Mean geo distance (km, Haversine): {mean_geo_comm:.2f}")
        print(f"  Gini of community coverage: {gini_comm:.3f}")

        # 2) FFT
        fft_nodes = fft_sample(g, k, weight_attr="length")
        mean_graph_fft = mean_pairwise_distance_graph(g, fft_nodes, weight_attr="length")
        mean_geo_fft = mean_pairwise_distance_geo(g, fft_nodes)
        gini_fft = community_coverage_gini(g, fft_nodes)

        print("\n[Farthest-First Traversal sampling]")
        print(f"  Sampled nodes: {fft_nodes}")
        print(f"  Mean graph distance (length): {mean_graph_fft:.2f}")
        print(f"  Mean geo distance (km, Haversine): {mean_geo_fft:.2f}")
        print(f"  Gini of community coverage: {gini_fft:.3f}")


if __name__ == "__main__":
    main()
