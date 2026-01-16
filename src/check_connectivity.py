"""
check_connectivity.py
---------------------
Checks component/community sizes in the graph.
Loads a graph (pickle), verifies the 'community' attribute exists, and analyzes
the distribution of community sizes.

Logic:
  - Loads graph from input pickle.
  - Counts nodes per 'community'.
  - Reports total number of communities.
  - Reports percentage of single-node and 2-node communities.
  - Plots the sizes of the 10 smallest communities.

Inputs:
  - --input: Path to graph pickle (must have 'community' vertex attribute).
  - --output_plot: Path to save the PNG plot (default: smallest_communities_sizes.png).

Outputs:
  - Prints statistics to stdout.
  - Saves a bar chart of the smallest community sizes.
"""

import igraph as ig
import pickle
import argparse
import sys
import os
from collections import Counter

def main():
    parser = argparse.ArgumentParser(description="Check connected components of a graph.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input pickle file.")
    parser.add_argument("--output_plot", type=str, default="smallest_communities_sizes.png", help="Path to save the output plot.")
    args = parser.parse_args()

    input_path = args.input

    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        sys.exit(1)

    print(f"Loading graph from {input_path}...")
    try:
        with open(input_path, 'rb') as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle: {e}")
        sys.exit(1)

    # Handle both direct graph and dictionary package
    if isinstance(data, dict) and "graph" in data:
        G = data["graph"]
        print("Loaded graph from dictionary package.")
    else:
        G = data
        print(f"Loaded object directly. Type: {type(G)}")

    if not isinstance(G, ig.Graph):
         print(f"Error: Expected igraph.Graph, but got {type(G)}")
         sys.exit(1)

    print(f"Graph loaded. Nodes: {G.vcount()}, Edges: {G.ecount()}")

    # Check for community attribute
    if "community" not in G.vs.attribute_names():
        print("Error: Graph does not have 'community' attribute. Cannot compute community sizes.")
        sys.exit(1)

    # Get community sizes
    communities = G.vs["community"]
    comm_counter = Counter(communities)
    component_sizes = list(comm_counter.values())
    num_components = len(comm_counter)
    
    print(f"Number of communities: {num_components}")

    # Sort ascending for smallest
    component_sizes.sort()

    # Compute percentages for communities with 1 and 2 nodes
    num_single = component_sizes.count(1)
    num_double = component_sizes.count(2)
    perc_single = (num_single / num_components * 100) if num_components > 0 else 0
    perc_double = (num_double / num_components * 100) if num_components > 0 else 0

    print(f"Percentage of communities with 1 node: {perc_single:.2f}%")
    print(f"Percentage of communities with 2 nodes: {perc_double:.2f}%")

    top_10_smallest = component_sizes[:10]
    print("Top 10 smallest communities sizes:")
    for i, size in enumerate(top_10_smallest, 1):
        print(f"{i}. {size} nodes")

    # Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, len(top_10_smallest) + 1), top_10_smallest, color='skyblue')
        plt.xlabel('Rank (Smallest to Largest)')
        plt.ylabel('Number of Nodes')
        plt.title('Sizes of the 10 Smallest Communities')
        plt.xticks(range(1, len(top_10_smallest) + 1))
        
        output_plot = "smallest_communities_sizes.png"
        if args.output_plot:
            output_plot = args.output_plot
            
        plt.savefig(output_plot)
        print(f"Plot saved to {output_plot}")
    except ImportError:
        print("Error: matplotlib is not installed, cannot generate plot.")
    except Exception as e:
        print(f"Error generating plot: {e}")

if __name__ == "__main__":
    main()
