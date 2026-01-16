"""
visualize_clusters.py
---------------------
Visualizes the graph communities (Leiden) on a static map (PNG).
Highlights the N largest (or smallest) communities with distinct colors,
while leaving the rest as a background layer.

Logic:
  1. Loads the graph (expected to have 'community' and 'x'/'y' or 'lon'/'lat' attributes).
  2. Analyzes community sizes to pick the top-k largest (or smallest).
  3. Assigns colors: distinct colors for the selected top clusters, dark grey for background.
  4. Plots using matplotlib (with Agg backend for headless environments).
  5. Saves the result as a high-DPI PNG.

Inputs:
  - --input: Pickled graph file (default: nord_est_pruned_with_communities.pkl).
  - --output: Output PNG file path.
  - --top_k: Number of communities to highlight (default 20).
  - --smallest: flag to visualize smallest communities instead.
  - --pruned: flag to visualize just the graph structure without community colors.

Outputs:
  - Saves the visualization plot to the specified output file.
"""

import argparse
import pickle
import os
import sys

# Force Agg backend for headless environments (clusters)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import igraph as ig

# -----------------------
# Constants / Defaults
# -----------------------
DEFAULT_INPUT = "nord_est_pruned_with_communities.pkl"
DEFAULT_OUTPUT = "nord_est_clusters.png"
TOP_N_COMMUNITIES = 20  # Number of largest communities to color distinctly


def main():
    """
    Parse CLI args, load the graph, and save a PNG scatter plot highlighting the largest communities.
    """
    parser = argparse.ArgumentParser(description="Visualize Leiden Communities on Map")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input pickled graph file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output PNG file path")
    parser.add_argument("--top_k", type=int, default=TOP_N_COMMUNITIES, help="Number of top communities to color")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output image")
    parser.add_argument("--markersize", type=float, default=0.1, help="Marker size for nodes")
    parser.add_argument("--smallest", action="store_true", help="Visualize smallest communities instead of largest")
    parser.add_argument("--pruned", action="store_true", help="Visualize pruned graph without community coloring")
    args = parser.parse_args()

    # 1. Load Graph
    print(f"Loading graph from {args.input}...")
    if not os.path.exists(args.input):
        print(f"ERROR: File {args.input} not found.")
        sys.exit(1)

    with open(args.input, "rb") as f:
        data = pickle.load(f)
    
    # Handle different package structures safely
    if isinstance(data, dict) and "graph" in data:
        g = data["graph"]
    elif isinstance(data, ig.Graph):
        g = data
    else:
        print("ERROR: Could not find igraph object in pickle.")
        sys.exit(1)

    print(f"Graph loaded. Nodes: {g.vcount():,}, Edges: {g.ecount():,}")

    # 2. Check Attributes
    if not args.pruned and "community" not in g.vs.attribute_names():
        print("ERROR: Graph nodes do not have 'community' attribute. Run Leiden first.")
        sys.exit(1)
    
    # Coordinate check (supports x/y or lon/lat)
    if "x" in g.vs.attribute_names() and "y" in g.vs.attribute_names():
        x = np.array(g.vs["x"])
        y = np.array(g.vs["y"])
    elif "lon" in g.vs.attribute_names() and "lat" in g.vs.attribute_names():
        x = np.array(g.vs["lon"])
        y = np.array(g.vs["lat"])
    else:
        print("ERROR: Graph nodes lack 'x'/'y' or 'lon'/'lat' attributes for plotting.")
        sys.exit(1)

    # 3. Analyze Communities & Color Mapping
    color_map_indices = np.full(g.vcount(), -1, dtype=int)
    label_type = "Filtered"

    if not args.pruned:
        communities = np.array(g.vs["community"])
        
        # Count sizes
        unique_comms, counts = np.unique(communities, return_counts=True)
        
        if args.smallest:
            sorted_indices = np.argsort(counts) # Ascending order
            label_type = "Smallest"
        else:
            sorted_indices = np.argsort(-counts) # Descending order
            label_type = "Largest"

        top_comms = unique_comms[sorted_indices[:args.top_k]]
        
        print(f"Total communities: {len(unique_comms)}")
        print(f"Top {args.top_k} ({label_type}) communities cover {counts[sorted_indices[:args.top_k]].sum() / g.vcount() * 100:.1f}% of nodes.")

        # Assign colors: -1 for background (others), 0..k-1 for top communities
        comm_to_color_idx = {comm_id: idx for idx, comm_id in enumerate(top_comms)}
        
        # Map each node to its color group
        for i in range(len(communities)):
            c_id = communities[i]
            if c_id in comm_to_color_idx:
                color_map_indices[i] = comm_to_color_idx[c_id]
                
    else:
        label_type = "Pruned"
        print("Visualizing pruned graph without communities.")

    # 4. Plotting
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(15, 15), facecolor='black')
    
    if args.pruned:
        # Uniform color for pruned graph structure
        ax.scatter(x, y, 
                   c='cyan', # Light blue for pruned
                   s=args.markersize, 
                   alpha=0.5, 
                   edgecolors='none', 
                   label='Pruned Graph')
    else:
        # Layer 1: Background Nodes (Dark Grey)
        mask_other = (color_map_indices == -1)
        ax.scatter(x[mask_other], y[mask_other], 
                   c='#333333', 
                   s=args.markersize, 
                   alpha=0.5, 
                   edgecolors='none', 
                   label='Other')

        # Layer 2: Highlighted Communities
        mask_top = (color_map_indices != -1)
        
        if np.any(mask_top):
            if args.smallest:
                 # Highlight small components in Red
                 ax.scatter(x[mask_top], y[mask_top], 
                            c='red', 
                            s=args.markersize, 
                            alpha=0.8, 
                            edgecolors='none',
                            label='Smallest Communities')
            else:
                # Highlight large communities with distinct colors
                # Using 'tab20' colormap which gives 20 distinct colors
                cmap = plt.get_cmap('tab20')
                ax.scatter(x[mask_top], y[mask_top], 
                                c=color_map_indices[mask_top], 
                                cmap=cmap, 
                                s=args.markersize, 
                                alpha=0.8, 
                                edgecolors='none')

    ax.set_aspect('equal')
    ax.axis('off')
    
    if args.pruned:
        title_text = "Pruned Graph Visualization"
    else:
        title_text = f"Graph Communities ({label_type} {args.top_k} {'Highlighted Red' if args.smallest else 'Colored'})"
    
    ax.set_title(title_text, color='white', fontsize=16)

    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, format='png', bbox_inches='tight', facecolor='black')
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
