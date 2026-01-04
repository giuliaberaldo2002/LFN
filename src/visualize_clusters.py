import argparse
import pickle
import igraph as ig
import matplotlib
# Force Agg backend for headless environments (clusters)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

DEFAULT_INPUT = "nord_est_pruned_with_communities.pkl"
DEFAULT_OUTPUT = "nord_est_clusters.png"
TOP_N_COMMUNITIES = 20  # Number of largest communities to color distinctly

def main():
    parser = argparse.ArgumentParser(description="Visualize Leiden Communities on Map")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input pickled graph file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output PNG file path")
    parser.add_argument("--top_k", type=int, default=TOP_N_COMMUNITIES, help="Number of top communities to color")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output image")
    parser.add_argument("--markersize", type=float, default=0.1, help="Marker size for nodes")
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
    if "community" not in g.vs.attribute_names():
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

    communities = np.array(g.vs["community"])
    
    # 3. Analyze Communities
    unique_comms, counts = np.unique(communities, return_counts=True)
    sorted_indices = np.argsort(-counts) # Descending order
    top_comms = unique_comms[sorted_indices[:args.top_k]]
    
    print(f"Total communities: {len(unique_comms)}")
    print(f"Top {args.top_k} communities cover {counts[sorted_indices[:args.top_k]].sum() / g.vcount() * 100:.1f}% of nodes.")

    # 4. Prepare Plot Data
    # Assign colors: -1 for background (others), 0..k-1 for top communities
    color_map_indices = np.full(g.vcount(), -1, dtype=int)
    
    comm_to_color_idx = {comm_id: idx for idx, comm_id in enumerate(top_comms)}
    
    # Vectorized mapping for speed could be tricky with dict, utilizing simple loop or broadcasting if K is small
    # Since K is small (20), we can just iterate
    for i in range(len(communities)):
        c_id = communities[i]
        if c_id in comm_to_color_idx:
            color_map_indices[i] = comm_to_color_idx[c_id]

    # 5. Plotting
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(15, 15), facecolor='black')
    
    # Plot background nodes (small communities) first
    mask_other = (color_map_indices == -1)
    ax.scatter(x[mask_other], y[mask_other], 
               c='#333333', # Dark Grey
               s=args.markersize, 
               alpha=0.5, 
               edgecolors='none', 
               label='Other')

    # Plot top communities
    # Using 'tab20' colormap which gives 20 distinct colors
    cmap = plt.get_cmap('tab20')
    
    # We scatter the rest in one go using the color map indices, but we need to filter non -1
    mask_top = (color_map_indices != -1)
    
    if np.any(mask_top):
        sc = ax.scatter(x[mask_top], y[mask_top], 
                        c=color_map_indices[mask_top], 
                        cmap=cmap, 
                        s=args.markersize, 
                        alpha=0.8, 
                        edgecolors='none')

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Graph Communities (Top {args.top_k} Colored)", color='white', fontsize=16)

    # Optional: Legend for top 5 to check colors? 
    # Usually for 700k nodes a legend is messy, but title is enough.

    plt.tight_layout()
    plt.savefig(args.output, dpi=args.dpi, format='png', bbox_inches='tight', facecolor='black')
    print(f"Saved visualization to {args.output}")

if __name__ == "__main__":
    main()
