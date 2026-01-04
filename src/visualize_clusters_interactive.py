import argparse
import pickle
import igraph as ig
import numpy as np
import os
import sys
import folium

DEFAULT_INPUT = "nord_est_with_communities.pkl"
DEFAULT_OUTPUT = "nord_est_map.html"
TOP_K = 50  # Check top 50 communities

def main():
    parser = argparse.ArgumentParser(description="Create Interactive Map of Communities")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input pickled graph file")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output HTML file path")
    parser.add_argument("--top_k", type=int, default=TOP_K, help="Number of communities to marker")
    args = parser.parse_args()

    # 1. Load Graph
    print(f"Loading graph from {args.input}...")
    if not os.path.exists(args.input):
        print(f"ERROR: File {args.input} not found.")
        sys.exit(1)

    with open(args.input, "rb") as f:
        data = pickle.load(f)
    
    if isinstance(data, dict) and "graph" in data:
        g = data["graph"]
    else:
        g = data

    # 2. Extract Data
    if "x" in g.vs.attribute_names() and "y" in g.vs.attribute_names():
        # igraph stores attributes as lists, convert to numpy for speed
        x = np.array(g.vs["x"])
        y = np.array(g.vs["y"])
    elif "lon" in g.vs.attribute_names() and "lat" in g.vs.attribute_names():
        x = np.array(g.vs["lon"])
        y = np.array(g.vs["lat"])
    else:
        print("ERROR: No coordinates found.")
        sys.exit(1)
        
    communities = np.array(g.vs["community"])
    
    # 3. Analyze Top Communities
    unique_comms, counts = np.unique(communities, return_counts=True)
    sorted_indices = np.argsort(-counts)
    top_comm_ids = unique_comms[sorted_indices[:args.top_k]]
    
    print(f"Calculating centroids for top {args.top_k} communities...")
    
    # Initialize map at the mean of all points
    center_lat, center_lon = np.mean(y), np.mean(x)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9, tiles="CartoDB positron")
    
    colors = [
        'red', 'blue', 'green', 'purple', 'orange', 'darkred',
        'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
        'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen',
        'gray', 'black', 'lightgray'
    ]
    
    for i, comm_id in enumerate(top_comm_ids):
        # Mask for nodes in this community
        mask = (communities == comm_id)
        
        # Calculate centroid
        comm_x = x[mask]
        comm_y = y[mask]
        
        centroid_lon = np.mean(comm_x)
        centroid_lat = np.mean(comm_y)
        size = counts[sorted_indices[i]]
        
        color = colors[i % len(colors)]
        
        # Add Marker
        folium.Marker(
            [centroid_lat, centroid_lon],
            popup=f"Rank: {i+1}<br>Community ID: {comm_id}<br>Size: {size} nodes",
            icon=folium.Icon(color=color, icon='info-sign')
        ).add_to(m)
        
        # Optional: Add a Circle to show extent (approximate)
        # Using std dev to estimate spread
        std_lat = np.std(comm_y)
        std_lon = np.std(comm_x)
        # Rough conversion of degrees to meters (very approx)
        radius = np.mean([std_lat, std_lon]) * 111000 
        
        folium.Circle(
            location=[centroid_lat, centroid_lon],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.2
        ).add_to(m)

    print(f"Saving interactive map to {args.output}...")
    m.save(args.output)
    print("Done.")

if __name__ == "__main__":
    main()
