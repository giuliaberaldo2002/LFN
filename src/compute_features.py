"""
Compute Graph Features
----------------------
Standalone step 2 in the pipeline (after graph_init).
Loads the raw processed graph, computes node features using graph_features.py,
and saves the graph with features for tuning and pruning.
"""

import os
import sys
import pickle
import logging
import igraph as ig

# Import the shared feature computation logic
import graph_features

# -----------------------
# Logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# -----------------------
# Constants
# -----------------------
import argparse

# -----------------------
# Constants
# -----------------------
# Defaults if not provided via CLI
DEFAULT_INPUT = "bremen_processed_graph.pkl"
DEFAULT_OUTPUT = "bremen_graph_with_features.pkl"

def main():
    parser = argparse.ArgumentParser(description="Compute features for the graph.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input raw graph pickle")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output graph with features pickle")
    args = parser.parse_args()

    logger.info("--- START JOB: Compute Graph Features ---")
    
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"{args.input} not found. Run graph_init_corrected.py first.")
        
    logger.info(f"Loading graph from {args.input}...")
    with open(args.input, "rb") as f:
        data = pickle.load(f)
    
    g = data["graph"]
    osmid_map = data.get("osmid_map", {})
    
    logger.info(f"Graph loaded. Nodes: {g.vcount():,}, Edges: {g.ecount():,}")
    
    # Compute features
    graph_features.add_tag_based_features(g)
    graph_features.compute_topology_features(g)
    
    # Save
    package = {
        "graph": g,
        "osmid_map": osmid_map,
        "crs": data.get("crs", "epsg:4326")
    }
    
    with open(args.output, "wb") as f:
        pickle.dump(package, f)
        
    logger.info(f"Saved graph with features to {os.path.abspath(args.output)}")
    logger.info("--- JOB COMPLETED ---")

if __name__ == "__main__":
    main()
