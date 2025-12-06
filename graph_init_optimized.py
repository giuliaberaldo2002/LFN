"""
Optimized OSM Graph Initialization
----------------------------------
Extracts road networks from .osm.pbf files using pyrosm, simplifies them with osmnx,
and converts to igraph for high-performance analysis.

Optimizations:
- uses pyrosm native to_graph to ensure topology is correct.
- Aggressive garbage collection.

Author: [Assistant]
"""

import gc
import os
import pickle
import sys
import time
import logging
import psutil
from typing import Dict, Any, Tuple

import pyrosm
import networkx as nx
import osmnx as ox
import igraph as ig
import pandas as pd

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
FP = "bremen-251019.osm.pbf"
OUTPUT_PKL = "bremen_processed_graph.pkl"
NETWORK_TYPE = "driving"

# Helper for memory logging
def log_resource_usage(step_name: str = ""):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)  # Convert to GB
    logger.info(f"[RES] {step_name} - RAM utilized: {mem:.2f} GB")

def load_graph_memory_efficient(filepath: str, network_type: str = "driving") -> nx.MultiDiGraph:
    """
    Loads OSM data using pyrosm.
    """
    logger.info(f"Initializing Pyrosm object for {filepath}...")
    osm = pyrosm.OSM(filepath)
    
    logger.info("Extracting network (nodes=True) for topology...")
    # We MUST load nodes to get the 'u' and 'v' topology columns and coordinates.
    nodes_gdf, edges_gdf = osm.get_network(
        network_type=network_type, 
        nodes=True
    )
    
    logger.info(f"Loaded {len(nodes_gdf):,} nodes and {len(edges_gdf):,} edges.")
    log_resource_usage("After GDF load")

    logger.info("Converting to NetworkX...")
    # osmnx_compatible=True ensures correct CRS and attribute names for OSMnx
    G = osm.to_graph(
        nodes_gdf, 
        edges_gdf, 
        graph_type="networkx",
        network_type=network_type,
        osmnx_compatible=True
    )

    # Release DataFrame memory immediately
    del nodes_gdf, edges_gdf
    del osm
    gc.collect()
    log_resource_usage("After DataFrame cleanup")
    
    return G

def main():
    start_global = time.time()
    logger.info(f"--- START JOB: Graph Init Optimized ---")
    log_resource_usage("Start")

    # 1. LOAD
    try:
        G_raw = load_graph_memory_efficient(FP, NETWORK_TYPE)
    except Exception as e:
        logger.error(f"Failed to load graph: {e}")
        return

    logger.info(f"Raw Graph: {len(G_raw.nodes):,} nodes, {len(G_raw.edges):,} edges")
    
    # Check/Set CRS (Pyrosm usually sets it, but good to ensure for OSMnx)
    if not G_raw.graph.get('crs'):
         G_raw.graph['crs'] = 'epsg:4326'

    # 2. SIMPLIFY
    logger.info("Running OSMnx simplification...")
    # remove_rings=False keeps roundabouts 
    G_simplified = ox.simplify_graph(G_raw, remove_rings=False)
    
    # Add physical attributes (speeds/times) - useful for 'betweenness' weights later
    G_simplified = ox.add_edge_speeds(G_simplified)
    G_simplified = ox.add_edge_travel_times(G_simplified)
    
    del G_raw
    gc.collect()
    log_resource_usage("After Simplification")
    
    logger.info(f"Simplified Graph: {len(G_simplified.nodes):,} nodes, {len(G_simplified.edges):,} edges")

    # 3. CONVERT TO IGRAPH
    logger.info("Converting to igraph...")
    
    # Pre-cleanup: Convert list attributes to strings to avoid iGraph errors
    for u, v, k, data in G_simplified.edges(keys=True, data=True):
        for attr, val in data.items():
            if isinstance(val, list):
                data[attr] = ",".join(map(str, val))

    # Convert
    g_ig = ig.Graph.from_networkx(G_simplified)
    
    # Create mapping from iGraph index to original OSM ID
    if "_nx_name" in g_ig.vs.attribute_names():
        osmid_map = {idx: name for idx, name in enumerate(g_ig.vs["_nx_name"])}
    else:
        # If node names were integers (likely with osmnx_compatible=True they are OSM IDs)
        # igraph stores original NX node IDs in vertex "name" attribute by default? 
        # Actually 'from_networkx' usually puts the NX node ID into '_nx_name'.
        # Let's check 'name' too.
        if "name" in g_ig.vs.attribute_names():
             osmid_map = {idx: name for idx, name in enumerate(g_ig.vs["name"])}
        else:
             logger.warning("Could not preserve OSM IDs properly. Using indices.")
             osmid_map = {idx: idx for idx in range(g_ig.vcount())}

    del G_simplified
    gc.collect()

    # 4. SAVE
    logger.info(f"Saving to {OUTPUT_PKL}...")
    data_package = {
        "graph": g_ig,
        "osmid_map": osmid_map,
        "crs": "epsg:4326"
    }
    
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(data_package, f)
        
    logger.info(f"--- JOB COMPLETATO in {(time.time()-start_global)/60:.2f} min ---")
    logger.info(f"Output: {os.path.abspath(OUTPUT_PKL)}")

if __name__ == "__main__":
    main()
