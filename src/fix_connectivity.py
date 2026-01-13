
import igraph as ig
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.sparse import csr_matrix
import logging

logger = logging.getLogger(__name__)

def connect_graph_components(g: ig.Graph, weight_attr: str = "length") -> ig.Graph:
    """
    Connects disconnected components of the graph by adding 'virtual' edges 
    between component representatives (centroids).
    
    OPTIMIZED VERSION: Uses cKDTree and sparse matrices to avoid O(N^2) memory usage.
    
    Args:
        g: The input igraph Graph (modified in-place, but returned for convenience).
        weight_attr: The edge attribute to use for weights (e.g., 'length').
    
    Returns:
        The modified graph with added virtual edges.
    """
    if g.vcount() == 0:
        return g

    # 1. Check if connected
    # (igraph's is_connected is fast)
    if g.is_connected():
        logger.info("Graph is already connected. No changes made.")
        return g
    
    # Use connected_components instead of clusters (deprecated)
    components = g.connected_components()
    n_components = len(components)
    logger.info(f"Graph is disconnected ({n_components} components). Connecting components...")

    if n_components < 2:
        return g
    
    # 2. Choose representatives
    if "x" in g.vs.attribute_names() and "y" in g.vs.attribute_names():
        coords = np.column_stack((g.vs["x"], g.vs["y"]))
        is_lonlat = False 
    elif "lon" in g.vs.attribute_names() and "lat" in g.vs.attribute_names():
        coords = np.column_stack((g.vs["lon"], g.vs["lat"]))
        is_lonlat = True
    else:
        logger.warning("No coordinates found. Cannot connect components geometrically.")
        return g

    reps_coords = []
    reps_indices = []
    
    # For very large number of components, this loop is fast enough (46k iters in python is <0.1s)
    for subgraph_indices in components:
        sub_coords = coords[subgraph_indices]
        centroid = np.mean(sub_coords, axis=0)
        # Find node closest to centroid
        # Optimization: Don't compute all dists if component is large? 
        # But components are usually small or checking just one is fine.
        # norm of (N_sub, 2) is cheap.
        dists = np.linalg.norm(sub_coords - centroid, axis=1)
        best_local_idx = np.argmin(dists)
        best_global_idx = subgraph_indices[best_local_idx]
        
        reps_coords.append(coords[best_global_idx])
        reps_indices.append(best_global_idx)
        
    reps_coords = np.array(reps_coords)
    reps_indices = np.array(reps_indices)
    
    # Scaling setup
    # Approximate degrees to meters (at 45 lat) -> ~ 78km lon, 111km lat
    # We use a rough multiplier to keep weights 'reasonable' relative to graph weights.
    scaling_factor = 111000.0 if is_lonlat else 1.0
    tortuosity = 1.5

    # 3. Build Sparse k-NN Graph
    # We want to connect these representatives.
    # Instead of full N^2 matrix, use cKDTree.
    logger.info(f"Building k-NN graph for {n_components} representatives...")
    tree = cKDTree(reps_coords)
    
    # k=10 neighbors should be sufficient to bridge most gaps
    k_neighbors = min(10, n_components) 
    dists, idxs = tree.query(reps_coords, k=k_neighbors)
    
    # Flatten
    row = np.repeat(np.arange(n_components), k_neighbors)
    col = idxs.flatten()
    data = dists.flatten() * scaling_factor * tortuosity
    
    # Create sparse matrix (directed, but we'll symmetrize/use undirected MST)
    sparse_graph = csr_matrix((data, (row, col)), shape=(n_components, n_components))
    
    # 4. Compute MST on Sparse Graph
    # This MST will connect all nodes that are reachable via k-NN
    logger.info("Computing MST on sparse k-NN graph...")
    mst = minimum_spanning_tree(sparse_graph) # Returns CSR matrix
    
    # 5. Check if MST is fully connected (it might be a forest if k-NN is disconnected)
    n_mst_comps, labels = connected_components(mst, directed=False)
    
    edges_to_add = []
    weights_to_add = []
    
    # Add MST edges
    mst_coo = mst.tocoo()
    for i, j, d in zip(mst_coo.row, mst_coo.col, mst_coo.data):
        if i < j:
            edges_to_add.append((reps_indices[i], reps_indices[j]))
            weights_to_add.append(d)

    # 6. Bridge any remaining disconnected components in the MST logic
    if n_mst_comps > 1:
        logger.info(f"k-NN graph forest has {n_mst_comps} components. Bridging forest...")
        
        # We need to connect the forest components. 
        # Pick one "meta-representative" for each forest component.
        meta_reps_indices = [] # Indices into `reps_coords` (0..n_components-1)
        for l in range(n_mst_comps):
            # Pick the first representative that belongs to this label
            # (Arbitrary is fine, we just need *a* point)
            idx = np.where(labels == l)[0][0]
            meta_reps_indices.append(idx)
            
        meta_coords = reps_coords[meta_reps_indices]
        
        # Now we have `n_mst_comps` points. 
        # If this number is distinctively small (likely yes), use full cdist.
        # If it's still huge (unlikely with k=10), use cKDTree again?
        # Let's assume it's small enough for cdist or just linear chain.
        # If > 5000, we fallback to linear chain to be safe on memory.
        
        if n_mst_comps > 5000:
            logger.warning(f"Too many forest components ({n_mst_comps}). Chaining linearly to avoid OOM.")
            # Linear chain: 0-1, 1-2, 2-3...
            for i in range(n_mst_comps - 1):
                u_local = meta_reps_indices[i]
                v_local = meta_reps_indices[i+1]
                
                # Compute distance
                d = np.linalg.norm(reps_coords[u_local] - reps_coords[v_local])
                w = d * scaling_factor * tortuosity * 2.0 # Higher penalty for these "rescue" links
                
                edges_to_add.append((reps_indices[u_local], reps_indices[v_local]))
                weights_to_add.append(w)
        else:
            # Full MST on meta-components
            from scipy.spatial.distance import cdist
            
            # This is (n_mst_comps x n_mst_comps)
            meta_dists = cdist(meta_coords, meta_coords) * scaling_factor * tortuosity * 2.0
            meta_mst = minimum_spanning_tree(csr_matrix(meta_dists))
            meta_mst_coo = meta_mst.tocoo()
            
            for i, j, d in zip(meta_mst_coo.row, meta_mst_coo.col, meta_mst_coo.data):
                if i < j:
                    u_local = meta_reps_indices[i]
                    v_local = meta_reps_indices[j]
                    edges_to_add.append((reps_indices[u_local], reps_indices[v_local]))
                    weights_to_add.append(d)
                    
    # 7. Add all edges to graph
    if edges_to_add:
        g.add_edges(edges_to_add)
        
        eid_start = g.ecount() - len(edges_to_add)
        
        # Verify attribute name
        if weight_attr not in g.es.attribute_names():
            # If graph is unweighted, creating this attribute might be needed or handled
            g.es[weight_attr] = [1.0] * g.ecount() # Init all to 1? Or just handle new ones?
            # If it didn't exist, we probably shouldn't rely on it, but we are supposed to.
            pass

        # Update weights for new edges
        # Note: Depending on igraph version, slicing might work differently? 
        # Safest to assign list.
        # We need to ensure we don't overwrite existing if we initialized the whole array
        # actually, setting slice works in python-igraph.
        
        g.es[eid_start:][weight_attr] = weights_to_add
        g.es[eid_start:]["type"] = "virtual"
        
        logger.info(f"Added {len(edges_to_add)} virtual edges to connect components.")
        
    return g
