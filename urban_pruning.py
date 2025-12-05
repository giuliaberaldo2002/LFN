import igraph as ig
import numpy as np
import math
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN  # TODO as a first attempt we can use DBSCAN 
from collections import Counter
import re

# ---------------- HELPER FUNCTIONS ------------------------
# helper function for the edge  label computation
def add_tag_based_node_features(g):
    """
    For each node v, compute features aggregated from incident edges:
    - avg_maxspeed
    - highway-type frequencies: freq_residential, freq_primary, freq_motorway, freq_service, freq_other
    Adds them as vertex attributes on g.
    """
    has_highway = "highway" in g.es.attribute_names()
    has_maxspeed = "maxspeed" in g.es.attribute_names()
    has_speed_kph = "speed_kph" in g.es.attribute_names()

    if not has_highway and not has_maxspeed and not has_speed_kph:
        print("Warning: no 'highway', 'maxspeed' or 'speed_kph' edge attributes found. "
              "Tag-based features will be trivial.")
    
    main_highway_types = ["residential", "primary", "motorway", "service"]

    freq_residential = []
    freq_primary = []
    freq_motorway = []
    freq_service = []
    freq_other = []
    avg_maxspeeds = []

    for v in range(g.vcount()):
        # incident edges of v (in/out/all – we don’t care about direction here)
        inc_edges = g.incident(v, mode="ALL")
        if not inc_edges:
            # isolated node – should be rare
            freq_residential.append(0.0)
            freq_primary.append(0.0)
            freq_motorway.append(0.0)
            freq_service.append(0.0)
            freq_other.append(0.0)
            avg_maxspeeds.append(0.0)
            continue

        hw_list = []
        maxspeeds = []

        for e_idx in inc_edges:
            e = g.es[e_idx]

            # highway tag
            if has_highway:
                hw = e["highway"]
                # may be list or string or None
                if isinstance(hw, list):
                    # take first element for now
                    hw = hw[0] if hw else None
                hw_list.append(str(hw) if hw is not None else "other")
            else:
                hw_list.append("other")

            # maxspeed / speed_kph
            val = None
            if has_maxspeed:
                val = e["maxspeed"]
            elif has_speed_kph:
                val = e["speed_kph"]

            parsed = _parse_maxspeed(val)
            if parsed is not None:
                maxspeeds.append(parsed)

        # highway frequencies
        counts = Counter(hw_list)
        total = sum(counts.values()) or 1  # avoid div by zero

        freq_residential.append(counts.get("residential", 0) / total)
        freq_primary.append(counts.get("primary", 0) / total)
        freq_motorway.append(counts.get("motorway", 0) / total)
        freq_service.append(counts.get("service", 0) / total)

        other_count = sum(
            c for k, c in counts.items() if k not in main_highway_types
        )
        freq_other.append(other_count / total)

        # avg maxspeed
        if maxspeeds:
            avg_maxspeeds.append(float(np.mean(maxspeeds)))
        else:
            avg_maxspeeds.append(0.0)

    # store as vertex attributes
    g.vs["freq_residential"] = freq_residential
    g.vs["freq_primary"] = freq_primary
    g.vs["freq_motorway"] = freq_motorway
    g.vs["freq_service"] = freq_service
    g.vs["freq_other"] = freq_other
    g.vs["avg_maxspeed"] = avg_maxspeeds

    print("Tag-based node features added: freq_*, avg_maxspeed.")

def _parse_maxspeed(val):
    """
    Parse a maxspeed value that may be:
    - a number ("50", 50, 50.0)
    - a string with units ("50 km/h")
    - "none", "signals", etc. -> return None
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        # common OSM patterns
        if val.lower() in {"none", "signals", "variable"}:
            return None
        # extract first number
        m = re.search(r"\d+(\.\d+)?", val)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    # fallback
    return None


# ------------------- START OF FEATURES COMPUTATION ---------------------

# Load the iGraph graph (output of graph_init_corrected.py)
# Ensure the path matches your environment
OUTPUT_PKL = "bremen_processed_graph.pkl" # change depending on the necessity
try:
    with open(OUTPUT_PKL, "rb") as f:
        data_package = pickle.load(f)
    g_ig = data_package["graph"]
    print(f"Graph loaded successfully. Nodes: {g_ig.vcount():,}, Edges: {g_ig.ecount():,}")
except FileNotFoundError:
    print(f"Error: Could not find {OUTPUT_PKL}. Please run the graph_init_corrected.py first.")
    exit()

# check for the edge features
print("Edge attributes:", g_ig.es.attribute_names())

# --- A. Node Degree and Clustering Coefficient (Direct iGraph methods) ---
g_ig.vs["degree"] = g_ig.degree()
g_ig.vs["clustering_coeff"] = g_ig.transitivity_local_undirected()

# manage possible NaN
clust = g_ig.vs["clustering_coeff"]
clust_clean = [0.0 if (c is None or (isinstance(c, float) and math.isnan(c))) else c
               for c in clust]
g_ig.vs["clustering_coeff"] = clust_clean

# --- B. Betweenness Centrality ---
# Use the 'length' attribute as the weight for shortest paths
# NOTE: If 'length' is not present in g_ig.es, this will fail or default to unweighted.
g_ig.vs["betweenness"] = g_ig.betweenness(weights=g_ig.es["length"])


# --- C. Average Edge Length ---
# For each node v, compute: sum(length of incident edges) / degree(v)
avg_edge_lengths = []
for v_idx in range(g_ig.vcount()):
    inc = g_ig.incident(v_idx)
    if not inc:
        avg_edge_lengths.append(0.0)
    else:
        lengths = g_ig.es[inc]["length"]
        avg_edge_lengths.append(float(np.mean(lengths)))
g_ig.vs["avg_edge_len"] = avg_edge_lengths

# ---D. Edge Label ----
add_tag_based_node_features(g_ig)


# Select the computed features
features_to_use = [
    "degree",
    "avg_edge_len",
    "betweenness",
    "clustering_coeff",
    "freq_residential",
    "freq_primary",
    "freq_motorway",
    "freq_service",
    "freq_other",
    "avg_maxspeed",
]

X = np.column_stack([g_ig.vs[feat] for feat in features_to_use])


# Standardize the features (Mean=0, StdDev=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Calculate Urbanity Score (Heuristic Example)  TODO: determine (at least heuristically?) the weights
# Weights (w) must reflect relevance. Example weights:
# High degree, High clustering, High betweenness -> Urban (Positive weight)
# High average length -> Rural (Negative weight)
# example heuristic weights; you can tune later
weights = np.array([
    2.0,   # degree
    -1.0,  # avg_edge_len
    0.5,   # betweenness
    1.5,   # clustering_coeff
    3.0,   # freq_residential (typical urban)
    1.0,   # freq_primary
    -1.0,  # freq_motorway (often less urban / bypass)
    1.0,   # freq_service
    0.5,   # freq_other
    -0.5,  # avg_maxspeed (lower speed -> more urban)
])
weights = weights / np.linalg.norm(weights)

g_ig.vs["urbanity_score"] = X_scaled @ weights


# Apply DBSCAN on the scaled feature matrix X_scaled
# NOTE: The parameters (eps, min_samples) need fine-tuning for your dataset.
db = DBSCAN(eps=0.5, min_samples=10).fit(X_scaled)
g_ig.vs["cluster_label"] = db.labels_

# Determine the 'rural' cluster label:
# The rural cluster will likely be the largest cluster (0 or 1) AND have the 
# lowest average urbanity score or average degree.
cluster_scores = {}
for label in np.unique(g_ig.vs["cluster_label"]):
    if label != -1: # Ignore noise points (-1)
        nodes_in_cluster = g_ig.vs.select(cluster_label=label)
        cluster_scores[label] = np.mean(nodes_in_cluster["urbanity_score"])

# The cluster with the minimum average score is the likely rural cluster
if cluster_scores:
    rural_label = min(cluster_scores, key=cluster_scores.get)
    print(f"Determined rural cluster label: {rural_label}")
    
    # Prune: keep all nodes that are NOT in the rural cluster
    urban_node_indices = g_ig.vs.select(cluster_label_ne=rural_label).indices
else:
    # If clustering failed, fall back to a simple threshold on the score
    print("Clustering failed or found no major clusters. Falling back to thresholding.")
    tau = np.percentile(g_ig.vs["urbanity_score"], 25) # e.g., keep the top 75%
    urban_node_indices = g_ig.vs.select(urbanity_score_ge=tau).indices

# Create the Pruned Subgraph G'
G_prime_ig = g_ig.subgraph(urban_node_indices)

print(f"Original Nodes: {g_ig.vcount():,}")
print(f"Pruned Nodes (V'): {G_prime_ig.vcount():,}")
print(f"Pruning successful. Pruned {g_ig.vcount() - G_prime_ig.vcount():,} nodes.")

with open("bremen_pruned_graph.pkl", "wb") as f:
    pickle.dump({"graph": G_prime_ig}, f)




