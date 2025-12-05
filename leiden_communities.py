import igraph as ig
import pickle
import numpy as np

# Path of the pruned graph produced by your pruning script
PRUNED_PKL = "bremen_pruned_graph.pkl"  

# Load pruned graph
try:
    with open(PRUNED_PKL, "rb") as f:
        data_pkg = pickle.load(f)
    Gp = data_pkg["graph"]
    print(f"Loaded pruned graph. Nodes: {Gp.vcount():,}, Edges: {Gp.ecount():,}")
except FileNotFoundError:
    print(f"ERROR: {PRUNED_PKL} not found. Run the pruning script first.")
    raise SystemExit

# Choose weights for Leiden (optional)
# Option A: unweighted
weights = None

# Option B: use 1 / length as strength (shorter roads = stronger connection)
if "length" in Gp.es.attribute_names():
    lengths = np.array(Gp.es["length"], dtype=float)
    # avoid division by zero
    strengths = 1.0 / (lengths + 1e-6)
    weights = strengths.tolist()

# Run Leiden community detection
print("Running Leiden community detection...")
part = Gp.community_leiden(
    weights=weights,
    objective_function="modularity",  # or "CPM"
    resolution_parameter=1.0,         # TODO: must be tuned :))
)

Gp.vs["community"] = part.membership
num_comms = len(part)
print(f"Leiden finished. Found {num_comms} communities.")

# (Optional) basic stats
sizes = sorted(part.sizes(), reverse=True)
print("Top 10 community sizes:", sizes[:10])

# Save graph-with-communities
OUT_PKL = "bremen_pruned_with_communities.pkl"
with open(OUT_PKL, "wb") as f:
    pickle.dump({"graph": Gp}, f)

print(f"Saved graph with community labels to: {OUT_PKL}")
