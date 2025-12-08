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

edge_attr_names = Gp.es.attribute_names()
if "weight" in edge_attr_names:
    WEIGHTS = "weight"       # name of the attribute, igraph will use it
    print("Using edge weights attribute: 'weight'")
else:
    WEIGHTS = None           # unweighted
    print("No 'weight' attribute found, running unweighted Leiden.")

# Option B: use 1 / length as strength (shorter roads = stronger connection)
if "length" in Gp.es.attribute_names():
    lengths = np.array(Gp.es["length"], dtype=float)
    # avoid division by zero
    strengths = 1.0 / (lengths + 1e-6)
    weights = strengths.tolist()

# ----------- HELPER FUNCTIONS ---------------
def summarize_partition(part):
    """
    part: igraph.clustering.VertexClustering (returned by community_leiden)

    Returns dict with:
      - n_communities
      - quality (objective value, part.q)
      - size_min / size_median / size_max
    """
    sizes = np.array(part.sizes())
    stats = {
        "n_communities": len(part),
        "quality": float(part.q),
        "size_min": int(sizes.min()),
        "size_median": float(np.median(sizes)),
        "size_max": int(sizes.max()),
    }
    return stats

# Run Leiden for parameter tuning of gamma
def run_leiden_cpm_tuning(
    g,
    gammas,
    weights=None,
    n_iterations=10,
):
    """
    g: igraph.Graph
    gammas: iterable of resolution_parameter values
    weights: None or name of edge weight attribute or list of weights
    n_iterations: iterations for community_leiden

    Returns list of dicts:
      { "gamma": ..., "partition": ..., "stats": {...} }
    """
    results = []

    # NB: For now: gamma with highest quality q
    for gamma in gammas:
        print(f"\n=== Gamma = {gamma} ===")
        part = g.community_leiden(
            objective_function="CPM",
            weights=weights,
            resolution_parameter=gamma,
            n_iterations=n_iterations,
        )

        stats = summarize_partition(part)
        results.append({
            "gamma": gamma,
            "partition": part,
            "stats": stats,
        })

        print(
            f" communities: {stats['n_communities']}"
            f" | size[min/med/max] = {stats['size_min']}/"
            f"{stats['size_median']}/{stats['size_max']}"
            f" | quality (q) = {stats['quality']:.4f}"
        )

    return results

GAMMAS = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05] # in general gamma \approx 0.01 so we explore an interval centered around it 
print("\nRunning Leiden (community_leiden) with CPM for different gamma values...")
results = run_leiden_cpm_tuning(
    Gp,
    gammas=GAMMAS,
    weights=WEIGHTS,
    n_iterations=10,
)
best = max(results, key=lambda r: r["stats"]["quality"])
best_gamma = best["gamma"]
best_part = best["partition"]
best_stats = best["stats"]


# Run Leiden community detection
print("Running Leiden community detection...")
part = Gp.community_leiden(
    weights=weights,
    objective_function="CPM",  # or "modularity"
    resolution_parameter=best_gamma,         # Here we use the best gamma according to the tuning
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
