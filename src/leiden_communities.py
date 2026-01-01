import igraph as ig
import pickle
import numpy as np

PRUNED_PKL = "bremen_pruned_graph.pkl"  

# ----------------- LOAD GRAPH -----------------
try:
    with open(PRUNED_PKL, "rb") as f:
        data_pkg = pickle.load(f)
    Gp = data_pkg["graph"]
    print(f"Loaded pruned graph. Nodes: {Gp.vcount():,}, Edges: {Gp.ecount():,}")
except FileNotFoundError:
    print(f"ERROR: {PRUNED_PKL} not found. Run the pruning script first.")
    raise SystemExit

# ----------------- CHOOSE WEIGHTS (ONE VARIABLE) -----------------
weights = None
edge_attr_names = Gp.es.attribute_names()

if "length" in edge_attr_names:
    # Option B: use 1 / length as strength (shorter roads = stronger connection)
    lengths = np.array(Gp.es["length"], dtype=float)
    strengths = 1.0 / (lengths + 1e-6)
    weights = strengths.tolist()
    print("Using custom weights: 1 / length")
elif "weight" in edge_attr_names:
    # Fallback: use existing 'weight' attribute by name
    weights = "weight"
    print("Using edge weights attribute: 'weight'")
else:
    print("No 'length' or 'weight' attribute found, running unweighted Leiden.")

# ----------------- HELPER FUNCTIONS -----------------
def summarize_partition(part):
    sizes = np.array(part.sizes())
    return {
        "n_communities": len(part),
        "quality": float(part.q),
        "size_min": int(sizes.min()),
        "size_median": float(np.median(sizes)),
        "size_max": int(sizes.max()),
    }

def run_leiden_cpm_tuning(g, gammas, weights=None, n_iterations=10):
    results = []
    for gamma in gammas:
        print(f"\n=== Gamma = {gamma} ===")
        part = g.community_leiden(
            objective_function="CPM",
            weights=weights,
            resolution_parameter=gamma,
            n_iterations=n_iterations,
        )
        stats = summarize_partition(part)
        results.append({"gamma": gamma, "partition": part, "stats": stats})
        print(
            f" communities: {stats['n_communities']}"
            f" | size[min/med/max] = {stats['size_min']}/"
            f"{stats['size_median']}/{stats['size_max']}"
            f" | quality (q) = {stats['quality']:.4f}"
        )
    return results

# ----------------- GAMMA SWEEP -----------------
GAMMAS = [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]
print("\nRunning Leiden (community_leiden) with CPM for different gamma values...")
results = run_leiden_cpm_tuning(
    Gp,
    gammas=GAMMAS,
    weights=weights,   # <-- same 'weights' we chose above
    n_iterations=10,
)

best = max(results, key=lambda r: r["stats"]["quality"])
best_gamma = best["gamma"]
best_part = best["partition"]
best_stats = best["stats"]

print("\nSelected gamma (max q):", best_gamma)
print(
    f" communities: {best_stats['n_communities']}"
    f" | size[min/med/max] = {best_stats['size_min']}/"
    f"{best_stats['size_median']}/{best_stats['size_max']}"
    f" | quality (q) = {best_stats['quality']:.4f}"
)

# ----------------- USE BEST PARTITION -----------------
# We already have best_part, no need to recompute
part = best_part

Gp.vs["community"] = part.membership
num_comms = len(part)
print(f"\nLeiden finished. Found {num_comms} communities with gamma = {best_gamma}.")

sizes = sorted(part.sizes(), reverse=True)
print("Top 10 community sizes:", sizes[:10])

OUT_PKL = "bremen_pruned_with_communities.pkl"
with open(OUT_PKL, "wb") as f:
    pickle.dump({"graph": Gp}, f)

print(f"Saved graph with community labels to: {OUT_PKL}")
