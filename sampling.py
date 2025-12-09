import math
from collections import defaultdict, Counter
from typing import Dict, Any, Optional

import igraph as ig
import numpy as np

# for ARI/NMI:
try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

# ---------------------- Sampling methods ---------------------

def sample_by_community(g: ig.Graph, k: int, seed: int | None = None) -> list[int]:
    """
    Community-based sampling.
    - Requires vertex attribute 'community'.
    - Round-robin over communities, sampling one random node per community
      until we have k nodes.
    """
    import random
    rnd = random.Random(seed)

    comm_values = g.vs["community"]
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, c in enumerate(comm_values):
        groups[int(c)].append(idx)

    community_ids = sorted(groups.keys())
    if not community_ids:
        return []

    samples: list[int] = []
    while len(samples) < k:
        for cid in community_ids:
            if len(samples) >= k:
                break
            verts = groups[cid]
            if not verts:
                continue
            v_idx = rnd.choice(verts)
            samples.append(v_idx)
    return samples


def fft_sample(
    g: ig.Graph,
    k: int,
    weight_attr: str | None = "length",
    seed_idx: int | None = None,
) -> list[int]:
    """
    Farthest-First Traversal (FFT) on graph shortest-path distances.

    - k: number of samples
    - weight_attr: edge attribute used as distance (e.g. 'length' or 'travel_time'),
                   or None for unweighted shortest paths
    - seed_idx: optional starting vertex; if None, choose max-degree vertex
    """
    n = g.vcount()
    if n == 0 or k <= 0:
        return []

    # Edge weights
    if weight_attr is not None and weight_attr in g.es.attribute_names():
        weights = list(map(float, g.es[weight_attr]))
    else:
        weights = None

    # Initial center
    if seed_idx is None:
        degrees = g.degree()
        seed_idx = int(np.argmax(degrees))
    centers: list[int] = [seed_idx]

    # Distances from first center
    dist_to_center = np.array(
        g.shortest_paths(source=seed_idx, weights=weights)[0],
        dtype=float,
    )

    # Replace infinities with large number
    inf_mask = ~np.isfinite(dist_to_center)
    if inf_mask.any():
        finite = dist_to_center[~inf_mask]
        max_finite = np.max(finite) if finite.size > 0 else 1.0
        dist_to_center[inf_mask] = max_finite * 10.0

    # Iteratively add farthest
    while len(centers) < k:
        next_idx = int(np.argmax(dist_to_center))
        centers.append(next_idx)

        new_dists = np.array(
            g.shortest_paths(source=next_idx, weights=weights)[0],
            dtype=float,
        )
        inf_mask = ~np.isfinite(new_dists)
        if inf_mask.any():
            finite = new_dists[~inf_mask]
            max_finite = np.max(finite) if finite.size > 0 else 1.0
            new_dists[inf_mask] = max_finite * 10.0

        dist_to_center = np.minimum(dist_to_center, new_dists)

    return centers


# ---------------------- Metrics ---------------------

# TODO: see if the haversine distance/metrics makes sense 
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """
    Great-circle (Haversine) distance in kilometers between two lat/lon points.
    """
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(
        dlambda / 2.0
    ) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c

def haversine_matrix(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Haversine distance matrix between two sets of points.

    coords1: shape (N1, 2)  -> [lat, lon] in degrees
    coords2: shape (N2, 2)  -> [lat, lon] in degrees

    Returns: matrix of shape (N1, N2) with distances in km.
    """
    R = 6371.0

    lat1 = np.radians(coords1[:, 0])[:, None]
    lon1 = np.radians(coords1[:, 1])[:, None]
    lat2 = np.radians(coords2[:, 0])[None, :]
    lon2 = np.radians(coords2[:, 1])[None, :]

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def mean_pairwise_distance_geo(g: ig.Graph, nodes: list[int]) -> float:
    """
    Mean pairwise straight-line distance (Haversine, km) between sampled nodes.
    Uses vertex attributes 'lat' and 'lon'.

    If you don't like Haversine, you can simply not call this function
    and use graph distances instead.
    """
    if len(nodes) < 2:
        return 0.0
    if "lat" not in g.vs.attribute_names() or "lon" not in g.vs.attribute_names():
        return 0.0

    lat = g.vs["lat"]
    lon = g.vs["lon"]

    dists = []
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            v1 = nodes[i]
            v2 = nodes[j]
            d = haversine_km(lat[v1], lon[v1], lat[v2], lon[v2])
            dists.append(d)

    return float(np.mean(dists)) if dists else 0.0


def mean_pairwise_distance_graph(
    g: ig.Graph,
    nodes: list[int],
    weight_attr: str | None = "length",
) -> float:
    """
    Mean pairwise shortest-path distance between sampled nodes,
    using graph distance (length, travel_time, or unweighted).
    """
    if len(nodes) < 2:
        return 0.0

    if weight_attr is not None and weight_attr in g.es.attribute_names():
        weights = list(map(float, g.es[weight_attr]))
    else:
        weights = None

    dist_matrix = g.shortest_paths(source=nodes, target=nodes, weights=weights)
    dist_matrix = np.array(dist_matrix, dtype=float)

    inf_mask = ~np.isfinite(dist_matrix)
    if inf_mask.any():
        finite = dist_matrix[~inf_mask]
        max_finite = np.max(finite) if finite.size > 0 else 1.0
        dist_matrix[inf_mask] = max_finite * 10.0

    k = len(nodes)
    dists = []
    for i in range(k):
        for j in range(i + 1, k):
            dists.append(dist_matrix[i, j])

    return float(np.mean(dists)) if dists else 0.0

def compute_community_coverage(labels_full: np.ndarray,
                               sampled_indices: np.ndarray) -> Dict[str, Any]:
    """
    labels_full: shape (N,), community label for each point in the full dataset
    sampled_indices: indices of sampled points w.r.t. the full dataset

    Returns dict with:
      - coverage_rate
      - n_communities_full
      - n_communities_represented
      - sampled_counts_per_community (dict: label -> count)
    """
    labels_full = np.asarray(labels_full)
    sampled_indices = np.asarray(sampled_indices)

    communities_full = np.unique(labels_full)
    sampled_labels = labels_full[sampled_indices]

    sampled_counts = {}
    for lbl in sampled_labels:
        sampled_counts[lbl] = sampled_counts.get(lbl, 0) + 1

    n_communities_full = len(communities_full)
    n_communities_repr = len(sampled_counts)

    coverage_rate = n_communities_repr / n_communities_full if n_communities_full > 0 else np.nan

    return {
        "coverage_rate": coverage_rate,
        "n_communities_full": n_communities_full,
        "n_communities_represented": n_communities_repr,
        "sampled_counts_per_community": sampled_counts,
    }

def compute_intra_community_stats(coords_full: np.ndarray,
                                  labels_full: np.ndarray,
                                  sampled_indices: np.ndarray) -> Dict[str, Any]:
    """
    Compute intra-community distances between each point and its community representative.

    coords_full: shape (N, 2)  [lat, lon] in degrees
    labels_full: shape (N,)
    sampled_indices: indices of sampled points (ideally one per community)

    Returns:
      - mean_intra_distance
      - median_intra_distance
      - p90_intra_distance
      - max_intra_distance
      - per_community (dict: label -> {mean, median, max, size})
    """
    coords_full = np.asarray(coords_full)
    labels_full = np.asarray(labels_full)
    sampled_indices = np.asarray(sampled_indices)

    # Build label -> representative index mapping
    rep_map = {}
    for idx in sampled_indices:
        lbl = labels_full[idx]
        # if multiple per community, last one wins; change if you prefer first
        rep_map[lbl] = idx

    all_distances = []
    per_comm_stats = {}

    for lbl, rep_idx in rep_map.items():
        comm_mask = (labels_full == lbl)
        comm_indices = np.where(comm_mask)[0]
        if comm_indices.size == 0:
            continue

        rep_coord = coords_full[rep_idx][None, :]   # shape (1,2)
        comm_coords = coords_full[comm_indices]     # shape (k,2)

        dists = haversine_matrix(comm_coords, rep_coord).ravel()
        all_distances.extend(dists.tolist())

        per_comm_stats[lbl] = {
            "size": int(comm_indices.size),
            "mean_distance": float(np.mean(dists)),
            "median_distance": float(np.median(dists)),
            "max_distance": float(np.max(dists)),
        }

    all_distances = np.asarray(all_distances) if all_distances else np.array([])

    if all_distances.size == 0:
        return {
            "mean_intra_distance": np.nan,
            "median_intra_distance": np.nan,
            "p90_intra_distance": np.nan,
            "max_intra_distance": np.nan,
            "per_community": per_comm_stats,
        }

    return {
        "mean_intra_distance": float(np.mean(all_distances)),
        "median_intra_distance": float(np.median(all_distances)),
        "p90_intra_distance": float(np.percentile(all_distances, 90)),
        "max_intra_distance": float(np.max(all_distances)),
        "per_community": per_comm_stats,
    }

def compute_global_coverage_stats(coords_full: np.ndarray,
                                  sampled_indices: np.ndarray) -> Dict[str, float]:
    """
    For each full point, compute distance to nearest sampled point.
    Then summarize.

    Returns:
      - max_nn_distance
      - mean_nn_distance
      - median_nn_distance
      - p90_nn_distance
    """
    coords_full = np.asarray(coords_full)
    sampled_indices = np.asarray(sampled_indices)
    coords_sampled = coords_full[sampled_indices]

    if coords_sampled.shape[0] == 0:
        return {k: np.nan for k in
                ["max_nn_distance", "mean_nn_distance", "median_nn_distance", "p90_nn_distance"]}

    dists = haversine_matrix(coords_full, coords_sampled)  # shape (N, k)
    nn_dists = dists.min(axis=1)

    return {
        "max_nn_distance": float(np.max(nn_dists)),
        "mean_nn_distance": float(np.mean(nn_dists)),
        "median_nn_distance": float(np.median(nn_dists)),
        "p90_nn_distance": float(np.percentile(nn_dists, 90)),
    }

def compute_sample_diversity_stats(coords_full: np.ndarray,
                                   sampled_indices: np.ndarray) -> Dict[str, float]:
    """
    Look only at sampled points and compute pairwise distances statistics.

    Returns:
      - min_pairwise_distance
      - mean_pairwise_distance
      - median_pairwise_distance
      - p10_pairwise_distance
      - p90_pairwise_distance
    """
    coords_full = np.asarray(coords_full)
    sampled_indices = np.asarray(sampled_indices)
    coords_sampled = coords_full[sampled_indices]

    n = coords_sampled.shape[0]
    if n < 2:
        return {
            "min_pairwise_distance": np.nan,
            "mean_pairwise_distance": np.nan,
            "median_pairwise_distance": np.nan,
            "p10_pairwise_distance": np.nan,
            "p90_pairwise_distance": np.nan,
        }

    dist_mat = haversine_matrix(coords_sampled, coords_sampled)
    # Take only upper triangle without diagonal
    iu, ju = np.triu_indices(n, k=1)
    pairwise = dist_mat[iu, ju]

    return {
        "min_pairwise_distance": float(np.min(pairwise)),
        "mean_pairwise_distance": float(np.mean(pairwise)),
        "median_pairwise_distance": float(np.median(pairwise)),
        "p10_pairwise_distance": float(np.percentile(pairwise, 10)),
        "p90_pairwise_distance": float(np.percentile(pairwise, 90)),
    }

def compute_clustering_agreement(coords_full: np.ndarray,
                                 labels_full: np.ndarray,
                                 sampled_indices: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Induce a clustering from sampled points (nearest sampled point)
    and compare it to the original community labels via ARI/NMI.

    Requires scikit-learn. If not available, returns None for both.

    Returns:
      - ari
      - nmi
    """
    if not _HAS_SKLEARN:
        return {"ari": None, "nmi": None}

    coords_full = np.asarray(coords_full)
    labels_full = np.asarray(labels_full)
    sampled_indices = np.asarray(sampled_indices)
    coords_sampled = coords_full[sampled_indices]

    if coords_sampled.shape[0] == 0:
        return {"ari": np.nan, "nmi": np.nan}

    dists = haversine_matrix(coords_full, coords_sampled)  # (N, k)
    nn_idx_in_sample = dists.argmin(axis=1)                # 0..k-1

    ari = adjusted_rand_score(labels_full, nn_idx_in_sample)
    nmi = normalized_mutual_info_score(labels_full, nn_idx_in_sample)

    return {"ari": float(ari), "nmi": float(nmi)}


def gini_index(counts: list[int]) -> float:
    """
    Gini index of a list of non-negative values.
    Used to measure how evenly samples are distributed across communities.
    """
    arr = np.array(counts, dtype=float)
    if arr.size == 0 or np.all(arr == 0):
        return 0.0

    arr_sorted = np.sort(arr)
    n = len(arr_sorted)
    gini = (2.0 * np.sum((np.arange(1, n + 1) * arr_sorted)) /
            (n * np.sum(arr_sorted)) - (n + 1) / n)
    return float(gini)


def community_coverage_gini(g: ig.Graph, sampled_nodes: list[int]) -> float:
    """
    Gini index over the number of samples per community.
    Lower = more balanced coverage of communities.
    """
    comm_values = g.vs["community"]
    counts = Counter(comm_values[v] for v in sampled_nodes)
    return gini_index(list(counts.values()))


def evaluate_sampling(coords_full: np.ndarray,
                      labels_full: np.ndarray,
                      sampled_indices: np.ndarray) -> Dict[str, Any]:
    """
    Run all evaluation components for one sampling method.
    """
    return {
        "community_coverage": compute_community_coverage(labels_full, sampled_indices),
        "intra_community": compute_intra_community_stats(coords_full, labels_full, sampled_indices),
        "global_coverage": compute_global_coverage_stats(coords_full, sampled_indices),
        "diversity": compute_sample_diversity_stats(coords_full, sampled_indices),
        "clustering_agreement": compute_clustering_agreement(coords_full, labels_full, sampled_indices),
    }


# Example usage:
# metrics_A = evaluate_sampling(coords, labels, sampled_idx_method_A)
# metrics_B = evaluate_sampling(coords, labels, sampled_idx_method_B)
