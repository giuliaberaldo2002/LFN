import math
from collections import defaultdict, Counter

import igraph as ig
import numpy as np


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
