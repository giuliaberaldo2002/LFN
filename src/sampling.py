#!/usr/bin/env python3
"""
sampling_midterm_optimized.py
----------------------------
Sampling + evaluation aligned with MidTerm report metrics.

Distance definition:
- use shortest-path distance on the road graph as underlying d(·,·)
  (weighted if edge weights are available, else hop distance). :contentReference[oaicite:1]{index=1}

Metrics (MidTerm):
- Community coverage: |{ i : Ci ∩ S ≠ ∅ }| / k, plus distribution of |Ci ∩ S| :contentReference[oaicite:2]{index=2}
- Global coverage (k-center style): d(x,S)=min_{y∈S} d(x,y); report R=max_x d(x,S), mean/median/p90 :contentReference[oaicite:3]{index=3}
- Diversity among sampled points: pairwise distances d(si,sj), min separation + stats :contentReference[oaicite:4]{index=4}
- Balance across communities: CV = σ(ni)/µ(ni) or entropy-based score :contentReference[oaicite:5]{index=5}

Optimizations:
- Multi-source Dijkstra (single run) to compute d(x,S) for all x and also nearest-center assignment.
- Optional evaluation on a subset of nodes to reduce runtime for huge graphs.

Author: Giulia
"""

import os
import json
import math
import pickle
import argparse
import heapq
from collections import defaultdict, Counter
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import igraph as ig

# Optional ARI/NMI
try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


# ============================================================
# Sampling methods
# ============================================================

def sample_round_robin(g: ig.Graph, k: int, seed: int | None = None) -> list[int]:
    """
    Community-based round-robin sampling WITHOUT duplicates.
    - Requires vertex attribute 'community'.
    - Round-robin over communities: take 1 node per community per pass (without replacement)
      until k samples (or all nodes exhausted).
    """
    if k <= 0 or g.vcount() == 0:
        return []
    if "community" not in g.vs.attribute_names():
        raise ValueError("Vertex attribute 'community' is required for round-robin sampling.")

    rng = np.random.default_rng(seed)

    comm_values = np.array(g.vs["community"], dtype=int)
    groups: dict[int, list[int]] = defaultdict(list)
    for idx, c in enumerate(comm_values):
        groups[int(c)].append(int(idx))

    community_ids = sorted(groups.keys())
    if not community_ids:
        return []

    # Shuffle within each community so we can pop without replacement
    for cid in community_ids:
        rng.shuffle(groups[cid])

    total_available = sum(len(groups[cid]) for cid in community_ids)
    target = min(k, total_available)

    samples: list[int] = []
    while len(samples) < target:
        progressed = False
        for cid in community_ids:
            if len(samples) >= target:
                break
            if groups[cid]:
                samples.append(groups[cid].pop())
                progressed = True
        if not progressed:
            break

    assert len(samples) == len(set(samples)), "BUG: duplicates in round-robin sampler"
    return samples


def fft_sample_graph(
    g: ig.Graph,
    k: int,
    weight_attr: str | None = "length",
    seed_idx: int | None = None,
) -> list[int]:
    """
    Farthest-First Traversal (FFT) on graph shortest-path distances.
    Uses igraph shortest paths each iteration (can be expensive for large k).

    - k: number of samples
    - weight_attr: edge attribute used as distance (e.g. 'length'), or None for unweighted
    - seed_idx: starting node; if None choose max-degree node
    """
    n = g.vcount()
    if n == 0 or k <= 0:
        return []
    k = min(k, n)

    # Edge weights
    if weight_attr is not None and weight_attr in g.es.attribute_names():
        weights = list(map(float, g.es[weight_attr]))
    else:
        weights = None

    if seed_idx is None:
        seed_idx = int(np.argmax(g.degree()))
    seed_idx = int(seed_idx)

    centers: list[int] = [seed_idx]

    dist_to_center = np.array(g.shortest_paths(source=seed_idx, weights=weights)[0], dtype=float)
    dist_to_center = _replace_infinite(dist_to_center)

    while len(centers) < k:
        nxt = int(np.argmax(dist_to_center))
        if nxt in centers:
            dist_to_center[nxt] = -np.inf
            continue
        centers.append(nxt)

        new_d = np.array(g.shortest_paths(source=nxt, weights=weights)[0], dtype=float)
        new_d = _replace_infinite(new_d)
        dist_to_center = np.minimum(dist_to_center, new_d)

    assert len(centers) == len(set(centers)), "BUG: duplicates in FFT sampler"
    return centers


# ============================================================
# Graph distance backend (CSR + multi-source Dijkstra)
# ============================================================

def _replace_infinite(x: np.ndarray) -> np.ndarray:
    x = np.array(x, dtype=float)
    inf = ~np.isfinite(x)
    if inf.any():
        finite = x[~inf]
        mx = np.max(finite) if finite.size else 1.0
        x[inf] = mx * 10.0
    return x


def build_csr_adjacency(g: ig.Graph, weight_attr: str | None = "length") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build CSR adjacency for fast multi-source Dijkstra.

    Returns:
      indptr: (n+1,)
      indices: (2m,) neighbors
      data: (2m,) weights (or 1.0 if unweighted)
    """
    n = g.vcount()
    m = g.ecount()

    edges = np.array(g.get_edgelist(), dtype=np.int32)  # (m,2)
    u0 = edges[:, 0]
    v0 = edges[:, 1]

    # Undirected: store both directions
    u = np.concatenate([u0, v0])
    v = np.concatenate([v0, u0])

    if weight_attr is not None and weight_attr in g.es.attribute_names():
        w0 = np.array(g.es[weight_attr], dtype=np.float64)
        w = np.concatenate([w0, w0])
    else:
        w = np.ones(u.shape[0], dtype=np.float64)

    order = np.argsort(u, kind="mergesort")
    u = u[order]
    v = v[order]
    w = w[order]

    counts = np.bincount(u, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indptr[1:] = np.cumsum(counts)

    indices = v.astype(np.int32, copy=False)
    data = w.astype(np.float64, copy=False)
    return indptr, indices, data


def multisource_dijkstra(
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    sources: List[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-source Dijkstra:
    Computes dist[x] = min_{s in sources} d(s,x) and owner[x] = argmin source (one of the nearest).

    Returns:
      dist: (n,) float
      owner: (n,) int (source id), -1 if unreachable
    """
    n = len(indptr) - 1
    dist = np.full(n, np.inf, dtype=np.float64)
    owner = np.full(n, -1, dtype=np.int32)

    h: list[tuple[float, int]] = []

    for s in sources:
        s = int(s)
        if dist[s] > 0.0:
            dist[s] = 0.0
            owner[s] = s
            heapq.heappush(h, (0.0, s))

    while h:
        d_u, u = heapq.heappop(h)
        if d_u != dist[u]:
            continue

        start, end = indptr[u], indptr[u + 1]
        for ei in range(start, end):
            v = int(indices[ei])
            alt = d_u + float(data[ei])
            if alt < dist[v]:
                dist[v] = alt
                owner[v] = owner[u]
                heapq.heappush(h, (alt, v))

    return dist, owner


# ============================================================
# Midterm metrics
# ============================================================

def metric_community_coverage(labels_full: np.ndarray, sampled_nodes: List[int]) -> Dict[str, Any]:
    """
    coverage = (#communities represented in S) / (#communities total)
    and counts distribution |Ci ∩ S|.
    :contentReference[oaicite:6]{index=6}
    """
    labels_full = np.asarray(labels_full)
    comms_full = np.unique(labels_full)
    n_full = int(len(comms_full))

    if len(sampled_nodes) == 0:
        return {
            "coverage": 0.0,
            "n_communities_full": n_full,
            "n_communities_represented": 0,
            "counts_per_community": {},
        }

    sampled_labels = labels_full[np.array(sampled_nodes, dtype=int)]
    counts = Counter(sampled_labels.tolist())
    n_repr = int(len(counts))
    cov = float(n_repr / n_full) if n_full > 0 else float("nan")

    return {
        "coverage": cov,
        "n_communities_full": n_full,
        "n_communities_represented": n_repr,
        "counts_per_community": dict(counts),
    }


def metric_balance(counts_per_community: Dict[Any, int]) -> Dict[str, float]:
    """
    Balance across communities: CV and entropy_norm.
    :contentReference[oaicite:7]{index=7}
    """
    if not counts_per_community:
        return {"cv": float("nan"), "entropy_norm": float("nan")}

    n = np.array(list(counts_per_community.values()), dtype=float)
    mu = n.mean()
    sigma = n.std()
    cv = float(sigma / mu) if mu > 0 else float("nan")

    p = n / n.sum() if n.sum() > 0 else np.array([])
    if p.size == 0:
        ent_norm = float("nan")
    else:
        ent = -np.sum(p * np.log(p + 1e-12))
        ent_max = math.log(len(p)) if len(p) > 1 else 0.0
        ent_norm = float(ent / ent_max) if ent_max > 0 else 1.0

    return {"cv": cv, "entropy_norm": ent_norm}


def metric_global_coverage_from_dist(dist_to_S: np.ndarray) -> Dict[str, float]:
    """
    Global coverage (k-center style): summarize d(x,S) with R=max, mean/median/p90.
    :contentReference[oaicite:8]{index=8}
    """
    d = np.asarray(dist_to_S, dtype=float)
    d = _replace_infinite(d)

    return {
        "R_max": float(np.max(d)),
        "mean": float(np.mean(d)),
        "median": float(np.median(d)),
        "p90": float(np.percentile(d, 90)),
    }


def metric_diversity_pairwise(
    g: ig.Graph,
    sampled_nodes: List[int],
    weight_attr: str | None = "length",
) -> Dict[str, float]:
    """
    Diversity among sampled points: pairwise shortest-path distances and min separation.
    :contentReference[oaicite:9]{index=9}
    """
    k = len(sampled_nodes)
    if k < 2:
        return {"min_sep": float("nan"), "mean": float("nan"), "median": float("nan"), "p10": float("nan"), "p90": float("nan")}

    if weight_attr is not None and weight_attr in g.es.attribute_names():
        weights = list(map(float, g.es[weight_attr]))
    else:
        weights = None

    S = list(map(int, sampled_nodes))
    D = np.array(g.shortest_paths(source=S, target=S, weights=weights), dtype=float)
    D = _replace_infinite(D)

    iu, ju = np.triu_indices(k, k=1)
    pair = D[iu, ju]

    return {
        "min_sep": float(np.min(pair)),
        "mean": float(np.mean(pair)),
        "median": float(np.median(pair)),
        "p10": float(np.percentile(pair, 10)),
        "p90": float(np.percentile(pair, 90)),
    }


def metric_clustering_agreement(labels_true: np.ndarray, nearest_center_assignment: np.ndarray) -> Dict[str, Optional[float]]:
    """
    Optional (needs sklearn): compare community labels vs nearest-sample induced assignment via ARI/NMI.
    Not required in MidTerm list, but useful sanity check.
    """
    if not _HAS_SKLEARN:
        return {"ari": None, "nmi": None}

    labels_true = np.asarray(labels_true)
    pred = np.asarray(nearest_center_assignment)

    # sklearn expects finite labels; ignore unreachable nodes (-1)
    mask = pred != -1
    if not np.any(mask):
        return {"ari": float("nan"), "nmi": float("nan")}

    ari = adjusted_rand_score(labels_true[mask], pred[mask])
    nmi = normalized_mutual_info_score(labels_true[mask], pred[mask])
    return {"ari": float(ari), "nmi": float(nmi)}


def evaluate_midterm(
    g: ig.Graph,
    labels_full: np.ndarray,
    sampled_nodes: List[int],
    indptr: np.ndarray,
    indices: np.ndarray,
    data: np.ndarray,
    eval_subset: Optional[np.ndarray] = None,
    weight_attr: str | None = "length",
) -> Dict[str, Any]:
    """
    Evaluate ONE sampled set S with MidTerm metrics using optimized multi-source Dijkstra.

    eval_subset: optional array of node indices to approximate global coverage / assignment.
    """
    sampled_nodes = list(map(int, sampled_nodes))

    # Multi-source Dijkstra gives dist to nearest sampled node + "owner" (nearest sampled id)
    dist_all, owner_all = multisource_dijkstra(indptr, indices, data, sources=sampled_nodes)

    if eval_subset is not None:
        eval_subset = np.asarray(eval_subset, dtype=int)
        dist_eval = dist_all[eval_subset]
        owner_eval = owner_all[eval_subset]
        labels_eval = np.asarray(labels_full)[eval_subset]
    else:
        dist_eval = dist_all
        owner_eval = owner_all
        labels_eval = np.asarray(labels_full)

    cov = metric_community_coverage(labels_full, sampled_nodes)
    bal = metric_balance(cov["counts_per_community"])
    glob = metric_global_coverage_from_dist(dist_eval)
    div = metric_diversity_pairwise(g, sampled_nodes, weight_attr=weight_attr)
    agree = metric_clustering_agreement(labels_eval, owner_eval)

    return {
        "community_coverage": cov,
        "balance": bal,
        "global_coverage": glob,
        "diversity": div,
        "clustering_agreement": agree,
    }


# ============================================================
# Experiment runner (optional, but handy)
# ============================================================

def load_graph(pkl_path: str) -> ig.Graph:
    with open(pkl_path, "rb") as f:
        pkg = pickle.load(f)
    return pkg["graph"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Pickle with {'graph': ig.Graph, ...} (use pruned graph)")
    ap.add_argument("--out", required=True, help="Output JSON path for results")
    ap.add_argument("--ks", nargs="+", type=int, required=True, help="List of k values (e.g., 50 100 200)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight_attr", default="length", help="Edge weight attr for shortest paths (default: length)")
    ap.add_argument("--eval_subset", type=int, default=0,
                    help="If >0, evaluate global coverage on a random subset of this size (faster for huge graphs).")
    args = ap.parse_args()

    g = load_graph(args.input)
    if "community" not in g.vs.attribute_names():
        raise ValueError("Graph must have vertex attribute 'community' before running this script.")

    labels = np.array(g.vs["community"], dtype=int)

    # Precompute CSR adjacency once (big win)
    indptr, indices, data = build_csr_adjacency(g, weight_attr=args.weight_attr)

    rng = np.random.default_rng(args.seed)
    eval_subset = None
    if args.eval_subset and args.eval_subset > 0 and args.eval_subset < g.vcount():
        eval_subset = rng.choice(g.vcount(), size=args.eval_subset, replace=False)

    rows = []
    for k in args.ks:
        S_rr = sample_round_robin(g, k, seed=args.seed)
        S_fft = fft_sample_graph(g, k, weight_attr=args.weight_attr, seed_idx=None)

        met_rr = evaluate_midterm(g, labels, S_rr, indptr, indices, data, eval_subset=eval_subset, weight_attr=args.weight_attr)
        met_fft = evaluate_midterm(g, labels, S_fft, indptr, indices, data, eval_subset=eval_subset, weight_attr=args.weight_attr)

        rows.append({
            "k": int(k),
            "round_robin": met_rr,
            "fft": met_fft,
            "sizes": {"round_robin": len(S_rr), "fft": len(S_fft)},
        })

        print(
            f"k={k:5d} | cov(rr)={met_rr['community_coverage']['coverage']:.3f} "
            f"cov(fft)={met_fft['community_coverage']['coverage']:.3f} "
            f"R(rr)={met_rr['global_coverage']['R_max']:.2f} "
            f"R(fft)={met_fft['global_coverage']['R_max']:.2f}"
        )

    out = {
        "meta": {
            "input": os.path.abspath(args.input),
            "weight_attr": args.weight_attr,
            "seed": args.seed,
            "eval_subset": int(args.eval_subset),
            "metrics_reference": "MidTerm Report: sampling metrics (graph shortest-path distance; coverage; k-center; diversity; balance).",
        },
        "rows": rows,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved results to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

