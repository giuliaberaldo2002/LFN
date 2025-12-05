# Urban Pruning & Sampling on Road Networks
## 

This project implements the pipeline described in the report for:

1. Building a cleaned road network graph from an OSM `.pbf` file  
2. Pruning rural areas based on structural + OSM tag features  
3. Detecting urban communities (Leiden)  
4. Sampling diverse locations for a GeoGuessr-style task  

The implementation is split into modular scripts so that each stage can be run and debugged independently.

---

## 0. Project structure

The pipeline is implemented in these scripts:

- `graph_init_corrected.py`  
  Build the initial **road network graph** from an OSM `.pbf` using Pyrosm + OSMnx, convert to `igraph`, and save.

- `urban_pruning.py`  
  Load the processed graph, compute node features (topology + tags), compute an *urbanity score*, cluster with DBSCAN, and **prune rural nodes**.

- `leiden_communities.py`  
  Load the pruned graph, run **Leiden community detection**, and save the graph with a `community` label for each node.

- `sampling.py`  
  Library of **sampling methods** (community-based and Farthest-First Traversal) and **evaluation metrics**.

- `sampling_experiments.py`  
  Script to run experiments with different `k`, using the sampling functions from `sampling_core.py` and printing metrics.

- `README.md`  
  This file, contains a high level descritpion of how this repository is organized

---

## 1. Dependencies

Python packages:

- `pyrosm`
- `osmnx`
- `networkx`
- `igraph`
- `numpy`
- `pandas`
- `scikit-learn`
- `psutil` (for resource logging, optional)
- `pickle` (standard library, no install needed)
- `math`, `collections` (standard library)

Install (example):

```bash
pip3 install pyrosm osmnx networkx igraph numpy pandas scikit-learn psutil
```

--- # TODO: to complete this part

## 2. Stage 1 – Graph construction (```graph_init_corrected.py```)
## 3. Stage 2 – Urban pruning (```urban_pruning.py```)
## 4. Stage 3 - Community detection (```leiden_communities.py```)
## 5. Stage 4 - Sampling (```sampling.py```)
## 6. Stage 5 - Experiments (```sampling_experiments_.py```)

