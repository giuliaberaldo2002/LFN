import pyrosm
import networkx as nx
import osmnx as ox
import igraph as ig
import pandas as pd
import pickle
import time
import sys
import psutil
import os

FP =  "data/bremen-251019.osm.pbf"  # "bremen-251019.osm.pbf" 
OUTPUT_PKL = "bremen_processed_graph.pkl"
NETWORK_TYPE = "driving"

def log_resource_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 ** 3)  # GB
    print(f"   [SYS] RAM utilized: {mem:.2f} GB")

print(f"--- START JOB: Elaboration OSM on cluster node ---")
start_global = time.time()

# ---------------------------------------------------------
# 1. LOADING THE PBF FILE (Pyrosm)
# ---------------------------------------------------------
print("\n[1/4] Processing PBF via Pyrosm...")
t0 = time.time()

osm = pyrosm.OSM(FP)

# Extract nodes and edges with original OSM IDs, in GeoDataFrame format
print("      Extracting nodes and edges from PBF...")
nodes_gdf, edges_gdf = osm.get_network(
    network_type=NETWORK_TYPE, 
    nodes=True
)

# Converting the DataFrames to a NetworkX MultiDiGraph
print("      Converting to Networkx...")
G_nx_raw = osm.to_graph(
    nodes_gdf, 
    edges_gdf, 
    graph_type="networkx",
    network_type=NETWORK_TYPE,
    osmnx_compatible=True  # Set nodes indeces to OSM IDs for compatibility with OSMnx
)


print(f"   Raw graph loaded in time {time.time()-t0:.2f}s")
print(f"   Nodes: {len(G_nx_raw.nodes):,}, Edges: {len(G_nx_raw.edges):,}")
log_resource_usage()

# Liberiamo memoria dai dataframe intermedi
del nodes_gdf, edges_gdf


# ALTERNATIVA PIU EFFICIENTE SECONDO GEMINI, DA VERIFICARE
# # ---------------------------------------------------------
# # 1. LOADING & MANUAL GRAPH BUILD (Memory Optimized)
# # ---------------------------------------------------------
# print("\n[1/4] Processing PBF via Pyrosm (Optimized)...")
# t0 = time.time()

# osm = pyrosm.OSM(FP)

# # A. Filtro "Intelligente": Per GeoGuessr non ci servono strade private o agricole.
# # Riduciamo il dataset alla fonte.
# custom_filter = {
#     "highway": [
#         "motorway", "trunk", "primary", "secondary", "tertiary", 
#         "unclassified", "residential", "motorway_link", "trunk_link",
#         "primary_link", "secondary_link", "tertiary_link"
#     ]
# }

# print("      Extracting ONLY edges (skipping node table)...")
# # TRUCCO: nodes=False. Scarichiamo solo gli archi con la loro geometria.
# # Risparmiamo GB di RAM non caricando i metadati di milioni di punti.
# edges_gdf = osm.get_network(
#     network_type="driving", 
#     nodes=False,
#     extra_attributes=["highway", "maxspeed", "oneway", "lanes", "name"]
# )

# # Filtriamo per tipo di strada (opzionale ma consigliato per risparmiare altro 20%)
# edges_gdf = edges_gdf[edges_gdf["highway"].isin(custom_filter["highway"])]

# print(f"      Edges loaded: {len(edges_gdf):,} rows. Building Graph...")

# # B. Costruzione Manuale del Grafo NetworkX
# # Usiamo from_pandas_edgelist che è molto efficiente
# G_nx_raw = nx.from_pandas_edgelist(
#     edges_gdf, 
#     source='u', 
#     target='v', 
#     edge_attr=True, 
#     create_using=nx.MultiDiGraph
# )

# # C. Recupero Coordinate (Cruciale per OSMnx simplify)
# # Poiché non abbiamo caricato i nodi, dobbiamo "indovinare" le coordinate
# # guardando le estremità delle geometrie degli archi (LineStrings).
# print("      Recovering node coordinates from edge geometries...")

# node_coords = {}
# # Iteriamo sugli archi per popolare le coordinate dei nodi u e v
# # Nota: questo loop impiega qualche minuto ma salva GB di RAM.
# for row in edges_gdf.itertuples(index=False):
#     # row.geometry è una shapely LineString
#     # row.u è il nodo start, row.v è il nodo end
#     if hasattr(row, "geometry") and row.geometry:
#         coords = row.geometry.coords
#         if row.u not in node_coords:
#             node_coords[row.u] = {"x": coords[0][0], "y": coords[0][1]}
#         if row.v not in node_coords:
#             node_coords[row.v] = {"x": coords[-1][0], "y": coords[-1][1]}

# # Assegnamo le coordinate al grafo
# nx.set_node_attributes(G_nx_raw, node_coords)

# # Impostiamo il CRS manualmente (necessario per osmnx)
# G_nx_raw.graph['crs'] = 'epsg:4326'

# print(f"   Raw graph loaded in time {time.time()-t0:.2f}s")
# print(f"   Nodes: {len(G_nx_raw.nodes):,}, Edges: {len(G_nx_raw.edges):,}")
# log_resource_usage()

# # Pulizia immediata
# del edges_gdf, node_coords, osm


# ---------------------------------------------------------
# 2. SEMPLIFICAZIONE TOPOLOGICA (OSMnx)
# ---------------------------------------------------------
print("\n[2/4] Semplificazione Topologica (OSMnx)...")
t0 = time.time()

# Imposta CRS (Pyrosm solitamente non lo setta nel grafo NX, lo forziamo a WGS84)
G_nx_raw.graph['crs'] = 'epsg:4326'

# Semplificazione
G_simplified = ox.simplify_graph(G_nx_raw, remove_rings=False)

# Aggiunta attributi fisici (tempo e velocità)
G_simplified = ox.add_edge_speeds(G_simplified)
G_simplified = ox.add_edge_travel_times(G_simplified)

print(f"   Semplificazione completata in {time.time()-t0:.2f}s")
print(f"   Nodi Finali: {len(G_simplified.nodes):,}, Archi Finali: {len(G_simplified.edges):,}")
log_resource_usage()

del G_nx_raw

# ---------------------------------------------------------
# 3. CONVERSIONE NX -> IGRAPH (Nativa con Clean-up)
# ---------------------------------------------------------
print("\n[3/4] Conversione in iGraph...")
t0 = time.time()

# Pre-pulizia: convertiamo le liste nei tag in stringhe per evitare errori in iGraph
for u, v, data in G_simplified.edges(data=True):
    for k, val in data.items():
        if isinstance(val, list):
            data[k] = ",".join([str(x) for x in val])

# Conversione diretta
g_ig = ig.Graph.from_networkx(G_simplified)

# Creazione mappa ID (Mapping da Indice iGraph -> OSM ID originale)
# 'from_networkx' salva l'ID originale (chiave del dizionario NX) nell'attributo '_nx_name'
osmid_map = {idx: original_id for idx, original_id in enumerate(g_ig.vs["_nx_name"])}

print(f"   Conversione completata in {time.time()-t0:.2f}s")
print(f"   Grafo iGraph: {g_ig.vcount():,} nodi, {g_ig.ecount():,} archi")
log_resource_usage()

# ---------------------------------------------------------
# 4. SALVATAGGIO SERIALIZZATO
# ---------------------------------------------------------
print("\n[4/4] Salvataggio Pickle...")
data_package = {
    "graph": g_ig,
    "osmid_map": osmid_map,
    "crs": "epsg:4326"
}

with open(OUTPUT_PKL, "wb") as f:
    pickle.dump(data_package, f)

print(f"--- JOB COMPLETATO in {(time.time()-start_global)/60:.2f} minuti ---")
print(f"Output salvato in: {os.path.abspath(OUTPUT_PKL)}")