import time
import gc
import igraph as ig
from pyrosm import OSM
import numpy as np
import pandas as pd

# Percorso file sul cluster
FILE_PATH = "nord-est-251118.osm.pbf" 
NETWORK_TYPE = "driving"

def generate_graph_hpc():
    print(f"1. [HPC] Inizio lettura PBF: {FILE_PATH}")
    start_time = time.time()
    
    osm = OSM(FILE_PATH)
    
    # Ora possiamo caricare tutto in una volta grazie alla RAM del cluster
    try:
        nodes, edges = osm.get_network(network_type=NETWORK_TYPE, 
                                       nodes=True)
    except MemoryError:
        print("ERRORE: Nemmeno il cluster ha abbastanza RAM allocata. Richiedi più memoria nel job.")
        return None

    print(f"Dati caricati. Nodi grezzi: {len(nodes)}, Archi grezzi: {len(edges)}")

    # 2. Pulizia Colonne (Lo facciamo comunque per velocità di calcolo successiva)
    print("2. Ottimizzazione DataFrame...")
    
    # Teniamo solo le feature utili per il tuo Pruning [cite: 37, 40]
    useful_edge_cols = ['u', 'v', 'length', 'highway', 'maxspeed', 'oneway']
    edges = edges[[c for c in useful_edge_cols if c in edges.columns]].copy()
    
    # Nodi
    nodes = nodes[['id', 'lat', 'lon']].copy()
    
    # Liberiamo subito la memoria dei dataframe originali giganti
    gc.collect()

    # 3. Mappatura ID per igraph (Necessario perché igraph vuole indici 0..N-1)
    print("3. Costruzione indici...")
    u_unique = edges['u'].unique()
    v_unique = edges['v'].unique()
    active_ids = np.unique(np.concatenate([u_unique, v_unique]))
    
    id_mapper = {osm_id: i for i, osm_id in enumerate(active_ids)}
    
    edges['source'] = edges['u'].map(id_mapper)
    edges['target'] = edges['v'].map(id_mapper)
    
    # Filtriamo e ordiniamo i nodi per allinearli agli indici
    nodes.set_index('id', inplace=True)
    nodes_ordered = nodes.loc[active_ids].reset_index()
    
    # 4. Creazione Grafo igraph
    print("4. Generazione Grafo C-based...")
    G = ig.Graph(n=len(active_ids), directed=False)
    G.add_edges(edges[['source', 'target']].values)
    
    # 5. Attributi (Fondamentali per i passaggi successivi del progetto)
    G.es['length'] = edges['length'].values
    if 'highway' in edges.columns: G.es['highway'] = edges['highway'].values
    if 'maxspeed' in edges.columns: G.es['maxspeed'] = edges['maxspeed'].values
    
    G.vs['osm_id'] = nodes_ordered['id'].values
    G.vs['lat'] = nodes_ordered['lat'].values
    G.vs['lon'] = nodes_ordered['lon'].values
    
    print(f"--- SUCCESSO ---")
    print(f"Grafo generato in {time.time() - start_time:.1f}s")
    print(f"Nodi: {G.vcount()}, Archi: {G.ecount()}")
    
    # Esempio salvataggio efficiente (formato pickle di igraph)
    # G.write_pickle("grafo_nordest.pkl") 

if __name__ == "__main__":
    generate_graph_hpc()