from pyrosm import OSM

# 1. Initialize the reader
osm = OSM("nord-est-251217.osm.pbf")

# 2. Extract the network (e.g., all drivable roads)
# Set nodes=True to get the vertex data as well
nodes, edges = osm.get_network(network_type="all", nodes=True)

# 3. Get the counts
num_vertices = len(nodes)
num_edges = len(edges)

print(f"Vertices: {num_vertices}")
print(f"Edges: {num_edges}")