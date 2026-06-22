import networkx as nx

# ==========================================
# TOGGLE GRAPH TYPE HERE: "UNIFORM" or "POWER_LAW"
# ==========================================
GRAPH_TYPE = "POWER_LAW"  

print(f"Generating a {GRAPH_TYPE} graph topology...")

if GRAPH_TYPE == "UNIFORM":
    # 1000 nodes, average degree of roughly 10
    G = nx.erdos_renyi_graph(n=1000, p=0.01, seed=42)
else:
    # 1000 nodes, scale-free network (has hubs)
    G = nx.barabasi_albert_graph(n=1000, m=5, seed=42)

print("Writing edges to edges.txt...")
with open("edges.txt", "w") as f:
    for edge in G.edges():
        f.write(f"{edge[0]} {edge[1]}\n")
        # Ensure undirected edge representation
        f.write(f"{edge[1]} {edge[0]}\n")

print("Done! edges.txt generated successfully.")