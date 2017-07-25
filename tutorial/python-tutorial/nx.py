"""doc"""
import networkx as nx
import matplotlib.pyplot as plt

g = nx.Graph()

g.add_node(1)
g.add_node(2)
g.add_nodes_from(range(10, 20))
g.add_edge(1, 2)
g.add_edge(3, 4)

g.add_weighted_edges_from([(1, 3, 0.5), (2, 4, 3)])
g.add_weighted_edges_from([(10, 20, 0.1)])

# pos = nx.spring_layout(g)
pos = nx.spectral_layout(g)

nx.draw(
    g,
    pos,
    with_labels=True,
    node_size=2000,
)
plt.show()
