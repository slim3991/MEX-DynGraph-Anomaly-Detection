from functools import cache
import networkx as nx
import numpy as np
import random
import matplotlib.pyplot as plt
from net_traffic_gen import PoissonTraffic

abiline = np.load("./data/abiline_ten.npy")

num_nodes = 50
num_edges = num_nodes + 30
days = 28


@cache
def get_shortest_path(G, start, end) -> list:
    return nx.shortest_path(G, source=start, target=end)


def add_ts(A, path, traffic) -> None:
    for i in range(len(path) - 1):
        source = path[i]
        dest = path[i + 1]
        A[source, dest, :] += traffic


while True:
    G = nx.gnm_random_graph(num_nodes, num_edges)
    if nx.is_connected(G):
        break

pt = PoissonTraffic(days)

adj_mat = np.zeros((num_nodes, num_nodes, days * 24))

for i in range(num_nodes):
    base_scaling = np.random.uniform(100, 800)
    for j in range(num_nodes):
        specific_scaling = np.random.uniform(-base_scaling / 2, base_scaling / 2)
        ts = pt.generate(base_scaling + specific_scaling)
        shortest_path = get_shortest_path(G, i, j)
        add_ts(adj_mat, shortest_path, ts)

plt.ion()
fig, ax = plt.subplots()
vmax = adj_mat.max()  # or a fixed value you choose
vmin = adj_mat.min()  # often 0
# vmax = 2000  # or a fixed value you choose
# vmin = abiline.min()  # often 0
# length = abiline.shape[2]

im = ax.imshow(adj_mat[:, :, 0], vmin=vmin, vmax=vmax, origin="lower")
fig.colorbar(im, ax=ax)

for i in range(200):
    plt.title(f"day: {i//24}, time: {i % 24}")
    im.set_data(adj_mat[:, :, i])
    plt.pause(0.1)
