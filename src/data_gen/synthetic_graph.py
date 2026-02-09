from functools import cache
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from net_traffic_gen import PoissonTraffic


@cache
def get_shortest_path(G, start, end) -> list:
    return nx.shortest_path(G, source=start, target=end)


def add_ts(A, path, traffic) -> None:
    for i in range(len(path) - 1):
        source = path[i]
        dest = path[i + 1]
        A[source, dest, :] += traffic


class DynamicNetGraphGen:
    def __init__(self, n_nodes: int, n_edges: int, n_days: int):
        self._n_nodes = n_nodes
        self._n_edges = n_edges
        self._n_days = n_days
        self._adj_mat = np.zeros((n_nodes, n_nodes, n_days * 24))
        self._traffic_generator = PoissonTraffic(n_days, 0.02)
        self._generated = False

    def is_generated(self):
        return self._generated

    @property
    def adj_mat(self):
        if self._generated:
            return self._adj_mat
        else:
            raise ValueError("Adjacency matrix not generated!")

    def generate(self):
        while True:
            G = nx.gnm_random_graph(self._n_nodes, self._n_edges)
            if nx.is_connected(G):
                break

        traffic_tensor = np.zeros((self._n_nodes, self._n_nodes, self._n_days * 24))
        for i in range(self._n_nodes):
            base_scaling = np.random.uniform(100, 800)
            for j in range(self._n_nodes):

                specific_scaling = np.random.uniform(
                    -0.5 * base_scaling, 0.5 * base_scaling
                )
                traffic_tensor[i, j] = self._traffic_generator.generate(
                    base_scaling + specific_scaling
                )
        for t in range(self._n_days * 24):
            hour_load = np.zeros((self._n_nodes, self._n_nodes))
            for u, v in G.edges():
                G[u][v]["weight"] = 1.0

            for i in range(self._n_nodes):
                for j in range(self._n_nodes):
                    if i == j:
                        continue

                    traffic = traffic_tensor[i, j, t]

                    # compute shortest path on CURRENT weights
                    path = nx.shortest_path(G, source=i, target=j, weight="weight")

                    # push traffic and update congestion
                    for u, v in zip(path[:-1], path[1:]):
                        hour_load[u, v] += traffic

                        # congestion-aware weight update
                        G[u][v]["weight"] += hour_load[u, v]
            self._adj_mat[:, :, t] = hour_load

        self._generated = True


if __name__ == "__main__":
    gGen = DynamicNetGraphGen(20, 50, 28)
    gGen.generate()
    adj_mat = gGen.adj_mat

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
