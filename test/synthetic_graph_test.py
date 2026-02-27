import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from data_gen.synthetic_graph import BurstAnomalyInjector, DynamicNetGraphGen

if __name__ == "__main__":
    n_days = 28
    n_nodes = 20
    burst_anomaly_injector = BurstAnomalyInjector(n_nodes, n_days * 24)

    for _ in range(20):
        burst_anomaly_injector.add_random_anomaly()
    print(burst_anomaly_injector)

    gGen = DynamicNetGraphGen(n_nodes=n_nodes, n_edges=60, n_days=28)
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
