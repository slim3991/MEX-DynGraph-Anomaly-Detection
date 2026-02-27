import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_gen.net_traffic_gen import PoissonTraffic


if __name__ == "__main__":
    data = np.load("data/abiline_ten.npy")

    pt = PoissonTraffic(14, 0.01)
    for _ in range(10):
        plt.plot(pt.generate(200))

    plt.show()
