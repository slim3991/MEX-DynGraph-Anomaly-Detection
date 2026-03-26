import gc
from itertools import product
from math import factorial
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from tqdm import tqdm

# from models.incremental_svd import IncrementalSVD

from utils.tensor_processing import de_anomalize_tensor, normalize_tensor
from utils.anomaly_injector import *
from utils.model_eval import *


T = np.load("data/abiline_ten.npy")
T = T[:, :, :6048]
source, dest = 4, 8
plt.plot(T[source, dest, :])
plt.title(f"Timeseries of OD pair {(source, dest)}")
plt.xlabel("Time")
plt.ylabel("Traffic")
plt.grid()
plt.show()
exit()


for i in range(12):
    for j in range(12):
        T[i, j, :] = normalize_tensor(T[i, j, :], "minmax")


source, dest = np.random.randint(0, 11), np.random.randint(0, 11)

T = de_anomalize_tensor(T, low_rank=20, keep_pecentile=95, alpha=0.4)
L = np.zeros_like(T, dtype=np.bool)

T, L = inject_random_spikes_normal(T, amplitide_factor=5, n_spikes=100)


# plt.plot(T[source, dest, :], label="smoothed-w anomaly")
# plt.show()
# exit()


# Combine masks
print(T.shape)
