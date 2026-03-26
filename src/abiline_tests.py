from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import tensorly as tl
import optuna

from models.lap_reg_cp import graph_regularized_als, estimate_from_laps
from utils.tensor_processing import (
    de_anomalize_tensor,
    make_mode_knn,
    make_mode_laplacian,
    normalize_tensor,
)
from utils.anomaly_injector import *
from utils.model_eval import *


T = np.load("data/abiline_ten.npy")
T = T[:, :, : 12 * 24 * 7 * 4]
# T = T[:, :, 10_000:15_000]
for i, j in product(range(12), repeat=2):
    T[i, j, :] = normalize_tensor(T[i, j, :], "minmax")

# T = normalize_tensor(T, "minmax")
source, dest = np.random.randint(0, 11), np.random.randint(0, 11)
# source, dest = 5, 8

T = de_anomalize_tensor(T, low_rank=20, keep_pecentile=95, alpha=0.4)

# best_score =score: {best_score}, best val {best_val}")
# best (7,9,1,3)
rank = 3

laps = []
# laps.append(make_laplacian(T, mode=0, k=11))
# laps.append(make_laplacian(T, mode=1, k=1))
# laps.append(make_laplacian(T, mode=2, k=300))


def plot_regualrizaton_tensor():
    laps = []
    laps.append(make_mode_laplacian(T, mode=0, k=12))
    laps.append(make_mode_laplacian(T, mode=1, k=12))
    laps.append(make_mode_laplacian(T, mode=2, k=500))
    factors = estimate_from_laps(rank=rank, laps=laps, mode_shapes=(0, 1, 2))
    x_hat = tl.cp_to_tensor(factors)

    for i in range(12):
        for j in range(12):
            x_hat[i, j, :] = normalize_tensor(x_hat[i, j, :], "minmax")

    plt.plot(T[source, dest, :], alpha=0.5, label="Original data")
    plt.plot(x_hat[source, dest, :], label="Pure Regularizer (scaled)")
    plt.title("Regularizers pure effect")
    plt.xlabel("Time")
    plt.legend()
    plt.show()


def decomp_recomp(T: tl.tensor, rank: int):
    init = "random" if rank > 12 else "svd"
    weights, factors = tl.decomposition.parafac2(T, rank=rank, verbose=False, init=init)
    reconst = tl.cp_to_tensor(cp_tensor=(weights, factors))
    return reconst


orig_shape = T.shape
# T = np.reshape(T, (12 * 24, 7, -1))
laps = []
laps.append(make_mode_laplacian(T, mode=0, k=3))
laps.append(make_mode_laplacian(T, mode=1, k=3))
laps.append(make_mode_laplacian(T, mode=2, k=30))
factors = estimate_from_laps(laps, 11, mode_shapes=(0, 1, 2))
x_hat = tl.cp_to_tensor(factors)

# x_hat = np.reshape(x_hat, orig_shape)
# T = np.reshape(T, orig_shape)
plt.plot(normalize_tensor(x_hat[3, 5, :], "minmax"))
plt.plot(T[3, 5, :])

plt.show()


print(T.shape)


exit()
sumsum = lambda x: np.sum(np.sum(x, axis=0), axis=0)

weights, factors, _ = tl.decomposition.parafac2(T, rank=6, verbose=False)
print(factors[2].shape)
plt.plot(np.sum(factors[2], axis=1))
plt.plot(sumsum(T))
plt.show()
