import gc
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from utils.anomaly_injector import *
from utils.model_eval import *


T = np.load("data/abiline_ten.npy")
T = T[:, :, :300]
for i in range(12):
    for j in range(12):
        T[i, j, :] = normalize_tensor(T[i, j, :], "minmax")

factors = tl.decomposition.parafac(
    T, rank=500, n_iter_max=100, tol=1e-3, verbose=0, init="random"
)
T_hat = tl.cp_to_tensor(factors)

# norm_T = tl.norm(T)
# rec_error = tl.norm(T - T_hat) / norm_T
# print(rec_error)
# exit()


def decomp_by_err(T, max_rank, target_error, init_method):
    T = T.copy()
    # np.random.seed(1)
    norm_T = tl.norm(T)
    for rank in range(1, max_rank + 1):
        factors = tl.decomposition.parafac(
            T, rank=rank, n_iter_max=100, tol=1e-3, verbose=0, init=init_method
        )
        T_hat = tl.cp_to_tensor(factors)

        rec_error = tl.norm(T - T_hat) / norm_T

        # print(f"Rank {rank}: error = {rec_error:.4f}")

        if rec_error < target_error:
            print(f"Stopping at rank {rank}")
            return factors, rank
    return factors, rank


target_error = 0.20
max_rank = 300
original_norm = tl.norm(T)

ranks = []

reconst = tl.zeros_like(T)
for i in range(7):

    init_method = "random"  # "svd" if i == 0 else "random"
    factors, rank = decomp_by_err(
        T, max_rank=max_rank, target_error=target_error, init_method=init_method
    )
    approx = tl.cp_to_tensor(factors)
    reconst += approx
    print(tl.norm(approx) / original_norm)
    T = T - approx
    ranks.append(rank)


print(ranks)
print(sum(ranks))
