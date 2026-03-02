import gc
from math import factorial
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl

# from models.incremental_svd import IncrementalSVD

from utils.tensor_processing import de_anomalize_tensor, normalize_tensor
from utils.anomaly_injector import *
from utils.model_eval import *


T = np.load("data/abiline_ten.npy")
T = T[:, :, :5000]
for i in range(12):
    for j in range(12):
        T[i, j, :] = normalize_tensor(T[i, j, :], "minmax")


source, dest = np.random.randint(0, 11), np.random.randint(0, 11)
# T = de_anomalize_tensor(T, 20)
# L = np.zeros_like(T, dtype=np.bool)
#
# for i in range(120):
#     source, dest = np.random.randint(0, 11), np.random.randint(0, 11)
#     sigma = float(np.std(T[source, dest, :]))
#     duration = np.random.randint(low=10, high=500)
#     start = np.random.randint(low=10, high=20_000)
#     shape = generate_shape(
#         duration=duration,
#         begin_shape="ramp",
#         end_shape="ramp",
#         ratios=(1, 0, 5),
#         amplitude=np.random.uniform(0.3, 1.0),
#     )
#
#     # start = 20_000
#     T[source, dest, start : start + duration] += shape
#     L[source, dest, start : start + duration] = 1
T = de_anomalize_tensor(T, 20)
L = np.zeros_like(T, dtype=np.bool)

for i in range(20):
    source, dest = np.random.randint(0, 11), np.random.randint(0, 11)
    sigma = float(np.std(T[source, dest, :]))
    duration = np.random.randint(low=10, high=100)
    start = np.random.randint(low=10, high=200)
    shape = generate_shape(
        duration=duration,
        begin_shape="ramp",
        end_shape="ramp",
        ratios=(1, 0, 5),
        amplitude=np.random.uniform(0.3, 1.0),
    )

    # start = 20_000
    T[source, dest, start : start + duration] += shape
    L[source, dest, start : start + duration] = 1


# plt.plot(T[source, dest, :], label="smoothed-w anomaly")

# T = normalize_tensor(T, "minmax")

# plt.plot(T[source, dest, :])
# plt.plot(L[source, dest, :])
# plt.show()
# exit()


# Combine masks
print(T.shape)

# T = T.reshape((144, -1))
# T = T.reshape((144, T.shape[1] // (24 * 12), -1))


# def incremental_tucker(T: tl.tensor, ranks):
#     rank_1, rank_2, rank_time = ranks
#
#     iSVD1 = IncrementalSVD(rank=rank_1, forgetting_factor=1)
#     iSVD2 = IncrementalSVD(rank=rank_2, forgetting_factor=1)
#     iSVD3 = IncrementalSVD(rank=rank_time, forgetting_factor=1)
#
#     print("fitting iSVD1")
#     T_unf = tl.base.unfold(T, 0)
#     iSVD1.fit(T_unf[:, :100])
#     for i in range(100, T_unf.shape[1], 100):
#         print(i)
#         iSVD1.increment(T_unf[:, i : i + 100])
#     U = iSVD1.U
#     del iSVD1
#
#     print("fitting iSVD2")
#     T_unf = tl.base.unfold(T, 1)
#     iSVD2.fit(T_unf[:, :100])
#     for i in range(100, T_unf.shape[1], 100):
#         print(i)
#         iSVD2.increment(T_unf[:, i : i + 100])
#     V = iSVD2.U
#     del iSVD2
#
#     print("fitting iSVD3")
#     T_unf = tl.base.unfold(T, 2)
#     iSVD3.fit(T_unf[:, :100])
#     for i in range(100, T_unf.shape[1], 100):
#         print(i)
#         iSVD3.increment(T_unf[:, i : i + 100])
#     W = iSVD3.U
#     del iSVD3
#
#     core = tl.tenalg.multi_mode_dot(T, [U.T, V.T, W.T], modes=[0, 1, 2])
#
#     T_hat = tl.tenalg.multi_mode_dot(core, [U, V, W], modes=[0, 1, 2])
#
#     err = (T_hat - T) ** 2
#
#     del T_hat
#     del T
#     gc.collect()
#     return err


# factors = tl.decomposition.CP(
#     tol=5e-5, rank=20, init="random", verbose=1
# ).fit_transform(T)
# T_reconstructed = tl.cp_to_tensor(factors)
# err = (T_reconstructed - T) ** 2

facotors = tl.decomposition.tucker(T, rank=[9, 9, 9], verbose=1, tol=1e-3)
T_reconstructed = tl.tucker_to_tensor(facotors)
err = (T_reconstructed - T) ** 2


# err = normalize_tensor(err, method="minmax")
# err = np.reshape(err, (12, 12, -1))

# plt.plot(err[source, dest, :], label="error")
# plt.plot(L[source, dest, :], "--", label="anomaly ins")
# plt.xlabel("time")
# plt.legend()
# plt.show()
# exit()

# del T_reconstructed
# del T
# gc.collect()

print(f"Reconstruction error: {np.linalg.norm(err)/np.linalg.norm(T)}")
del T
gc.collect()
eval_tensor_model(err, L)
