import re
import numpy as np
import matplotlib.pyplot as plt

# from scipy import sparse
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.tenalg import unfolding_dot_khatri_rao
from scipy.linalg import solve_sylvester

from utils.tensor_processing import make_mode_knn, normalize_tensor

T = np.load("data/abiline_ten.npy")
T = T[:, :, :500]
for i in range(12):
    for j in range(12):
        T[i, j, :] = normalize_tensor(T[i, j, :], "minmax")


# 2. Construct the Graph Laplacian for the first mode (A)
# We'll assume the rows of mode 0 have some intrinsic relationship
# Here, we build a k-Nearest Neighbors graph
def make_laplacian(T, mode, k):
    W = make_mode_knn(T, mode=mode, k_neighbors=k)  # AnnoyIndex approximation
    W = 0.5 * (W + W.T)
    deg = np.array(W.sum(axis=1).flatten())
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_inv_sqrt = d_inv_sqrt.ravel()
    D_inv_sqrt = np.diag(d_inv_sqrt)
    I = np.eye(W.shape[0])
    return I - D_inv_sqrt @ W @ D_inv_sqrt


laps = []
for i in range(3):
    laps.append(make_laplacian(T, mode=i, k=10))
    print(laps[i].shape)


# 3. Define the Regularized Loss/Update
# Note: Standard parafac in Tensorly doesn't have a 'laplacian' flag out of the box.
# We can implement a proximal gradient step or use an iterative solver.
# For simplicity, let's use a modified Alternating Least Squares (ALS) logic.


def graph_regularized_als(tensor, rank, laps, lmbda=0.1, n_iter=20):
    shape = tensor.shape
    # 1. Initialize factors and weights
    # We use a 1D array of ones for weights in CP decomposition
    cp_ten = tl.decomposition.parafac(tensor=tensor, rank=rank, n_iter_max=100)
    tensor = tensor.copy()

    for i in range(n_iter):
        print(f"starting iteration {i}...")
        A = cp_ten[1][0]
        B = cp_ten[1][1]
        C = cp_ten[1][2]
        eps = 1e-8

        S1 = (B.T @ B) * (C.T @ C)
        mttkrp1 = unfolding_dot_khatri_rao(tensor, cp_ten, mode=0)
        if laps is None or laps[0] is None:
            A = np.linalg.solve(S1 + eps * np.eye(rank), mttkrp1.T).T
        else:
            A = solve_sylvester(lmbda * laps[0], S1, mttkrp1)
        cp_ten[1][0] = A

        S2 = (A.T @ A) * (C.T @ C)
        mttkrp2 = unfolding_dot_khatri_rao(tensor, cp_ten, mode=1)

        # print(f"B:{B.shape}, L:{laps[1].shape}, mttkrp:{mttkrp2.shape}, S:{S2.shape}")
        if laps is None or laps[1] is None:
            B = np.linalg.solve(S2 + eps * np.eye(rank), mttkrp2.T).T
        else:
            B = solve_sylvester(lmbda * laps[1], S2, mttkrp2)
        cp_ten[1][1] = B

        S3 = (A.T @ A) * (B.T @ B)
        mttkrp3 = unfolding_dot_khatri_rao(tensor, cp_ten, mode=2)
        # C = np.linalg.solve(S3, mttkrp3.T).T
        # print(f"C:{C.shape}, L:{laps[2].shape}, mttkrp:{mttkrp3.shape}, S:{S3.shape}")
        if laps is None or laps[2] is None:
            C = np.linalg.solve(S3 + eps * np.eye(rank), mttkrp3.T).T
        else:
            C = solve_sylvester(lmbda * laps[2], S3, mttkrp3)
        cp_ten[1][2] = C

        # 2. Normalize factors to prevent scale explosion & update weights
        # for mode in range(3):
        #     norms = tl.norm(cp_ten[1][mode], axis=0)
        #     cp_ten[0] = norms
        #     cp_ten[1][mode] = cp_ten[1][mode] / (norms + 1e-9)

        # Optional: Print reconstruction error every 10 iterations
        if i % 1 == 0:
            rec_error = tl.norm(tensor - tl.cp_to_tensor(cp_ten)) / tl.norm(tensor)
            print(f"Iteration {i}: Relative Reconstruction Error = {rec_error:.4f}")
    return cp_ten


# Run the decomposition
rank = 20

# custom_factors = graph_regularized_als(T, rank=rank, laps=laps, lmbda=1, n_iter=100)
# reconst = tl.cp_to_tensor(custom_factors)
# np.save("cache/lap_reg_cp/recost.npy", reconst)

reconst = np.load("cache/lap_reg_cp/recost.npy")

cp_ten = tl.cp_to_tensor(tl.decomposition.parafac(tensor=T, rank=rank, n_iter_max=100))
source, dest = np.random.randint(0, 11), np.random.randint(0, 11)

plt.plot(T[source, dest, :], "--", alpha=0.5, label="original")
plt.plot(reconst[source, dest, :], label="regularized")
plt.plot(cp_ten[source, dest, :], label="simple cp")
plt.legend()
plt.show()


print("Decomposition complete.")
