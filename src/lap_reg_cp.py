import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl

from tensorly.tenalg import unfolding_dot_khatri_rao
from scipy.linalg import solve_sylvester
from scipy.sparse import diags, eye, csr_matrix

from utils.anomaly_injector import generate_shape, inject_alpha_anomaly
from utils.model_eval import eval_tensor_model
from utils.tensor_processing import de_anomalize_tensor, make_mode_knn, normalize_tensor
from utils.utils import global_cg_sylvester

np.random.seed(42)

T = np.load("data/abiline_ten.npy")
T = T[:, :, :5_000]

for i in range(12):
    for j in range(12):
        T[i, j, :] = normalize_tensor(T[i, j, :], "minmax")

T = de_anomalize_tensor(T, 23)


def make_laplacian(T, mode, k, sparse_output=True):
    """
    Build the normalized graph Laplacian for the k-NN graph of a tensor mode.

    Parameters
    ----------
    T : tensorly tensor or ndarray
        Input tensor.
    mode : int
        Mode along which to compute the Laplacian.
    k : int
        Number of neighbors for k-NN graph.
    sparse_output : bool
        If True, return a sparse Laplacian; otherwise, return dense.

    Returns
    -------
    L : ndarray or csr_matrix
        Normalized Laplacian matrix.
    """
    # Build k-NN adjacency matrix (symmetric)
    W = make_mode_knn(T, mode=mode, k_neighbors=k, sparse=True)
    W = 0.5 * (W + W.T)  # make symmetric

    # Degree vector
    deg = np.array(W.sum(axis=1)).flatten()

    # D^(-1/2)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    if sparse_output:
        D_inv_sqrt = diags(d_inv_sqrt)
        I = eye(W.shape[0], format="csr")
        L = I - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(W.shape[0]) - D_inv_sqrt @ W.toarray() @ D_inv_sqrt

    return L


laps = []
laps.append(make_laplacian(T, mode=0, k=2))
laps.append(make_laplacian(T, mode=1, k=2))
laps.append(make_laplacian(T, mode=2, k=300))
for i in range(3):
    print(laps[i].shape)

# source, dest = 4, 6

source, dest = np.random.randint(0, 11), np.random.randint(0, 11)


# plt.plot(T[source, dest, :], "--", alpha=0.5, label="original")


def graph_regularized_als(tensor, rank, laps, lmbda=0.1, n_iter=20):
    shape = tensor.shape
    # 1. Initialize factors and weights
    # We use a 1D array of ones for weights in CP decomposition
    cp_ten = tl.decomposition.parafac(
        tensor=tensor,
        rank=rank,
        n_iter_max=100,
        init="random",
    )

    # plt.plot(tl.cp_to_tensor(cp_ten)[source, dest, :], label=f"iteration {-1}")

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
            # A = solve_sylvester(lmbda * laps[0], S1, mttkrp1)
            A = global_cg_sylvester(lmbda * laps[0], S1, mttkrp1)
        cp_ten[1][0] = A

        S2 = (A.T @ A) * (C.T @ C)
        mttkrp2 = unfolding_dot_khatri_rao(tensor, cp_ten, mode=1)

        # print(f"B:{B.shape}, L:{laps[1].shape}, mttkrp:{mttkrp2.shape}, S:{S2.shape}")
        if laps is None or laps[1] is None:
            B = np.linalg.solve(S2 + eps * np.eye(rank), mttkrp2.T).T
        else:
            # B = solve_sylvester(lmbda * laps[1], S2, mttkrp2)
            B = global_cg_sylvester(lmbda * laps[1], S2, mttkrp2)
        cp_ten[1][1] = B

        S3 = (A.T @ A) * (B.T @ B)
        mttkrp3 = unfolding_dot_khatri_rao(tensor, cp_ten, mode=2)
        # C = np.linalg.solve(S3, mttkrp3.T).T
        # print(f"C:{C.shape}, L:{laps[2].shape}, mttkrp:{mttkrp3.shape}, S:{S3.shape}")
        if laps is None or laps[2] is None:
            C = np.linalg.solve(S3 + eps * np.eye(rank), mttkrp3.T).T
        else:
            # C = solve_sylvester(lmbda * laps[2], S3, mttkrp3)
            C = global_cg_sylvester(lmbda * laps[2], S3, mttkrp3)
        cp_ten[1][2] = C
        # ---- Normalize factors ----
        col_norms_A = np.linalg.norm(A, axis=0) + eps
        col_norms_B = np.linalg.norm(B, axis=0) + eps
        col_norms_C = np.linalg.norm(C, axis=0) + eps

        # Absorb scaling into weights
        cp_ten[0] *= col_norms_A * col_norms_B * col_norms_C

        # Normalize columns
        A /= col_norms_A
        B /= col_norms_B
        C /= col_norms_C

        # Optional: redistribute weights evenly (more stable)
        weight_root = np.cbrt(cp_ten[0])
        A *= weight_root
        B *= weight_root
        C *= weight_root
        cp_ten[0][:] = 1.0  # reset weights after redistribution

        # Store updated factors
        cp_ten[1] = [A, B, C]
        # Optional: Print reconstruction error every 10 iterations
        if i == n_iter:

            def graph_penalty(factors, laps):
                return sum(
                    np.trace(f.T @ L @ f)
                    for f, L in zip(factors, laps)
                    if L is not None
                )

            obj = tl.norm(
                tensor - tl.cp_to_tensor(cp_ten)
            ) ** 2 + lmbda * graph_penalty(cp_ten[1], laps)
            print(f"Iteration {i}: Objective = {obj:.4f}")
    return cp_ten


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


# Run the decomposition
rank = 15
laps = None
custom_factors = graph_regularized_als(
    T,
    rank=rank,
    laps=laps,
    lmbda=100,
    n_iter=10,
)
reconst = tl.cp_to_tensor(custom_factors)

resid = reconst - T


eval_tensor_model(resid**2, L)
exit()


# np.save("cache/lap_reg_cp/recost.npy", reconst)
# reconst = np.load("cache/lap_reg_cp/recost.npy")

print(f"regularized err: {tl.norm(reconst-T)/tl.norm(T)}")

cp_ten = tl.cp_to_tensor(tl.decomposition.parafac(tensor=T, rank=rank, n_iter_max=100))
print(f"sipmple cp err: {tl.norm(cp_ten-T)/tl.norm(T)}")

plt.plot(reconst[source, dest, :], label="regularized")
# plt.plot(cp_ten[source, dest, :], label="simple cp")
plt.legend()
plt.title(f"source, dest: {(source, dest)}")
plt.show()


print("Decomposition complete.")
