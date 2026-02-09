import matplotlib.pyplot as plt
import numpy as np
from annoy import AnnoyIndex
from joblib import Memory

memory = Memory("./cache", verbose=False)

# --- Load data ---
tensor = np.load("./data/abiline_ten.npy")  # shape (nn, nn, nt)
nn, _, nt = tensor.shape
rank = 100
max_iter = 5
l1_reg = 0.01
neighbor_reg = 0.05
k_neighbors = 10

ten_norm = np.sqrt(np.sum(tensor**2, axis=None))


def khatri_rao(X, Y):
    return np.einsum("ik,jk->ijk", X, Y).reshape(X.shape[0] * Y.shape[0], X.shape[1])


@memory.cache
def compute_lsh_neighbors(data_matrix, k_neighbors=10, n_trees=10):
    n_items, dim = data_matrix.shape
    norm_data = data_matrix / (
        np.linalg.norm(data_matrix, axis=1, keepdims=True) + 1e-8
    )

    index = AnnoyIndex(dim, metric="dot")
    for i in range(n_items):
        index.add_item(i, norm_data[i])
    index.build(n_trees)

    neighbors = []
    for i in range(n_items):
        nn_idx = index.get_nns_by_item(i, k_neighbors + 1)
        neighbors.append([j for j in nn_idx if j != i])
    return neighbors


A = np.random.randn(nn, rank)
B = np.random.randn(nn, rank)
C = np.random.randn(nt, rank)

# Mode 0 (A): flatten along first mode slices (i.e., tensor[i,:,:])
A_slices = np.array([tensor[i, :, :].flatten() for i in range(nn)], dtype=np.float32)
neighbors_A = compute_lsh_neighbors(A_slices, k_neighbors)

# Mode 1 (B): flatten along second mode slices (tensor[:, j, :])
B_slices = np.array([tensor[:, j, :].flatten() for j in range(nn)], dtype=np.float32)
neighbors_B = compute_lsh_neighbors(B_slices, k_neighbors)

# Mode 2 (C): flatten along third mode slices (tensor[:, :, t])
C_slices = np.array([tensor[:, :, t].flatten() for t in range(nt)], dtype=np.float32)
neighbors_C = compute_lsh_neighbors(C_slices, k_neighbors)

# --- ALS loop with neighbor regularization ---
for iteration in range(max_iter):
    # Update A
    Z = khatri_rao(C, B)
    tensor_unfold = tensor.transpose(0, 1, 2).reshape(nn, nn * nt)
    lhs_base = Z.T @ Z + l1_reg * np.eye(rank)
    A_new = np.zeros_like(A)
    for i in range(nn):
        neigh = neighbors_A[i]
        if neigh:
            reg_lhs = neighbor_reg * len(neigh) * np.eye(rank)
            reg_rhs = neighbor_reg * np.sum(A[neigh], axis=0)
        else:
            reg_lhs = 0
            reg_rhs = 0
        A_new[i] = np.linalg.solve(lhs_base + reg_lhs, tensor_unfold[i] @ Z + reg_rhs)
    A = A_new

    # Update B
    Z = khatri_rao(C, A)
    tensor_unfold = tensor.transpose(1, 0, 2).reshape(nn, nn * nt)
    lhs_base = Z.T @ Z + l1_reg * np.eye(rank)
    B_new = np.zeros_like(B)
    for i in range(nn):
        neigh = neighbors_B[i]
        if neigh:
            reg_lhs = neighbor_reg * len(neigh) * np.eye(rank)
            reg_rhs = neighbor_reg * np.sum(B[neigh], axis=0)
        else:
            reg_lhs = 0
            reg_rhs = 0
        B_new[i] = np.linalg.solve(lhs_base + reg_lhs, tensor_unfold[i] @ Z + reg_rhs)
    B = B_new

    # Update C
    Z = khatri_rao(B, A)
    tensor_unfold = tensor.transpose(2, 0, 1).reshape(nt, nn * nn)
    lhs_base = Z.T @ Z + l1_reg * np.eye(rank)
    C_new = np.zeros_like(C)
    for t in range(nt):
        neigh = neighbors_C[t]
        if neigh:
            reg_lhs = neighbor_reg * len(neigh) * np.eye(rank)
            reg_rhs = neighbor_reg * np.sum(C[neigh], axis=0)
        else:
            reg_lhs = 0
            reg_rhs = 0
        C_new[t] = np.linalg.solve(lhs_base + reg_lhs, tensor_unfold[t] @ Z + reg_rhs)
    C = C_new

    # Compute reconstruction error (optional)
    recon = np.einsum("ir,jr,kr->ijk", A, B, C)
    error = np.linalg.norm(tensor - recon) / ten_norm
    print(f"Iteration {iteration+1}/{max_iter}, Reconstruction error: {error:.4f}")

recon = np.einsum("ir,jr,kr->ijk", A, B, C)
err = recon - tensor
errors = np.zeros(err.shape[2])
for i in range(err.shape[2]):
    errors[i] = np.linalg.norm(err[:, :, i]) / ten_norm

plt.plot(errors)
plt.show()
