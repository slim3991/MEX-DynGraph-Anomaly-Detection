from itertools import product
import numpy as np
import tensorly as tl
from tensorly.tenalg import khatri_rao
from scipy.linalg import solve_sylvester

from models.lap_reg_cp import make_laplacian
from utils.tensor_processing import de_anomalize_tensor, normalize_tensor


def soft_thresholding(X, penalty):
    """Soft thresholding operator for L1 regularization."""
    return np.sign(X) * np.maximum(np.abs(X) - penalty, 0)


def graph_regularized_robust_cp(
    X, rank, L_list, alphas, lambda_s, rho=1.0, max_iter=100, tol=1e-4
):
    """
    Graph Regularized Robust CP Decomposition via ADMM.

    Parameters:
    X        : ndarray, the input tensor to be decomposed.
    rank     : int, the number of components (CP rank).
    L_list   : list of ndarrays, the graph Laplacian for each mode.
    alphas   : list of floats, regularization weights for each mode's graph.
    lambda_s : float, weight for the L1 sparsity penalty on S.
    rho      : float, ADMM penalty parameter.
    max_iter : int, maximum number of ADMM iterations.
    tol      : float, tolerance for convergence.

    Returns:
    factors  : list of ndarrays, the estimated CP factor matrices.
    S        : ndarray, the estimated sparse anomaly tensor.
    """
    shape = X.shape
    N = tl.ndim(X)

    # 1. Initialize variables
    # Initialize factor matrices randomly
    factors = [np.random.randn(shape[n], rank) for n in range(N)]

    S = np.zeros_like(X)  # Sparse component
    Y = np.zeros_like(X)  # Dual variable (Lagrange multiplier)

    for iteration in range(max_iter):
        X_prev_reconstructed = tl.cp_to_tensor((np.ones(rank), factors)) + S

        # --- Update S (Sparse Component) ---
        # S = argmin_S lambda_s * ||S||_1 + (rho/2) * ||X - L - S + Y/rho||_F^2
        L_tensor = tl.cp_to_tensor((np.ones(rank), factors))
        Z_S = X - L_tensor + Y / rho
        S = soft_thresholding(Z_S, lambda_s / rho)

        # --- Update Factor Matrices A^(n) ---
        for n in range(N):
            print(L_list)
            # Calculate the Hadamard product of cross-products of all OTHER factors
            V = np.ones((rank, rank))
            for i, f in enumerate(factors):
                if i != n:
                    V = V * (f.T @ f)

            # Khatri-Rao product of all OTHER factors
            KR = khatri_rao(factors, skip_matrix=n)

            # Target matrix for the n-th mode
            Z_A = X - S + Y / rho
            Z_A_unfolded = tl.unfold(Z_A, n)

            # The update for A^(n) requires solving the Sylvester equation:
            # (alpha_n / rho) * L_n * A^(n) + A^(n) * V = Z_A_unfolded * KR
            A_syl = (alphas[n] / rho) * L_list[n]
            B_syl = V
            Q_syl = Z_A_unfolded @ KR

            # Solve A_syl * X + X * B_syl = Q_syl
            factors[n] = solve_sylvester(A_syl, B_syl, Q_syl)

        # --- Update Y (Dual Variable) ---
        L_tensor = tl.cp_to_tensor((np.ones(rank), factors))
        Y = Y + rho * (X - L_tensor - S)

        # --- Check Convergence ---
        reconstructed = L_tensor + S
        error = np.linalg.norm(X_prev_reconstructed - reconstructed) / np.linalg.norm(X)

        if error < tol:
            print(
                f"Converged at iteration {iteration} with relative change: {error:.6e}"
            )
            break

    return factors, S


# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":

    T = np.load("data/abiline_ten.npy")
    # T = T[:, :, :1000]
    T = T[:, :, 10_000:15_000]
    for i, j in product(range(12), repeat=2):
        T[i, j, :] = normalize_tensor(T[i, j, :], "minmax")

    # T = normalize_tensor(T, "minmax")
    source, dest = np.random.randint(0, 11), np.random.randint(0, 11)
    # source, dest = 5, 8

    T = de_anomalize_tensor(T, low_rank=20, keep_pecentile=95, alpha=0.4)

    rank = 4

    laps = []
    laps.append(make_laplacian(T, mode=0, k=11))
    laps.append(make_laplacian(T, mode=1, k=1))
    laps.append(make_laplacian(T, mode=2, k=300))
    alphas = [0.1, 0.1, 0.1]  # Graph regularization weights
    lambda_s = 5.0  # Sparsity weight

    print("Starting optimization...")
    est_factors, est_S = graph_regularized_robust_cp(
        T, rank=rank, L_list=laps, alphas=alphas, lambda_s=lambda_s, rho=1.0
    )

    # Evaluate Results
    est_L = tl.cp_to_tensor((np.ones(rank), est_factors))
    print(
        f"Error in low-rank recovery: {np.linalg.norm(T - est_L) / np.linalg.norm(T):.4f}"
    )
