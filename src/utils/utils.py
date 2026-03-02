import numpy as np
from scipy import sparse


def global_cg_sylvester(A, B, C, max_iter=1000, tol=1e-10):
    """
    Solves AX + XB = C using the Global Conjugate Gradient Method.
    A, B: symmetric matrices (sparse or dense)
    C: dense matrix
    """
    X = np.zeros_like(C)  # dense initialization
    R = C.copy()
    P = R.copy()

    r_norm_sq = np.sum(R * R)

    for i in range(max_iter):
        W = A @ P + P @ B  # matrix multiplication
        p_dot_w = np.sum(P * W)  # Frobenius inner product
        alpha = r_norm_sq / p_dot_w

        X = X + alpha * P
        R_new = R - alpha * W

        r_new_norm_sq = np.sum(R_new * R_new)
        if np.sqrt(r_new_norm_sq) < tol:
            print(f"CG converged in {i+1} iterations.")
            return X

        beta = r_new_norm_sq / r_norm_sq
        P = R_new + beta * P

        R = R_new
        r_norm_sq = r_new_norm_sq

    print("Reached max iterations without full convergence.")
    return X
