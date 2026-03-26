from itertools import product
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

from utils.model_eval import (
    compute_tensor_model_metrics,
    metrics_to_latex,
)
from utils.tensor_processing import de_anomalize_tensor, normalize_tensor


# def detect_anomalies(S, M_hat, epsilon):
#     residuals = np.abs(S - M_hat)
#     flat_residuals = residuals.flatten()
#
#     if epsilon >= len(flat_residuals):
#         threshold = 0
#     else:
#         threshold = np.partition(flat_residuals, -epsilon)[-epsilon]
#
#     E = np.where(residuals >= threshold, S, 0)
#     anomaly_indices = np.argwhere(residuals >= threshold)
#
#     return E, anomaly_indices


def detect_anomalies_soft(res):
    abs_res = np.abs(res)
    # sigma = np.median(abs_res[abs_res < np.percentile(abs_res, 50)]) / 0.6745

    sigma = np.median(np.abs(res)) / 0.6745
    # lam = 2.5 * sigma
    lam = sigma * np.sqrt(2 * np.log(res.size))

    E = np.sign(res) * np.maximum(np.abs(res) - lam, 0)
    return E


def robust_cp(
    X, rank, n_iter=50, tol=1e-6, n_anomalies=1000, verbose=False, init="svd"
):
    """
    Robust CP decomposition (CP + anomaly separation)

    Parameters
    ----------
    X : ndarray
        Input tensor
    rank : int
        CP rank
    n_iter : int
        Max iterations
    tol : float
        Convergence tolerance
    n_anomalies : int
        Number of anomalies to detect

    Returns
    -------
    weights : ndarray
    factors : list of factor matrices
    S : sparse anomaly tensor
    """

    # Initial CP decomposition
    cp_tensor = tl.decomposition.parafac(X, rank=rank, n_iter_max=10, init=init)
    weights, factors = cp_tensor

    M = X.copy()
    old_error = 1e20
    S = np.zeros_like(M)

    for iteration in range(n_iter):

        # Step 1: CP on cleaned tensor
        cp_tensor = tl.decomposition.parafac(
            M, rank=rank, n_iter_max=10, init=(weights, factors)
        )
        weights, factors = cp_tensor

        # Reconstruction
        X_hat = tl.cp_to_tensor((weights, factors))
        residuals = X - X_hat

        # Error
        error = np.linalg.norm(residuals) / tl.norm(M)
        diff = abs(old_error - error)

        # Step 2: anomaly detection (same as your version)
        S = detect_anomalies_soft(residuals)

        # Update clean tensor
        M = X - S

        if verbose and iteration % 10 == 0:
            print("diff:", diff)
            print("iteration:", iteration)

        if diff < tol:
            if verbose:
                print("reached tol")
            break

        old_error = error

    return weights, factors, S


##################################
