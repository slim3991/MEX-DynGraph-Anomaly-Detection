from logging import warning
from typing import Tuple
import warnings
import numpy as np
from scipy import sparse
from sklearn.metrics import precision_recall_curve

from utils.metrics import Metrics

FIND_COND_NR = True


def optimal_f1_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[float, float]:
    """
    Finds the optimal threshold based on the best F1 score,
    then computes all metrics.
    """
    probs = np.array(probs).flatten()
    y_true = (np.array(y_true).flatten() > 0).astype(int)

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, probs)

    f1_scores = (2 * precision_curve * recall_curve) / (
        precision_curve + recall_curve + 1e-10
    )

    best_idx = np.argmax(f1_scores)

    optimal_threshold = pr_thresholds[min(best_idx, len(pr_thresholds) - 1)]

    return optimal_threshold, f1_scores[best_idx]


def detect_anomalies_soft(
    res, percentile: float | None = None, threshold: float | None = None
):
    abs_res = np.abs(res)
    if threshold is not None:
        lam = threshold
    elif percentile is not None:
        lam = np.percentile(abs_res, percentile)
    else:
        sigma = np.median(abs_res) / 0.6745
        lam = 2.5 * sigma

    E = np.sign(res) * np.maximum(abs_res - lam, 0)
    return E


def gershgorin_bounds(matrix):
    """
    Estimates the lower and upper bounds of eigenvalues using
    Gershgorin Circle Theorem.
    """
    A = np.array(matrix)
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")

    n = A.shape[0]
    min_bound = float("inf")
    max_bound = float("-inf")

    for i in range(n):
        # Center is the diagonal element
        center = A[i, i]

        # Radius is the sum of absolute values of off-diagonal elements in the row
        radius = np.sum(np.abs(A[i, :])) - np.abs(center)

        # Calculate the interval for this specific disc
        current_min = center - radius
        current_max = center + radius

        # Update global bounds
        if current_min < min_bound:
            min_bound = current_min
        if current_max > max_bound:
            max_bound = current_max

    return min_bound, max_bound


def print_approx_cond_number(A, B):
    la, ua = gershgorin_bounds(A.toarray())
    lb, ub = gershgorin_bounds(B)
    print(f"Est condition number: {(ua+ub)/(la+lb)}")


def global_cg_sylvester(A, B, C, x0=None, max_iter=1000, tol=1e-6, verbose=False):
    # Precompute diagonal preconditioner
    dA = A.diagonal()
    dB = B.diagonal()

    # Avoid division by zero
    denom = dA[:, None] + dB[None, :]
    denom[denom == 0] = 1e-12

    def apply_preconditioner(R):
        return R / denom

    if x0 is None:
        X = np.zeros_like(C)
        R = C.copy()
    else:
        X = x0
        R = C - (A @ X + X @ B)

    R = C.copy()
    Z = apply_preconditioner(R)
    P = Z.copy()

    rz_inner = np.vdot(R, Z).real

    for k in range(max_iter):
        W = A @ P + P @ B

        denom_cg = np.vdot(P, W).real
        if denom_cg <= 1e-16:
            if verbose:
                print(f"Breakdown at iter {k}")

            break

        alpha = rz_inner / denom_cg

        X += alpha * P
        R -= alpha * W

        # Check convergence (relative to initial residual)
        if np.vdot(R, R).real <= (tol**2) * np.vdot(C, C).real:
            if verbose:
                print(f"Converged in {k+1} iterations")
            return X

        Z = apply_preconditioner(R)

        rz_new = np.vdot(R, Z).real
        beta = rz_new / rz_inner

        P = Z + beta * P
        rz_inner = rz_new

        if verbose and k % 10 == 0:
            res = np.sqrt(np.vdot(R, R).real)
            print(f"iter {k}, residual {res:.2e}")
    return X
