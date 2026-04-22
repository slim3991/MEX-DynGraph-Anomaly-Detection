from collections import defaultdict
import warnings
from annoy import AnnoyIndex
import time
import numpy.typing as npt
import numpy as np
from scipy import sparse
from scipy.sparse import diags, eye, lil_matrix
import tensorly as tl
from typing import Dict, List, Literal, Optional, Sequence
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from tensorly.cp_tensor import unfolding_dot_khatri_rao

from utils.tensor_processing import (
    make_ar_similarity_laplacian,
    make_gaussian_proximity_laplacian,
    make_interval_lap,
    make_mode_laplacian,
)
from utils.utils import detect_anomalies_soft, optimal_f1_threshold


type Tensor = tl.tensor | npt.NDArray


class MyGRTenDecomp(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        laplacian_parameters: Dict[str, float | str],
        rank: int = 5,
        local_threshold: Optional[float] = None,
        threshold: Optional[float] = None,
        recompute_laps: bool = True,  # New flag to trigger internal updates
        measure: Literal[
            "angular", "euclidean", "manhattan", "hamming", "dot"
        ] = "euclidean",
        tol=1e-6,
    ):
        self.rank = rank
        self.laplacian_parameters = laplacian_parameters
        self.local_threshold = local_threshold
        self.recompute_laps = recompute_laps
        self.threshold = threshold
        self.tol = tol

        # Learned attributes
        self.threshold_ = None
        self.X_hat_ = None
        self.laps_ = None
        self.E_ = None

    @staticmethod
    def name():
        return "GR-CP"

    def fit(self, X: Tensor, y: Optional[Tensor] = None):

        # Pass recompute parameters into the ALS function
        factors, self.E_ = graph_regularized_als(
            X,
            self.rank,
            threshold=self.local_threshold,
            laplacian_parameters=self.laplacian_parameters,
            tol=self.tol,
        )

        self.X_hat_ = tl.cp_to_tensor(factors)

        if y is not None:
            self.threshold_, _ = optimal_f1_threshold(self.X_hat_, y)

        return self

    def transform(self, X: Tensor) -> Tensor:
        check_is_fitted(self, ["X_hat_"])
        return self.X_hat_

    def residuals(self, X: Tensor) -> Tensor:
        X_hat = self.transform(X)
        return X - X_hat


def make_laplacians(tensor, lap_param):
    # Mode 0 & 1 (Pre-weighted)
    laps = []
    for m in [0, 1]:
        k_key = f"ks_{m+1}"
        l_key = f"lambda_{m+1}"
        if lap_param.get(k_key, 0) != 0 and lap_param.get(l_key, 0) != 0:
            L = lap_param[l_key] * make_mode_laplacian(
                tensor, mode=m, k=lap_param[k_key], measure=lap_param["measure"]
            )
            laps.append(L)
        else:
            laps.append(None)

    # Mode 2 (Composite Temporal)
    size_3 = tensor.shape[2]
    lap3 = sparse.csr_matrix((size_3, size_3))

    if lap_param.get("lambda_interval", 0) != 0:
        lap3 += lap_param["lambda_interval"] * make_interval_lap(
            size=size_3, interval=lap_param.get("interval", 288)
        )

    if lap_param.get("lambda_smooth", 0) != 0:
        # Note: Ensure make_ar_similarity_laplacian is imported
        lap3 += lap_param["lambda_smooth"] * make_ar_similarity_laplacian(
            size=size_3, lookback=lap_param.get("lookback", 5), decay=0.5
        )

    laps.append(lap3 if lap3.nnz > 0 else None)
    return laps


def graph_regularized_als(
    tensor,
    rank,
    laplacian_parameters,
    n_iter=20,
    tol=1e-6,
    threshold=None,
    verbose=False,
):
    # Dictionary to track cumulative time
    stats = defaultdict(float)

    # Initial CP Decomposition
    weights, factors = tl.decomposition.parafac(
        tensor, rank=rank, tol=tol, init="random"
    )

    laps = make_laplacians(tensor, lap_param=laplacian_parameters)

    M = tensor.copy()
    old_err = 1e10

    for i in range(n_iter):
        for mode in range(3):
            # --- 1. Setup components ---
            start_setup = time.perf_counter()
            idx = [m for m in range(3) if m != mode]
            G1 = np.dot(factors[idx[0]].T, factors[idx[0]])
            G2 = np.dot(factors[idx[1]].T, factors[idx[1]])
            S = G1 * G2 * (weights[:, None] * weights[None, :])

            mttkrp = unfolding_dot_khatri_rao(M, (weights, factors), mode=mode)
            stats["mttkrp_and_setup"] += time.perf_counter() - start_setup

            if laps[mode] is None:
                factors[mode] = np.linalg.solve(S + 1e-8 * np.eye(rank), mttkrp.T).T
            else:
                X = factors[mode]
                X = global_cg_sylvester(
                    A=laps[mode],
                    B=S,
                    C=mttkrp,
                    x0=X,
                    verbose=False,
                    tol=tol,
                )
                factors[mode] = X

        for r in range(rank):
            norm = 1.0
            for mode in range(3):
                col_norm = np.linalg.norm(factors[mode][:, r]) + 1e-12
                factors[mode][:, r] /= col_norm
                norm *= col_norm
            weights[r] *= norm

        if i == 0 or i % 4 == 0:
            X_tensor = tl.cp_to_tensor((weights, factors))
            res = tensor - X_tensor

            if threshold != 0:
                E = detect_anomalies_soft(res, threshold=threshold)
                M = tensor - E

            err = np.linalg.norm(res) / tl.norm(M)
            delta = np.abs(err - old_err)

            if verbose:
                print(f"Iter {i}, Error: {err:.4f}")

            if delta < tol:
                break
            old_err = err

    return (weights, factors), (tensor - tl.cp_to_tensor((weights, factors)))


def global_cg_sylvester(A, B, C, x0=None, max_iter=1000, tol=1e-6, verbose=False):

    t0 = time.perf_counter()
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

    Z = apply_preconditioner(R)
    P = Z.copy()

    rz_inner = np.vdot(R, Z).real
    t_setup = time.perf_counter() - t0
    t1 = time.perf_counter()
    for k in range(max_iter):
        W = A @ P + P @ B

        denom_cg = np.vdot(P, W).real
        if denom_cg <= 1e-16:
            # if verbose:
            #     print(f"Breakdown at iter {k}")

            break

        alpha = rz_inner / denom_cg

        X += alpha * P
        R -= alpha * W

        # Check convergence (relative to initial residual)
        if np.vdot(R, R).real <= (tol**2) * np.vdot(C, C).real:
            if verbose:
                t_loop = time.perf_counter() - t1
                print("=" * 30)
                print(f"Setup: {t_setup:.4f}s | Loop ({k} iters): {t_loop:.4f}s")
                print(f"Converged in {k+1} iterations")
                print("=" * 30 + "\n")
            return X

        Z = apply_preconditioner(R)

        rz_new = np.vdot(R, Z).real
        beta = rz_new / rz_inner

        P = Z + beta * P
        rz_inner = rz_new

        # if verbose and k % 10 == 0:
        #     res = np.sqrt(np.vdot(R, R).real)
        #     print(f"iter {k}, residual {res:.2e}")
    if verbose:
        t_loop = time.perf_counter() - t1
        print(f"Setup: {t_setup:.4f}s | Loop ({k} iters): {t_loop:.4f}s")
    warnings.warn(f"CG did not converge, finished at iteration: {k}")
    return X
