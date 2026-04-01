from annoy import AnnoyIndex
import numpy.typing as npt
import numpy as np
from scipy.sparse import diags, eye, lil_matrix
import tensorly as tl
from typing import List, Literal, Optional, Sequence
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from tensorly.cp_tensor import unfolding_dot_khatri_rao

from utils.tensor_processing import make_mode_laplacian
from utils.utils import detect_anomalies_soft, global_cg_sylvester, optimal_f1_threshold

# Assuming these are available in your local environment
# from models.implementations.lap_reg_cp import graph_regularized_als
# from utils.tensor_processing import make_mode_laplacian
# from utils.utils import detect_anomalies_soft, global_cg_sylvester, optimal_f1_threshold

type Tensor = tl.tensor | npt.NDArray


class MyGRTenDecomp(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rank: int = 5,
        lambdas: Sequence[float] = (0.1, 0.1, 0.1),
        ks: Optional[Sequence[int]] = (5, 5, 5),
        local_threshold: Optional[float] = None,
        threshold: Optional[float] = None,
        recompute_laps: bool = True,  # New flag to trigger internal updates
        measure: Literal[
            "angular", "euclidean", "manhattan", "hamming", "dot"
        ] = "euclidean",
    ):
        self.rank = rank
        self.lambdas = lambdas
        self.ks = ks
        self.local_threshold = local_threshold
        self.recompute_laps = recompute_laps
        self.measure = measure
        self.threshold = threshold

        # Learned attributes
        self.threshold_ = None
        self.X_hat_ = None
        self.laps_ = None
        self.E_ = None

    def fit(self, X: Tensor, y: Optional[Tensor] = None):
        if (self.ks is None) and (self.laps is None):
            raise ValueError("One of 'ks' or 'laps' must be provided.")

        # Pass recompute parameters into the ALS function
        factors, self.E_ = graph_regularized_als(
            X,
            self.rank,
            lmbda=self.lambdas,
            threshold=self.local_threshold,
            ks=self.ks,
            measure=self.measure,
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


def graph_regularized_als(
    tensor,
    rank,
    lmbda=(0.1, 0.1, 0.1),
    n_iter=20,
    verbose=False,
    threshold=None,
    ks: Optional[Sequence[int]] = None,
    measure="euclidean",
):
    weights, factors = tl.decomposition.parafac(
        tensor, rank=rank, n_iter_max=10, init="random"
    )

    M = tensor.copy()
    E = np.zeros_like(M)
    if ks is None:
        ks = [0] * 3
    laps = [
        (
            make_mode_laplacian(tensor=tensor, k=ks[i], mode=i, measure=measure)
            if ks[i] != 0
            else None
        )
        for i in range(3)
    ]

    old_err = 1e10
    tol = 1e-4

    for i in range(n_iter):
        if verbose:
            print(f"Iteration {i}...")

        # Update factors A (0), B (1), C (2)
        for mode in range(3):
            idx = [m for m in range(3) if m != mode]
            G1 = tl.dot(factors[idx[0]].T, factors[idx[0]])
            G2 = tl.dot(factors[idx[1]].T, factors[idx[1]])

            S = G1 * G2
            S = S * (weights[:, None] * weights[None, :])

            mttkrp = unfolding_dot_khatri_rao(M, (weights, factors), mode=mode)

            if laps[mode] is None:
                eps = 1e-8
                factors[mode] = np.linalg.solve(S + eps * np.eye(rank), mttkrp.T).T
            else:
                factors[mode] = global_cg_sylvester(
                    lmbda[mode] * laps[mode],
                    S,
                    mttkrp,
                    max_iter=1000,
                    verbose=verbose,
                    tol=1e-4,
                )

        # Normalize factors
        for r in range(rank):
            norm = 1.0
            for mode in range(3):
                col_norm = np.linalg.norm(factors[mode][:, r]) + 1e-12
                factors[mode][:, r] /= col_norm
                norm *= col_norm
            weights[r] *= norm

        # Update Anomaly Tensor E and Denoised Tensor M
        res = tensor - tl.cp_to_tensor((weights, factors))
        if threshold != 0:
            E = detect_anomalies_soft(res, threshold=threshold)
            M = tensor - E

        err = np.linalg.norm(res)
        delta = np.abs(err - old_err) / (old_err + 1e-12)

        if verbose:
            print(f"Iteration {i}, Reconstruction Error: {err:.6f}, Delta: {delta:.6f}")

        if delta < tol:
            break

        old_err = err

    return (weights, factors), E
