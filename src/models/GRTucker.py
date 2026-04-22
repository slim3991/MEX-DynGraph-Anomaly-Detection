from annoy import AnnoyIndex
import numpy.typing as npt
import numpy as np
from scipy import sparse
from scipy.sparse import diags, eye, lil_matrix
import tensorly as tl
from typing import Dict, List, Literal, Optional, Sequence, Tuple
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from tensorly.cp_tensor import unfolding_dot_khatri_rao
from tensorly.tucker_tensor import multi_mode_dot

from utils.tensor_processing import (
    make_ar_similarity_laplacian,
    make_interval_lap,
    make_mode_laplacian,
)
from utils.utils import detect_anomalies_soft, global_cg_sylvester, optimal_f1_threshold


type Tensor = tl.tensor | npt.NDArray


def make_laplacians(tensor, lap_param):
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

    size_3 = tensor.shape[2]
    lap3 = sparse.csr_matrix((size_3, size_3))

    if lap_param.get("lambda_interval", 0) != 0:
        lap3 += lap_param["lambda_interval"] * make_interval_lap(
            size=size_3, interval=lap_param.get("interval", 288)
        )

    if lap_param.get("lambda_smooth", 0) != 0:
        lap3 += lap_param["lambda_smooth"] * make_ar_similarity_laplacian(
            size=size_3, lookback=lap_param.get("lookback", 5), decay=0.5
        )

    laps.append(lap3 if lap3.nnz > 0 else None)
    return laps


class MyGRTuckerDecomp(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        laplacian_parameters: Dict[str, float],
        rank: Tuple[int, int, int] = (5, 5, 5),
        local_threshold: Optional[float] = None,
        threshold: Optional[float] = None,
        tol=1e-6,
    ):
        if type(rank) == int:
            self.rank = (rank, rank, rank)
        else:
            self.rank = rank

        self.laplacian_parameters = laplacian_parameters
        self.local_threshold = local_threshold
        self.threshold = threshold
        self.tol = tol

    @staticmethod
    def name():
        return "GR-Tucker"

    def fit(self, X: Tensor, y: Optional[Tensor] = None):
        core, factors, self.E_ = graph_regularized_als(
            X,
            self.rank,
            laplacian_parameters=self.laplacian_parameters,  # Pass the dict
            threshold=self.local_threshold,
            tol=self.tol,
        )
        self.X_hat_ = tl.tenalg.multi_mode_dot(core, factors)
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
    ranks,
    laplacian_parameters,  # Dict replaces individual args
    n_iter=100,
    tol=1e-6,
    threshold=None,
    verbose=False,
):
    t_decomp = tl.decomposition.tucker(
        tensor=tensor, rank=ranks, init="random", tol=tol
    )
    core, factors = t_decomp.core, t_decomp.factors

    # 1. Generate the pre-weighted Laplacians
    laps = make_laplacians(tensor, lap_param=laplacian_parameters)

    M = tensor.copy()
    E = np.zeros_like(M)
    old_err = 1e10

    for i in range(n_iter):
        # --- Update Factor Matrices ---
        for mode in range(len(ranks)):
            other_modes = [m for m in range(len(ranks)) if m != mode]

            # Project tensor onto other factors
            Y = multi_mode_dot(
                M, [factors[m].T for m in other_modes], modes=other_modes
            )
            Y_n = tl.unfold(Y, mode)
            G_n = tl.unfold(core, mode)

            B = Y_n @ G_n.T
            S = G_n @ G_n.T

            # 2. Regularized Update
            if laps[mode] is None:
                # Solve standard least squares: factors[mode] @ S = B
                factors[mode] = np.linalg.solve(S + 1e-8 * np.eye(S.shape[0]), B.T).T
            else:
                # Solve Sylvester: L @ X + X @ S = B
                factors[mode] = global_cg_sylvester(
                    laps[mode], S, B, x0=factors[mode], tol=tol
                )

            # Re-orthogonalize for Tucker stability
            factors[mode], _ = np.linalg.qr(factors[mode])

        # --- Update Core ---
        core = multi_mode_dot(M, [f.T for f in factors])

        # --- Convergence and Anomaly Detection ---
        X_hat = multi_mode_dot(core, factors)
        res = tensor - X_hat
        if i == 0 or i % 4 == 0:
            if threshold != 0:
                E = detect_anomalies_soft(res, threshold=threshold)
                M = tensor - E

            error = np.linalg.norm(res) / tl.norm(M)
            if abs(old_err - error) < tol:
                break
            old_err = error

    return core, factors, E
