from annoy import AnnoyIndex
import numpy.typing as npt
import numpy as np
from scipy.sparse import diags, eye, lil_matrix
import tensorly as tl
from typing import List, Literal, Optional, Sequence, Tuple
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted
from tensorly.cp_tensor import unfolding_dot_khatri_rao
from tensorly.tucker_tensor import multi_mode_dot

from utils.tensor_processing import make_mode_laplacian
from utils.utils import detect_anomalies_soft, global_cg_sylvester, optimal_f1_threshold


type Tensor = tl.tensor | npt.NDArray


class MyGRTuckerDecomp(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rank: Tuple[int, int, int] = (5, 5, 5),
        lambdas: Sequence[float] = (0.1, 0.1, 0.1),
        ks: Optional[Sequence[int]] = (5, 5, 5),
        local_threshold: Optional[float] = None,
        threshold: Optional[float] = None,
        measure: Literal[
            "angular", "euclidean", "manhattan", "hamming", "dot"
        ] = "euclidean",
        tol=1e-6,
    ):
        self.rank = rank
        self.lambdas = lambdas
        self.ks = ks
        self.local_threshold = local_threshold
        self.measure = measure
        self.threshold = threshold
        self.tol = tol

        # Learned attributes
        self.threshold_ = None
        self.X_hat_ = None
        self.laps_ = None
        self.E_ = None

    @property
    def name(self):
        return "GRRTucker" if self.local_threshold != 0 else "GRRTucker No Robust"

    def fit(self, X: Tensor, y: Optional[Tensor] = None):

        # Pass recompute parameters into the ALS function
        core, factors, self.E_ = graph_regularized_als(
            X,
            self.rank,
            lmbda=self.lambdas,
            threshold=self.local_threshold,
            ks=self.ks,
            measure=self.measure,
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
    ks,
    measure,
    lmbda=(0.1, 0.1, 0.1),
    n_iter=20,
    tol=1e-6,
    threshold=None,
    verbose=False,
):

    # Initial CP Decomposition
    t_decomp = tl.decomposition.tucker(
        tensor=tensor,
        rank=ranks,
        init="random",
        tol=tol,
    )
    assert t_decomp is not None
    core = t_decomp.core
    factors = t_decomp.factors
    # make Laplacians
    laps: List[Optional[npt.NDArray]] = [
        (
            None
            if ks[0] is None
            else make_mode_laplacian(tensor, mode=0, k=ks[0], measure=measure)
            * lmbda[0]
        ),
        (
            None
            if ks[0] is None
            else make_mode_laplacian(tensor, mode=1, k=ks[1], measure=measure)
            * lmbda[1]
        ),
        (
            None
            if ks[0] is None
            else make_mode_laplacian(tensor, mode=2, k=ks[2], measure=measure)
            * lmbda[2]
        ),
    ]

    M = tensor.copy()
    E = np.zeros_like(M)
    old_err = 1e10
    for i in range(n_iter):
        # --- Update Factor Matrices ---
        for mode in range(len(ranks)):
            # Project tensor onto all OTHER factors
            # This creates a 'partial' reconstruction to isolate the target mode
            other_modes = [m for m in range(len(ranks)) if m != mode]
            Y = multi_mode_dot(
                M, [factors[m].T for m in other_modes], modes=other_modes
            )

            # Compute the 'Tucker MTTKRP' equivalent
            # Flatten Y in current mode and multiply by the flattened core
            Y_n = tl.unfold(Y, mode)
            G_n = tl.unfold(core, mode)

            # This RHS is the target for your Sylvester solver
            B = Y_n @ G_n.T

            # S is the quadratic term from the core tensor
            S = G_n @ G_n.T

            # Use your existing CG Sylvester solver here!
            # It solves: L_mode * factors[mode] + factors[mode] * S = B
            factors[mode] = global_cg_sylvester(laps[mode], S, B, x0=factors[mode])

            # Orthogonalize to keep Tucker stable (optional but recommended)
            factors[mode], _ = np.linalg.qr(factors[mode])

        # --- Update Core Tensor ---
        # core = M x1 A.T x2 B.T x3 C.T
        core = multi_mode_dot(M, [f.T for f in factors])

        # --- Robust Step (Outlier Separation) ---

        X_hat = multi_mode_dot(core, factors)
        res = tensor - X_hat
        if i == 0 or i % 4 == 0:
            # Only update S after initial burn-in to stabilize factors
            if threshold != 0:
                E = detect_anomalies_soft(res, threshold=threshold)
                M = tensor - E
            error = np.linalg.norm(res) / tl.norm(M)
            diff = abs(old_err - error)
            if i > 0 and diff < tol:
                print(f"Converged at iteration {i}")
                break

            old_err = error

        # Your existing soft thresholding/anomaly detection

    return core, factors, E
