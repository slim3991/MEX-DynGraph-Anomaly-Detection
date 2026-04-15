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
        tol=1e-6,
    ):
        self.rank = rank
        self.lambdas = lambdas
        self.ks = ks
        self.local_threshold = local_threshold
        self.recompute_laps = recompute_laps
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
        return "GRRCP"

    def fit(self, X: Tensor, y: Optional[Tensor] = None):

        # Pass recompute parameters into the ALS function
        factors, self.E_ = graph_regularized_als(
            X,
            self.rank,
            lmbda=self.lambdas,
            threshold=self.local_threshold,
            ks=self.ks,
            measure=self.measure,
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


def graph_regularized_als(
    tensor,
    rank,
    ks,
    measure,
    lmbda=(0.1, 0.1, 0.1),
    n_iter=20,
    tol=1e-6,
    threshold=None,
    verbose=False,
):

    # Initial CP Decomposition
    weights, factors = tl.decomposition.parafac(
        tensor, rank=rank, tol=tol, init="random"
    )

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
    old_err = 1e10
    for i in range(n_iter):
        for mode in range(3):
            # 1. Setup Sylvester components
            idx = [m for m in range(3) if m != mode]
            G1 = np.dot(factors[idx[0]].T, factors[idx[0]])
            G2 = np.dot(factors[idx[1]].T, factors[idx[1]])

            S = G1 * G2 * (weights[:, None] * weights[None, :])

            mttkrp = unfolding_dot_khatri_rao(M, (weights, factors), mode=mode)

            # 2. Solve the System
            if laps[mode] is None:
                # Standard ALS step
                factors[mode] = np.linalg.solve(S + 1e-8 * np.eye(rank), mttkrp.T).T
            else:
                # --- INLINED GLOBAL CG SYLVESTER ---
                X = factors[mode]  # Warm start from previous iter

                # Preconditioner setup (Diagonal of Sylvester Operator)
                dA = laps[mode].diagonal()
                dB = S.diagonal()
                denom_pre = dA[:, None] + dB[None, :]
                denom_pre[denom_pre == 0] = 1e-12

                # Initial Residual R = C - (AX + XB)
                # Note: AX is L @ X, XB is X @ S
                R = mttkrp - (laps[mode] @ X + X @ S)
                Z = R / denom_pre
                P = Z.copy()

                rz_inner = np.vdot(R, Z).real

                # Inner CG iterations
                for _ in range(50):
                    W = laps[mode] @ P + P @ S  # Apply Sylvester Operator

                    denom_cg = np.vdot(P, W).real
                    if denom_cg <= 1e-16:
                        break

                    alpha = rz_inner / denom_cg
                    X += alpha * P
                    R -= alpha * W

                    # Convergence check
                    if np.vdot(R, R).real <= (1e-5**2) * np.vdot(mttkrp, mttkrp).real:
                        break

                    Z = R / denom_pre
                    rz_new = np.vdot(R, Z).real
                    P = Z + (rz_new / rz_inner) * P
                    rz_inner = rz_new

                factors[mode] = X

        # 3. Normalization
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

            # Only update S after initial burn-in to stabilize factors
            E = detect_anomalies_soft(res, threshold=threshold)
            M = tensor - E

            err = np.linalg.norm(res)
            delta = np.abs(err - old_err) / (old_err + 1e-12)
            if verbose:
                print(f"Iter {i}, Error: {err:.4f}, Delta: {delta:.6f}")

            if delta < tol:
                break
            old_err = err

    return (weights, factors), (tensor - tl.cp_to_tensor((weights, factors)))
