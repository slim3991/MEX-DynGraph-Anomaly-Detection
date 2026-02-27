from typing import Sequence, Tuple
import numpy as np
import numpy.typing as npt
import tensorly as tl


class IncrementalSVD:
    def __init__(self, rank: int, forgetting_factor: float = 1.0) -> None:
        self.U: npt.NDArray | None = None
        self.S: npt.NDArray | None = None
        self.rank = rank
        self.f = forgetting_factor
        self.is_fitted = False
        self.time_since_reorth = 0

    def fit(self, X: npt.NDArray) -> None:
        # Assumes X is (Features x Samples)
        U, S, _ = np.linalg.svd(X, full_matrices=False, compute_uv=True)

        # Handle cases where initial data rank < self.rank
        current_rank = min(self.rank, S.shape[0])

        self.U = U[:, :current_rank]
        self.S = np.diag(S[:current_rank])
        self.is_fitted = True

    def _reoth(self):
        self.U, _ = np.linalg.qr(self.U)

    def increment(self, new: npt.NDArray) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call fit before increment")

        if new.ndim == 1:
            new = new[:, None]
        M = self.U.T @ new
        P = new - self.U @ M
        p_norm = np.linalg.norm(P)
        BK = self.S * self.f
        if p_norm > 1e-10:
            Q, R = np.linalg.qr(P, mode="reduced")

            # K = [ f*S   M ]
            #     [  0    R ]

            top = np.hstack((BK, M))

            bottom = np.hstack((np.zeros((R.shape[0], BK.shape[1])), R))

            K = np.vstack((top, bottom))
            U_hat, S_hat, _ = np.linalg.svd(K, full_matrices=False)

            self.U = np.hstack((self.U, Q)) @ U_hat[:, : self.rank]
            self.S = np.diag(S_hat[: self.rank])

        else:
            K = np.hstack((BK, M))

            U_hat, S_hat, _ = np.linalg.svd(K, full_matrices=False)

            self.U = self.U @ U_hat[:, : self.rank]
            self.S = np.diag(S_hat[: self.rank])
        self.time_since_reorth += 1
        if self.time_since_reorth > 10:
            self._reoth()
            self.time_since_reorth = 0

    def __repr__(self):
        if not self.is_fitted:
            return f"IncrementalSVD(rank={self.rank}, fitted=False)"
        return (
            f"IncrementalSVD(U: {self.U.shape}, S: {self.S.shape}, rank: {self.rank})"
        )
