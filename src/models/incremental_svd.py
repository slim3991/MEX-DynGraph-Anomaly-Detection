from typing import Sequence, Tuple
import numpy as np
import numpy.typing as npt
import tensorly as tl


class IncrementalSVD:

    def __init__(self, rank: int, forgetting_factor: float = 1) -> None:
        self.U: npt.NDArray
        self.S: npt.NDArray
        self.rank = rank
        self.f = forgetting_factor

        self.is_fitted = False

    def fit(self, X: npt.NDArray) -> None:
        U, S, _ = np.linalg.svd(X, full_matrices=False, compute_uv=True)

        self.U = U[:, : self.rank]
        self.S = np.diag(S[: self.rank])

        self.is_fitted = True

    def increment(self, new: npt.NDArray) -> None:
        if not self.is_fitted:
            raise RuntimeError("Call fit before increment")

        if new.ndim == 1:
            new = new[:, None]

        M = self.U.T @ new
        P = new - self.U @ M
        if np.linalg.norm(P) < 1e-10:
            return
        Q, R = np.linalg.qr(P, mode="reduced")

        top = np.hstack((self.S * self.f, M))
        bottom = np.hstack((np.zeros((R.shape[0], self.S.shape[1])), R))
        K = np.vstack((top, bottom))

        U_hat, S_hat, _ = np.linalg.svd(K, full_matrices=False, compute_uv=True)

        self.U = np.hstack((self.U, Q)) @ U_hat[:, : self.rank]
        self.S = np.diag(S_hat[: self.rank])

    def __repr__(self):
        return f"U: {self.U.shape}, S: {self.U.shape}, rank: {self.rank}"


class IncrementalHOSVD:

    def __init__(self, rank: int):
        self.rank = rank
        self.U1: tl.tensor = None
        self.U2: tl.tensor = None
        self.U3: tl.tensor = None
        self.S1: tl.tensor = None
        self.S2: tl.tensor = None
        self.S3: tl.tensor = None
        self.is_fitted = False

    def fit(self, T: tl.tensor) -> None:
        if T.ndim == 2:
            T = tl.reshape(T, (T.shape[0], T.shape[1], 1))

        unfold_u = tl.unfold(T, 0)
        self.U1, S1, _ = tl.svd_interface(unfold_u, n_eigenvecs=self.rank)

        unfold_u = tl.unfold(T, 1)
        self.U2, S2, _ = tl.svd_interface(unfold_u, n_eigenvecs=self.rank)

        unfold_u = tl.unfold(T, 2)
        self.U3, S3, _ = tl.svd_interface(unfold_u, n_eigenvecs=self.rank)

        self.is_fitted = True

    def increment(self, new: tl.tensor):
        if not self.is_fitted:
            raise RuntimeError("Call fit before increment")

        self.U1 = increment(self.U1, tl.unfold(new, 0))
        self.U2 = increment(self.U2, tl.unfold(new, 1))
        self.U3 = increment(self.U3, tl.unfold(new, 2))
        self.S = tl.tenalg.multi_mode_dot(new, [self.U1.T, self.U2.T, self.U2.T])


def increment(U: tl.tensor, new: tl.tensor) -> tl.tensor:
    if new.ndim == 2:
        new = tl.reshape(new, (new.shape[0], new.shape[1], 1))
    rank = U.shape[1]

    M = U.T @ new
    P = new - U @ M
    Q, R = tl.qr(P, mode="reduced")

    top = tl.concatenate([tl.eye(U.shape[1]), M], axis=1)
    bottom = tl.concatenate([tl.zeros((R.shape[0], U.shape[1])), R], axis=1)
    K = tl.concatenate([top, bottom], axis=0)

    U_hat, _, _ = tl.svd_interface(K, n_eigenvecs=rank)

    U = tl.concatenate([U, Q], axis=1) @ U_hat[:, :rank]
    return U
