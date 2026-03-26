import numpy.typing as npt
import tensorly as tl
from typing import List, Optional, Sequence
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

from models.lap_reg_cp import graph_regularized_als
from utils.tensor_processing import make_mode_laplacian

type Tensor = tl.tensor | npt.NDArray


class MyGRTenDecomp(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rank: int = 5,
        lambdas: Sequence[float] = (0.1, 0.1, 0.1),
        ks: Optional[Sequence[int]] = (5, 5, 5),
        threshold: Optional[float] = None,
        laps: Optional[List] = None,
    ):
        if (ks is None) == (laps is None):
            raise ValueError("One and only one of ks or laps must be set")
        self.laps_ = laps
        self.rank = rank
        self.lambdas = lambdas
        self.ks = ks
        self.threshold = threshold

    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Learns the Laplacians from the training data.
        """
        if self.laps_ is None:
            self.laps_ = (
                make_mode_laplacian(X, mode=0, k=self.ks[0], normalize=True),
                make_mode_laplacian(X, mode=1, k=self.ks[1], normalize=True),
                make_mode_laplacian(X, mode=2, k=self.ks[2], normalize=True),
            )

        return self

    def transform(self, X: Tensor) -> Tensor:
        """
        Applies the decomposition using learned Laplacians.
        """
        check_is_fitted(self, ["laps_"])

        factors, self.E = graph_regularized_als(
            X, self.rank, self.laps_, lmbda=self.lambdas, threshold=self.threshold
        )
        X_hat = tl.cp_to_tensor(factors)

        return X_hat

    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        check_is_fitted(self, ["laps_"])

        # Get the reconstruction using your existing transform logic
        X_hat = self.transform(X)

        # Return the residual tensor (E = X - X_hat)
        return X - X_hat
