import numpy.typing as npt
import tensorly as tl
from typing import List, Literal, Optional, Sequence
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

from models.implementations.lap_reg_cp import graph_regularized_als
from utils.tensor_processing import make_mode_laplacian

type Tensor = tl.tensor | npt.NDArray


class MyGRTenDecomp(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rank: int,
        lambdas: Sequence[float],
        ks: Optional[Sequence[int]],
        threshold: float,
        local_threshold: Optional[float] = None,
        laps: Optional[List] = None,
        measure: Literal[
            "angular", "euclidean", "manhattan", "hamming", "dot"
        ] = "euclidean",
    ):
        if (ks is None) == (laps is None):
            raise ValueError("One and only one of ks or laps must be set")
        self.laps = laps
        self.rank = rank
        self.lambdas = lambdas
        self.ks = ks
        self.threshold = threshold
        self.local_threshold = local_threshold
        self.measure = measure

    @property
    def name(self):
        return "GRT"

    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Learns the Laplacians from the training data.
        """
        assert self.ks is not None
        if self.laps is None:
            self.laps = (
                make_mode_laplacian(
                    X, mode=0, k=self.ks[0], normalize=True, measure=self.measure
                ),
                make_mode_laplacian(
                    X, mode=1, k=self.ks[1], normalize=True, measure=self.measure
                ),
                make_mode_laplacian(
                    X, mode=2, k=self.ks[2], normalize=True, measure=self.measure
                ),
            )

        return self

    def transform(self, X: Tensor) -> Tensor:
        """
        Applies the decomposition using learned Laplacians.
        """

        factors, self.E = graph_regularized_als(
            X, self.rank, self.laps, lmbda=self.lambdas, threshold=self.local_threshold
        )
        X_hat = tl.cp_to_tensor(factors)

        return X_hat

    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:

        # Get the reconstruction using your existing transform logic
        X_hat = self.transform(X)

        # Return the residual tensor (E = X - X_hat)
        return X - X_hat
