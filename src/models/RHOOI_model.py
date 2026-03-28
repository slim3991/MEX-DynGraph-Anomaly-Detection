import numpy.typing as npt
import tensorly as tl
from typing import Callable, List, Optional, Protocol, Sequence
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

from models.implementations.RHOOI import r_hooi


type Tensor = tl.tensor | npt.NDArray
type objective_func = Callable[[optuna.Trial, Transformer, Tensor, Tensor], float]


class Transformer(Protocol):
    def fit_transform(self, X: Tensor, y: Optional[Tensor]) -> Tensor: ...
    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor: ...


class MyRHOOITenDecomp(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        ranks: Sequence[int],
        local_threshold: float,
        threshold: Optional[float] = None,
    ):
        self.ranks = ranks
        self.threshold = threshold
        self.local_threshold = local_threshold

    @property
    def name(self):
        return "RHOOI"

    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Learns the Laplacians from the training data.
        """
        return self

    def transform(self, X: Tensor) -> Tensor:
        """
        Applies the decomposition using learned Laplacians.
        """
        factors, _ = r_hooi(X, ranks=self.ranks, threshold=self.local_threshold)
        X_hat = tl.tucker_to_tensor(factors)

        return X_hat

    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:

        # Get the reconstruction using your existing transform logic
        X_hat = self.fit_transform(X)

        # Return the residual tensor (E = X - X_hat)
        return X - X_hat
