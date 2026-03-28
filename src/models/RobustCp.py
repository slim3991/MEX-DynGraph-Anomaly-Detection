import numpy.typing as npt
import tensorly as tl
from typing import Optional, Protocol, Sequence
from sklearn.base import BaseEstimator, TransformerMixin

from models.implementations.robust_cp import robust_cp


type Tensor = tl.tensor | npt.NDArray
# type objective_func = Callable[[optuna.Trial, Transformer, Tensor, Tensor], float]


class Transformer(Protocol):
    def fit_transform(self, X: Tensor, y: Optional[Tensor]) -> Tensor: ...
    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor: ...


class MyRCPTenDecomp(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rank: int,
        threshold: float,
        local_threshold: Optional[float] = None,
    ):
        self.rank = rank
        self.threshold = threshold
        self.local_threshold = local_threshold

    @property
    def name(self):
        return "robust CP"

    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Learns the Laplacians from the training data.
        """
        return self

    def transform(self, X: Tensor) -> Tensor:
        """
        Applies the decomposition using learned Laplacians.
        """
        factors, _ = robust_cp(
            X, rank=self.rank, init="random", threshold=self.local_threshold
        )
        X_hat = tl.cp_to_tensor(factors)

        return X_hat

    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:

        # Get the reconstruction using your existing transform logic
        X_hat = self.fit_transform(X)

        # Return the residual tensor (E = X - X_hat)
        return X - X_hat
