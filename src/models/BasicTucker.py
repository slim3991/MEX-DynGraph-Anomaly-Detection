import numpy.typing as npt
import tensorly as tl
from typing import Optional, Protocol, Sequence
from sklearn.base import BaseEstimator, TransformerMixin


type Tensor = tl.tensor | npt.NDArray


class Transformer(Protocol):
    def fit_transform(self, X: Tensor, y: Optional[Tensor]) -> Tensor: ...
    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor: ...


class MyTuckerTenDecomp(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        threshold: float,
        ranks: Sequence[int] = (5, 5, 5),
    ):
        self.ranks = ranks
        self.threshold = threshold

    @property
    def name(self):
        return "basic_tucker"

    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Learns the Laplacians from the training data.
        """
        return self

    def transform(self, X: Tensor) -> Tensor:
        """
        Applies the decomposition using learned Laplacians.
        """
        factors = tl.decomposition.tucker(X, rank=self.ranks, init="random")
        X_hat = tl.tucker_to_tensor(factors)

        return X_hat

    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:

        # Get the reconstruction using your existing transform logic
        X_hat = self.fit_transform(X)

        # Return the residual tensor (E = X - X_hat)
        return X - X_hat
