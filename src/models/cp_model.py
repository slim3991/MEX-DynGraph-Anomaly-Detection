import numpy.typing as npt
import tensorly as tl
from typing import List, Optional, Protocol, Sequence
from sklearn.base import BaseEstimator, TransformerMixin


type Tensor = tl.tensor | npt.NDArray


class Transformer(Protocol):
    def fit_transform(self, X: Tensor, y: Optional[Tensor]) -> Tensor: ...
    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor: ...


class MyCPTenDecomp(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rank: int = 5,
    ):
        self.rank = rank

    def fit(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Learns the Laplacians from the training data.
        """
        return self

    def transform(self, X: Tensor) -> Tensor:
        """
        Applies the decomposition using learned Laplacians.
        """
        factors = tl.decomposition.CP(rank=self.rank, init="random").fit_transform(X)
        X_hat = tl.cp_to_tensor(factors)

        return X_hat

    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:

        # Get the reconstruction using your existing transform logic
        X_hat = self.fit_transform(X)

        # Return the residual tensor (E = X - X_hat)
        return X - X_hat
