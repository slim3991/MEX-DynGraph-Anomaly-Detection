import numpy.typing as npt
import tensorly as tl
from typing import Optional, Protocol, Sequence
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

from utils.utils import optimal_f1_threshold


type Tensor = tl.tensor | npt.NDArray


class Transformer(Protocol):
    def fit_transform(self, X: Tensor, y: Optional[Tensor]) -> Tensor: ...
    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor: ...


class MyTuckerTenDecomp(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rank: Sequence[int] = (5, 5, 5),
        threshold: Optional[float] = None,
        tol=1e-6,
    ):
        if type(rank) == int:
            self.ranks = (rank, rank, rank)
        else:
            self.ranks = rank

        self.tol = tol
        self.threshold_ = None
        self.threshold = threshold
        self.tucker_parts_ = None

    @staticmethod
    def name():
        return "Basic Tucker"

    def fit(self, X: Tensor, y: Tensor):
        self.tucker_parts_ = tl.decomposition.tucker(
            X, rank=self.ranks, init="random", tol=self.tol
        )

        X_hat = tl.tucker_to_tensor(self.tucker_parts_)

        if y is not None:
            self.threshold_, _ = optimal_f1_threshold(X_hat, y)
        return self

    def transform(self, X: Tensor) -> Tensor:
        """
        Returns the reconstructed tensor based on the learned Tucker components.
        """
        check_is_fitted(self, ["tucker_parts_"])

        return tl.tucker_to_tensor(self.tucker_parts_)

    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Returns the error tensor (E = X - X_hat)
        """
        X_hat = self.transform(X)
        return X - X_hat
