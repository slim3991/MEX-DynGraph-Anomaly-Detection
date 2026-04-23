import numpy.typing as npt
import tensorly as tl
from typing import Callable, List, Optional, Protocol, Sequence
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

from models.implementations.RHOOI import r_hooi
from utils.utils import optimal_f1_threshold


type Tensor = tl.tensor | npt.NDArray


class Transformer(Protocol):
    def fit_transform(self, X: Tensor, y: Optional[Tensor]) -> Tensor: ...
    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor: ...


class MyRHOOITenDecomp(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rank: int | Sequence[int],
        local_threshold: Optional[float] = None,
        tol=1e-6,
        verbose: bool = False,
    ):
        if type(rank) == int:
            self.ranks = (rank, rank, rank)
        else:
            self.ranks = rank

        self.verbose = verbose
        self.local_threshold = local_threshold
        self.tol = tol

        # Learned attributes
        self.threshold_ = None
        self.factors_ = None

    @staticmethod
    def name():
        return "Robust Tucker (RHOOI)"

    def fit(self, X: Tensor, y: Tensor):
        factors, _ = r_hooi(
            X,
            ranks=self.ranks,
            threshold=self.local_threshold,
            tol=self.tol,
            verbose=self.verbose,
        )
        self.factors_ = factors

        return self

    def transform(self, X: Tensor) -> Tensor:
        check_is_fitted(self, ["factors_"])
        # Returning the reconstruction
        return tl.tucker_to_tensor(self.factors_)

    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        X_hat = self.transform(X)
        return X - X_hat
