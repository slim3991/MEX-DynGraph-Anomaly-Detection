import numpy.typing as npt
import tensorly as tl
from typing import Optional, Protocol, Sequence
from sklearn.base import BaseEstimator, TransformerMixin

from models.implementations.robust_cp import robust_cp
from utils.utils import optimal_f1_threshold


type Tensor = tl.tensor | npt.NDArray
# type objective_func = Callable[[optuna.Trial, Transformer, Tensor, Tensor], float]


class Transformer(Protocol):
    def fit_transform(self, X: Tensor, y: Optional[Tensor]) -> Tensor: ...
    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor: ...


class MyRCPTenDecomp(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rank: int = 5,
        local_threshold: Optional[float] = None,
    ):
        self.rank = rank
        self.local_threshold = local_threshold

        # Initializing learned parameters to None
        self.threshold_ = None
        self.X_hat_ = None
        self.factors_ = None

    @property
    def name(self):
        return "robust CP"

    def fit(self, X: Tensor, y: Optional[Tensor] = None):
        """
        Fits the Robust CP decomposition and learns the classification threshold.
        """
        factors, _ = robust_cp(
            X, rank=self.rank, init="random", threshold=self.local_threshold
        )

        self.factors_ = factors
        self.X_hat_ = tl.cp_to_tensor(self.factors_)

        if y is not None:
            # Storing result in self.threshold_
            self.threshold_, _ = optimal_f1_threshold(self.X_hat_, y)

        return self

    def transform(self, X: Tensor) -> Tensor:
        """
        Returns the low-rank reconstruction learned during fit.
        """
        check_is_fitted(self, ["X_hat_"])
        return self.X_hat_

    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor:
        """
        Returns the residual tensor (E = X - X_hat)
        """
        X_hat = self.transform(X)
        return X - X_hat
