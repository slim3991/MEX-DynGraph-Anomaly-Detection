import numpy.typing as np
import numpy.typing as npt
import tensorly as tl
from typing import Optional, Protocol
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from utils.utils import optimal_f1_threshold

# Assuming optimal_f1_threshold is defined in your utils
# from utils.utils import optimal_f1_threshold

type Tensor = npt.NDArray  # Simplified for clarity


class MyCPTenDecomp(BaseEstimator, TransformerMixin):
    def __init__(self, rank: int = 5, threshold: Optional[float] = None):
        self.threshold = threshold
        self.rank = rank
        self.threshold_ = None
        self.factors = None

    @property
    def name(self):
        return "basic_CP"

    def fit(self, X: Tensor, y: Optional[Tensor] = None):
        self.factors_ = tl.decomposition.parafac(X, rank=self.rank, init="random")

        X_hat = tl.cp_to_tensor(self.factors_)

        if y is not None:
            self.threshold_, _ = optimal_f1_threshold(X_hat, y)
        return self

    def transform(self, X: Tensor) -> Tensor:
        """
        In CP decomposition, 'transforming' usually means projecting
        new data or returning the reconstruction.
        """
        check_is_fitted(self, ["factors_"])

        # Returning the reconstructed tensor
        return tl.cp_to_tensor(self.factors_)

    def residuals(self, X: Tensor) -> Tensor:
        """
        Returns the error tensor (E = X - X_hat)
        """
        X_hat = self.transform(X)
        return X - X_hat
