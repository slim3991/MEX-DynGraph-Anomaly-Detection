import numpy.typing as npt
import tensorly as tl
from typing import List, Literal, Optional, Sequence
from sklearn.base import BaseEstimator, TransformerMixin, check_is_fitted

from models.implementations.lap_reg_cp import graph_regularized_als
from utils.tensor_processing import make_mode_laplacian
from utils.utils import optimal_f1_threshold

type Tensor = tl.tensor | npt.NDArray


class MyGRTenDecomp(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        rank: int = 5,
        lambdas: Sequence[float] = (0.1, 0.1, 0.1),
        ks: Optional[Sequence[int]] = (5, 5, 5),
        local_threshold: Optional[float] = None,
        threshold: Optional[float] = None,
        laps: Optional[List] = None,
        measure: Literal[
            "angular", "euclidean", "manhattan", "hamming", "dot"
        ] = "euclidean",
    ):
        self.rank = rank
        self.lambdas = lambdas
        self.ks = ks
        self.local_threshold = local_threshold
        self.laps = laps
        self.measure = measure
        self.threshold = threshold

        # Learned attributes
        self.threshold_ = None
        self.X_hat_ = None
        self.laps_ = None
        self.E_ = None

    @property
    def name(self):
        return "GRT"

    def _is_no_mode(self, mode) -> bool:
        return False

    def fit(self, X: Tensor, y: Optional[Tensor] = None):
        if (self.ks is None) and (self.laps is None):
            raise ValueError("One of 'ks' or 'laps' must be provided.")

        if self.laps is not None:
            self.laps_ = self.laps
        else:
            # Dynamically generate Laplacians for each mode
            self.laps_ = [
                (
                    make_mode_laplacian_annoy(
                        X, mode=i, k=self.ks[i], normalize=True, measure=self.measure
                    )
                )
                for i in range(X.ndim)
            ]

        factors, self.E_ = graph_regularized_als(
            X, self.rank, self.laps_, lmbda=self.lambdas, threshold=self.local_threshold
        )

        self.X_hat_ = tl.cp_to_tensor(factors)

        # 4. Global thresholding
        if y is not None:
            self.threshold_, _ = optimal_f1_threshold(self.X_hat_, y)

        return self

    def transform(self, X: Tensor) -> Tensor:
        check_is_fitted(self, ["X_hat_"])
        return self.X_hat_

    def residuals(self, X: Tensor) -> Tensor:
        X_hat = self.transform(X)
        return X - X_hat
