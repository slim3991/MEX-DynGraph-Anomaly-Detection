import scipy
from typing import override
from sklearn.metrics import precision_recall_curve
import tensorly as tl
import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from models.RHOOI import r_hooi

from models.robust_cp import robust_cp
from utils.tensor_processing import normalize_tensor


def detect_anomalies_soft(res):
    abs_res = np.abs(res)
    # sigma = np.median(abs_res[abs_res < np.percentile(abs_res, 50)]) / 0.6745

    sigma = np.median(np.abs(res)) / 0.6745
    # lam = 2.5 * sigma
    lam = sigma * np.sqrt(2 * np.log(res.size))

    E = np.sign(res) * np.maximum(np.abs(res) - lam, 0)
    return E


class CPTensorAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(self, rank):
        self.rank = rank
        self.threshold_ = None

    # def _residuals_to_proba(self, residuals):
    #     E = detect_anomalies_soft(residuals)
    #     E = np.where(E <= 0, 0, E)
    #     return normalize_tensor(E, "minmax").flatten()

    def _residuals_to_proba(self, residuals):
        flat = residuals.flatten()
        return normalize_tensor(flat, method="minmax")

    def _calculate_residuals(self, X):
        init = "random" if self.rank > 12 else "svd"
        cp_decomp = tl.decomposition.CP(rank=self.rank, init=init).fit_transform(X)
        reconst = tl.cp_to_tensor(cp_tensor=cp_decomp)
        return X - reconst

    def fit(self, X, y):

        if y is None:
            raise ValueError("This models requires a y argument")
        if X.shape != y.shape:
            raise ValueError("argument shapes must match")

        residuals = self._calculate_residuals(X)
        probs = self._residuals_to_proba(residuals=residuals)
        y_flat = np.array(y).flatten()

        precision_curve, recall_curve, thresholds = precision_recall_curve(
            y_flat, probs
        )

        f1_scores = (2 * precision_curve[:-1] * recall_curve[:-1]) / (
            precision_curve[:-1] + recall_curve[:-1] + 1e-10
        )

        self.threshold_ = thresholds[np.argmax(f1_scores)]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["threshold_"])
        residuals = self._calculate_residuals(X)
        return self._residuals_to_proba(residuals)

    def predict(self, X):
        """1 for detected anomaly, -1 for normal activity"""
        probs = self.predict_proba(X)
        return np.where(probs > self.threshold_, 1, -1)


class RCPAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(self, rank):
        self.rank = rank
        self.threshold_ = None

    def _calculate_residuals(self, X):
        init = "random" if self.rank > 12 else "svd"
        weights, factors, E = robust_cp(X, rank=self.rank, init=init)
        reconst = tl.cp_to_tensor((weights, factors))
        return X - reconst, E

    def _outliers_to_proba(self, E):
        E = np.where(E >= 0, E, 0)
        probs = normalize_tensor(E, method="minmax")
        return probs.flatten()

    def fit(self, X, y):

        if y is None:
            raise ValueError("This models requires a y argument")
        if X.shape != y.shape:
            raise ValueError("argument shapes must match")

        residuals, E = self._calculate_residuals(X)
        probs = self._outliers_to_proba(E)
        y_flat = np.array(y).flatten()

        precision_curve, recall_curve, thresholds = precision_recall_curve(
            y_flat, probs
        )

        f1_scores = (2 * precision_curve[:-1] * recall_curve[:-1]) / (
            precision_curve[:-1] + recall_curve[:-1] + 1e-10
        )

        self.threshold_ = thresholds[np.argmax(f1_scores)]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["threshold_"])
        residuals, E = self._calculate_residuals(X)
        return self._outliers_to_proba(E)

    def predict(self, X):
        """1 for detected anomaly, -1 for normal activity"""
        probs = self.predict_proba(X)
        return np.where(probs > 0, 1, -1)
        # return np.where(probs > self.threshold_, 1, -1)


class RHOOIAnomalyDetector(BaseEstimator, OutlierMixin):
    def __init__(self, rank):
        self.rank = rank
        self.threshold_ = None

    def _calculate_residuals(self, X):
        weights, factors, E = r_hooi(X, ranks=self.rank)
        reconst = tl.tucker_to_tensor((weights, factors))
        return X - reconst, E

    def _outliers_to_proba(self, E):
        E = np.where(E >= 0, E, 0)
        probs = normalize_tensor(E, method="minmax")
        return probs.flatten()

    def fit(self, X, y):

        if y is None:
            raise ValueError("This models requires a y argument")
        if X.shape != y.shape:
            raise ValueError("argument shapes must match")

        residuals, E = self._calculate_residuals(X)
        probs = self._outliers_to_proba(E)
        y_flat = np.array(y).flatten()

        precision_curve, recall_curve, thresholds = precision_recall_curve(
            y_flat, probs
        )

        f1_scores = (2 * precision_curve[:-1] * recall_curve[:-1]) / (
            precision_curve[:-1] + recall_curve[:-1] + 1e-10
        )

        self.threshold_ = thresholds[np.argmax(f1_scores)]
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ["threshold_"])
        residuals, E = self._calculate_residuals(X)
        return self._outliers_to_proba(E)

    def predict(self, X):
        """1 for detected anomaly, -1 for normal activity"""
        probs = self.predict_proba(X)
        return np.where(probs > self.threshold_, 1, -1)
