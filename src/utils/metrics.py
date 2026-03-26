"""
evaluation.py

Utilities for evaluating tensor anomaly detection models.

Provides:
- metric computation (pure logic)
- console display
- LaTeX table generation
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from utils.anomaly_injector import AnomalyEvent
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    f1_score,
    auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
)


# ============================================================
# Dataclasses
# ============================================================


@dataclass
class Metrics:
    n_anomalies: int
    roc_auc: float
    pr_auc: Optional[float]
    precision: float
    recall: float
    f1: Optional[float]
    fpr: float
    tpr: float
    threshold: Optional[float]
    TP: int
    FP: int
    TN: int
    FN: int

    def _safe_add(self, a, b):
        if a is None or b is None:
            return None
        return a + b

    def _safe_div(self, a, n):
        if a is None:
            return None
        return a / n

    def __add__(self, other: "Metrics") -> "Metrics":
        return Metrics(
            n_anomalies=self.n_anomalies + other.n_anomalies,
            roc_auc=self.roc_auc + other.roc_auc,
            pr_auc=self._safe_add(self.pr_auc, other.pr_auc),
            precision=self.precision + other.precision,
            recall=self.recall + other.recall,
            f1=self._safe_add(self.f1, other.f1),
            fpr=self.fpr + other.fpr,
            tpr=self.tpr + other.tpr,
            threshold=self._safe_add(self.threshold, other.threshold),
            TP=self.TP + other.TP,
            FP=self.FP + other.FP,
            TN=self.TN + other.TN,
            FN=self.FN + other.FN,
        )

    def __truediv__(self, n: float) -> "Metrics":
        return Metrics(
            n_anomalies=self.n_anomalies / n,
            roc_auc=self.roc_auc / n,
            pr_auc=self._safe_div(self.pr_auc, n),
            precision=self.precision / n,
            recall=self.recall / n,
            f1=self._safe_div(self.f1, n),
            fpr=self.fpr / n,
            tpr=self.tpr / n,
            threshold=self._safe_div(self.threshold, n),
            TP=self.TP / n,
            FP=self.FP / n,
            TN=self.TN / n,
            FN=self.FN / n,
        )


# ============================================================
# Compute Functions
# ============================================================
def compute_tensor_model_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: Optional[float] = None,
) -> Metrics:
    """
    Compute evaluation metrics for tensor anomaly detection.

    Parameters
    ----------
    probs : np.ndarray
        Flattened anomaly probabilities (higher = more anomalous)
    y_true : np.ndarray
        Ground truth labels (1 = anomaly, 0 = normal)
    threshold : Optional[float]
        Decision threshold. If None, optimal F1 threshold is computed.

    Returns
    -------
    Metrics
    """

    # Flatten inputs
    probs = np.array(probs).flatten()
    y_true = np.array(y_true).flatten()

    # Convert labels to {0,1}
    y_true = (y_true > 0).astype(int)

    roc_auc = roc_auc_score(y_true, probs)

    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, probs)
    pr_auc = auc(recall_curve, precision_curve)

    if threshold is None:
        f1_scores = (2 * precision_curve * recall_curve) / (
            precision_curve + recall_curve + 1e-10
        )
        best_idx = np.argmax(f1_scores)
        threshold = pr_thresholds[max(best_idx - 1, 0)]  # align sizes
        f1_val = f1_scores[best_idx]
    else:
        y_pred_tmp = (probs > threshold).astype(int)
        f1_val = f1_score(y_true, y_pred_tmp)

    y_pred = (probs > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision_val = precision_score(y_true, y_pred, zero_division=0)
    recall_val = recall_score(y_true, y_pred, zero_division=0)

    # FPR / TPR
    fpr_val = fp / (fp + tn + 1e-10)
    tpr_val = tp / (tp + fn + 1e-10)

    return Metrics(
        n_anomalies=int(np.sum(y_true)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        precision=precision_val,
        recall=recall_val,
        f1=f1_val,
        fpr=fpr_val,
        tpr=tpr_val,
        threshold=threshold,
        TP=int(tp),
        FP=int(fp),
        TN=int(tn),
        FN=int(fn),
    )


# ============================================================
# Display Functions
# ============================================================


def print_metrics(metrics: Metrics) -> None:
    """
    Pretty console output of metrics.
    """

    print("-" * 40)
    if metrics.threshold is not None:
        print(f"Optimal Threshold: {metrics.threshold:.4f}")
    print(f"No. anomalies: {metrics.n_anomalies}")
    if metrics.roc_auc is not None:
        print(f"ROC-AUC: {metrics.roc_auc:.4f}")

    if metrics.pr_auc is not None:
        print(f"PR-AUC: {metrics.pr_auc:.4f}")

    print(f"Precision: {metrics.precision:.2%}")
    print(f"Recall (TPR): {metrics.recall:.2%}")

    if metrics.f1 is not None:
        print(f"F1-score: {metrics.f1:.4f}")

    print(f"False Positive Rate: {metrics.fpr:.2%}")
    print(f"True Positive Rate: {metrics.tpr:.2%}")

    print("\nConfusion Matrix:")
    print(f"TP: {metrics.TP}, FP: {metrics.FP}")
    print(f"FN: {metrics.FN}, TN: {metrics.TN}")


# ============================================================
# LaTeX Export
# ============================================================


def metrics_to_latex(
    metrics: Metrics,
    name: str = "Value",
) -> str:
    """
    Convert metrics to a LaTeX table row.
    """

    headers = [
        "PR-AUC",
        "Precision",
        "Recall",
        "F1",
        "FPR",
        "TPR",
    ]

    values = [
        metrics.pr_auc,
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.fpr,
        metrics.tpr,
    ]

    formatted_values = ["-" if v is None else f"{v:.4f}" for v in values]

    table = "\\begin{tabular}{lccccccc}\n"
    table += "\\toprule\n"
    table += "Metric & " + " & ".join(headers) + " \\\\\n"
    table += "\\midrule\n"
    table += f"{name} & " + " & ".join(formatted_values) + " \\\\\n"
    table += "\\bottomrule\n"
    table += "\\end{tabular}"

    return table


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":
    # Example random data
    err = np.random.rand(20, 20)
    L = np.random.randint(0, 2, (20, 20))

    metrics = compute_tensor_model_metrics(err, L)

    print_metrics(metrics)

    print("\nLaTeX table:\n")
    print(metrics_to_latex(metrics))
