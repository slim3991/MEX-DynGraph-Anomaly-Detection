import numpy as np
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)


def eval_tensor_model(err: np.typing.NDArray, L: np.typing.NDArray):
    print("-" * 20)
    print("-" * 20)
    # Flatten everything
    y_true = L.flatten().astype(int)
    y_score = err.flatten()

    # Compute ROC
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    n_anomalies = np.sum(L)
    print(f"No. anomalies: {n_anomalies}")
    print(f"AUC: {roc_auc:.4f}")

    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (Average Precision): {pr_auc:.4f}")

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    print(f"Optimal Threshold (Max F1): {best_threshold:.4f}")
    print(f"Best F1-Score: {best_f1:.4f}")

    err_binary = y_score > best_threshold

    TP = np.sum((y_true == 1) & (err_binary == 1))
    TN = np.sum((y_true == 0) & (err_binary == 0))
    FP = np.sum((y_true == 0) & (err_binary == 1))
    FN = np.sum((y_true == 1) & (err_binary == 0))
    fpr_opt = FP / (FP + TN + 1e-10)

    print(f"Optimized Precision: {TP / (TP + FP):.2%}")
    print(f"Optimized Recall (TPR): {TP / (TP + FN):.2%}")
    print(f"False Positive Rate (FPR): {fpr_opt:.2%}")
