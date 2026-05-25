import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, precision_recall_curve, average_precision_score


def plot_demo_pr_curve():
    # 1. Generate synthetic binary classification data
    X, y = make_classification(
        n_samples=1000, n_classes=2, weights=[0.7, 0.3], random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 2. Fit a quick model and get probability scores
    model = LogisticRegression()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    # 3. Calculate Precision and Recall at various thresholds
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)
    print(pr_auc)

    # 4. Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall,
        precision,
        color="darkblue",
        lw=2,
        label=f"PR curve (PR-AUC = {pr_auc:.2f})",
    )
    plt.fill_between(recall, precision, alpha=0.2, color="#3498db", label="PR-AUC")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


# Run the demo
plot_demo_pr_curve()
