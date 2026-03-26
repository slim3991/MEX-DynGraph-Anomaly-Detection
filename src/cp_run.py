import numpy as np
import matplotlib.pyplot as plt
from decomp_results import preprocess

from models.decomp_model import CPTensorAnomalyDetector
from utils.anomaly_injector import *

from sklearn.pipeline import FunctionTransformer, Pipeline

from utils.model_eval import compute_tensor_model_metrics, print_metrics


def generate_datasets():
    T = np.load("data/abiline_ten.npy")
    X_train = T[:, :, :5000]
    X_test = T[:, :, 10_000:15_000]

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)
    X_train, y_train = inject_random_spikes(X_train)
    X_test, y_test = inject_random_spikes(X_test)
    y_test = np.where(y_test == 1, 1, -1)
    y_train = np.where(y_train == 1, 1, -1)
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = generate_datasets()


model = CPTensorAnomalyDetector(rank=10)
model.fit(X_train, y_train)

train_probs = model.predict_proba(X_train)
test_probs = model.predict_proba(X_test)
# plt.plot(train_probs)
# plt.plot(y_train.flatten(), "--", alpha=0.5)
# # plt.plot(test_probs)
# plt.show()
# exit()

# Convert labels to (1 = anomaly, 0 = normal)
y_test_bin = (y_test.flatten() == 1).astype(int)
y_train_bin = (y_train.flatten() == 1).astype(int)

train_metrics = compute_tensor_model_metrics(
    probs=train_probs, y_true=y_train, threshold=model.threshold_
)
test_metrics = compute_tensor_model_metrics(
    probs=test_probs, y_true=y_test, threshold=model.threshold_
)

print_metrics(train_metrics)
print_metrics(test_metrics)
