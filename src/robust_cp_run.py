import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from models.decomp_model import CPTensorAnomalyDetector, RCPAnomalyDetector
from utils.anomaly_injector import *

from sklearn.pipeline import FunctionTransformer, Pipeline

from utils.model_eval import (
    compute_binary_model_metrics,
    compute_tensor_model_metrics,
    print_metrics,
)
from utils.tensor_processing import preprocess


def generate_datasets():
    T = np.load("data/abiline_ten.npy")
    X_train = deepcopy(T[:, :, :5000])
    X_test = deepcopy(T[:, :, 10_000:15_000])
    del T

    X_train = preprocess(X_train)
    X_test = preprocess(X_test)
    X_train, y_train = inject_random_spikes(X_train, 500)
    X_test, y_test = inject_random_spikes(X_test, 500)
    y_test = np.where(y_test == 1, 1, -1)
    y_train = np.where(y_train == 1, 1, -1)
    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = generate_datasets()


model = RCPAnomalyDetector(rank=10)
model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
# plt.plot(train_pred, label="train_proba")
# plt.plot(y_train.flatten(), alpha=0.5, label="y_train")
# # plt.plot(test_probs)
# plt.legend()
# plt.show()
# exit()

# Convert labels to (1 = anomaly, 0 = normal)
y_test_bin = (y_test.flatten() == 1).astype(int)
y_train_bin = (y_train.flatten() == 1).astype(int)

train_metrics = compute_binary_model_metrics(y_pred=train_pred, y_true=y_train)
test_metrics = compute_binary_model_metrics(y_pred=test_pred, y_true=y_test)

print_metrics(train_metrics)
print_metrics(test_metrics)
