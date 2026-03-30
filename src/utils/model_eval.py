from dataclasses import asdict
from typing import Any, Callable, Dict, Literal, Optional, Protocol, Tuple
import mlflow
from tqdm import tqdm
import numpy.typing as npt

from utils.metrics import compute_metrics_with_threshold, print_metrics
from utils.datasets import (
    create_event_dataset_train,
    create_event_dataset_test,
    create_spike_dataset_train,
    create_spike_dataset_test,
)

dataset_fetch_func = Callable[[], Tuple[npt.NDArray, npt.NDArray, Optional[list], dict]]


def fetch_dataset_fetch_func(
    anomaly_type: Literal["spikes", "events"] = "spikes",
    train_test: Literal["train", "test"] = "test",
) -> dataset_fetch_func:
    if anomaly_type == "spikes":
        return (
            create_spike_dataset_train
            if train_test == "train"
            else create_spike_dataset_test
        )
    else:
        return (
            create_event_dataset_train
            if train_test == "train"
            else create_event_dataset_test
        )


class Transformer(Protocol):
    def fit_transform(self, X, y): ...
    @property
    def name(self) -> str: ...
    def get_params(self) -> dict: ...


def evaluate_model(
    model: Transformer,
    n_runs: int = 10,
    anomaly_type: Literal["spikes", "events"] = "spikes",
    train_test: Literal["train", "test"] = "test",
):
    metric_sum = None
    dataset_func = fetch_dataset_fetch_func(
        anomaly_type=anomaly_type, train_test=train_test
    )
    for _ in tqdm(range(n_runs)):
        T, L, events, _ = dataset_func()
        T_hat = model.fit_transform(T, None)

        resids = T - T_hat
        metrics = compute_metrics_with_threshold(
            resids, L, threshold=model.get_params()["threshold"], events=events
        )
        tqdm.write(str("PR_AUC: " + metrics.pr_auc))
        metric_sum = metrics if metric_sum is None else metric_sum + metrics
    ave_metrics = metric_sum / n_runs
    print(print_metrics(ave_metrics))
    ave_dict = asdict(ave_metrics)
    ave_dict = {k: v for k, v in ave_dict.items() if v is not None}

    mlflow.log_metrics(ave_dict)

    print_metrics(ave_metrics)
