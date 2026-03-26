from dataclasses import asdict
from typing import Optional, Protocol
import mlflow
from tqdm import tqdm
from copy import deepcopy

from utils.anomaly_injector import inject_random_shapes, inject_random_spikes_normal
from metrics import compute_tensor_model_metrics, print_metrics


class Transformer(Protocol):
    def fit_transform(self, X, y): ...


def _create_spike_dataset(T):
    n_spikes = 1000
    amplitude_factor = 10

    T, L = inject_random_spikes_normal(
        T, amplitide_factor=amplitude_factor, n_spikes=n_spikes
    )
    mlflow.log_params({"amplitude_factor": amplitude_factor, "n_spikes": n_spikes})

    return T, L


def _create_event_dataset(T):
    start_min = 20
    params = {
        "start_min": 20,
        "start_max": 4000,
        "min_durantion": 10,
        "max_duration": 100,
        "n_shapes": 20,
        "amplitude_factor": 10,
    }
    T, L, events = inject_random_shapes(T, **params)
    mlflow.log_params(params)
    return T, L, events


def evaluate_model(
    T,
    model: Transformer,
    n_runs: int,
    log_params: Optional[dict],
    test_events: bool = False,
):

    metric_sum = None
    with mlflow.start_run(run_name=f"spike_anomalies_{model.name}"):
        if log_params:
            mlflow.log_params(log_params)
        for _ in tqdm(range(n_runs)):
            T_loc, L = _create_spike_dataset(deepcopy(T))
            T_hat = model.fit_transform(T, None)

            resids = T_loc - T_hat
            metrics = compute_tensor_model_metrics(resids, L)
            metric_sum = metrics if metric_sum is None else metric_sum + metrics
        ave_metrics = metric_sum / n_runs
        mlflow.log_metrics(asdict(ave_metrics))
        print_metrics(ave_metrics)

    if test_events == False:
        return

    metric_sum = None
    with mlflow.start_run(run_name=f"ev_anomalies_{model.name}"):
        if log_params:
            mlflow.log_params(log_params)
        for _ in tqdm(range(n_runs)):
            T_loc, L, _ = _create_event_dataset(deepcopy(T))
            T_hat = model.fit_transform(T)

            resids = T_loc - T_hat

            # TODO: fix better event anomaly detection
            metrics = compute_tensor_model_metrics(resids, L)
            metric_sum = metrics if metric_sum is None else metric_sum + metrics
        ave_metrics = metric_sum / n_runs
        mlflow.log_metrics(asdict(ave_metrics))
        print_metrics(ave_metrics)
