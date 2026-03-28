from dataclasses import asdict
from functools import partial
from typing import Optional, Protocol
import numpy as np
import optuna
import tensorly as tl
import mlflow

from utils.datasets import create_event_dataset_train, create_spike_dataset_train
from utils.metrics import compute_metrics_with_threshold
from utils.tensor_processing import (
    make_mode_laplacian,
)
from utils.anomaly_injector import *
from models import MyGRTenDecomp
import subprocess

type Tensor = tl.tensor | npt.NDArray
type objective_func = Callable[[optuna.Trial, Transformer, Tensor, Tensor], float]


class Transformer(Protocol):
    def fit_transform(self, X: Tensor, y: Optional[Tensor]) -> Tensor: ...
    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor: ...


def get_git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def objective(trial: optuna.Trial, T, L, events=None):
    with mlflow.start_run(nested=True):
        trial_params = {
            "rank": trial.suggest_int("rank", 1, 20),
            "lambda_0": trial.suggest_float("lambda_0", 1e-2, 1e2, log=True),
            "lambda_1": trial.suggest_float("lambda_1", 1e-2, 1e2, log=True),
            "lambda_2": trial.suggest_float("lambda_2", 1e-2, 1e2, log=True),
            "distance": trial.suggest_categorical(
                "distance", ["dot", "euclidean", "angular"]
            ),
            "k1": trial.suggest_int("k1", 1, min(T.shape[0], 500)),
            "k2": trial.suggest_int("k2", 1, min(T.shape[0], 500)),
            "k3": trial.suggest_int("k3", 1, min(T.shape[0], 500)),
            "threshold": trial.suggest_float("threshold", 0, 1),
            "local_threshold": trial.suggest_float("local_threshold", 0, 2),
            "anomaly_type": "spikes" if events is None else "events",
        }
        mlflow.log_params(trial_params)

        laps = [
            make_mode_laplacian(
                T,
                mode=0,
                k=trial_params["k1"],
                normalize=True,
                measure=trial_params["distance"],
            ),
            make_mode_laplacian(
                T,
                mode=1,
                k=trial_params["k2"],
                normalize=True,
                measure=trial_params["distance"],
            ),
            make_mode_laplacian(
                T,
                mode=2,
                k=trial_params["k3"],
                normalize=True,
                measure=trial_params["distance"],
            ),
        ]

        lambdas = [
            trial_params["lambda_0"],
            trial_params["lambda_1"],
            trial_params["lambda_2"],
        ]
        model = MyGRTenDecomp(
            rank=trial_params["rank"],
            lambdas=lambdas,
            ks=None,
            threshold=trial_params["threshold"],
            local_threshold=trial_params["local_threshold"],
            laps=laps,
        )
        resids = model.residuals(T)
        metrics = compute_metrics_with_threshold(
            resids, L, model.get_params()["threshold"], events=events
        )

        metrics_dict = {k: v for k, v in asdict(metrics).items() if v is not None}
        mlflow.log_metrics(metrics_dict)
        obj = metrics.pr_auc
        return obj if obj is not None else 0


def main():

    mlflow.set_experiment(experiment_name="GRTen")
    mlflow.set_active_model(name="GrTen")
    name = "GRTen"
    seed = 42
    np.random.seed(seed)
    anomaly_type = "spike"
    if anomaly_type == "spike":
        data_fetch_func = create_spike_dataset_train
    else:
        data_fetch_func = create_event_dataset_train
    with mlflow.start_run(run_name=name):
        mlflow.log_param("name", name)
        mlflow.log_param("git_hash", get_git_hash())
        mlflow.log_param("seed", seed)
        mlflow.log_param("anomaly_type", anomaly_type)

        T, L, events, params = data_fetch_func()
        mlflow.log_params(params=params)

        study = optuna.create_study(direction="maximize", study_name="GRT")
        study.optimize(lambda trial: objective(trial, T, L, events), n_trials=80)
        mlflow.log_params(study.best_params)
    print("Best trial: ", study.best_trial.number, ", with value: ", study.best_value)
    print("Best Params", study.best_params, end="\n\n")


if __name__ == "__main__":
    main()
