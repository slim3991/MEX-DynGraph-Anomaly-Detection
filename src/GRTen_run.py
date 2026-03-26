from dataclasses import asdict
from functools import partial
from typing import Optional, Protocol
import numpy as np
import optuna
import tensorly as tl
import mlflow

from utils.metrics import compute_tensor_model_metrics
from utils.datasets import create_spike_dataset_std
from utils.tensor_processing import (
    make_mode_laplacian,
)
from utils.anomaly_injector import *
from utils.metrics import (
    compute_tensor_model_metrics,
)
from sk_models.gr_cp import MyGRTenDecomp
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
            "k1": trial.suggest_int("k1", 1, 144),
            "k2": trial.suggest_int("k2", 1, 133),
            "k3": trial.suggest_int("k3", 1, 500),
            "threshold": trial.suggest_float("threshold", 0, 2),
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
            laps=laps,
        )
        resids = model.residuals(T)
        metrics = compute_tensor_model_metrics(resids, L)

        metrics_dict = {k: v for k, v in asdict(metrics).items() if v is not None}
        mlflow.log_metrics(metrics_dict)
        obj = metrics.f1
        return obj if obj is not None else 0


def main():

    mlflow.set_experiment(experiment_id=1)
    mlflow.set_active_model(name="GrTen")
    name = "GRTen"
    seed = 42
    np.random.seed(seed)
    with mlflow.start_run(run_name=name):
        mlflow.log_param("name", name)
        mlflow.log_param("git_hash", get_git_hash())
        mlflow.log_param("seed", seed)

        T, L, params = create_spike_dataset_std()
        mlflow.log_params(params=params)

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, T, L, None), n_trials=30)
        mlflow.log_params(study.best_params)


if __name__ == "__main__":
    main()
