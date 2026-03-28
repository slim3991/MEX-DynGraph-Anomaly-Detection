from dataclasses import asdict
from functools import partial
from threading import Thread
from typing import Optional, Protocol
import numpy as np
import optuna
import tensorly as tl
import mlflow

from models.RHOOI_model import MyRHOOITenDecomp
from utils.anomaly_injector import *
import subprocess

from utils.datasets import create_event_dataset_train, create_spike_dataset_train
from utils.metrics import compute_metrics_with_threshold

#################### Type def ########################
type Tensor = tl.tensor | npt.NDArray
type objective_func = Callable[[optuna.Trial, Transformer, Tensor, Tensor], float]


class Transformer(Protocol):
    def fit_transform(self, X: Tensor, y: Optional[Tensor]) -> Tensor: ...
    def residuals(self, X: Tensor, y: Optional[Tensor] = None) -> Tensor: ...


######################################################
def get_git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def objective(trial: optuna.Trial, T, L, events=None):
    n1, n2, n3 = T.shape
    with mlflow.start_run(nested=True):
        trial_params = {
            "rank_0": trial.suggest_int("rank_0", 1, min(20, n1)),
            "rank_1": trial.suggest_int("rank_1", 1, min(20, n2)),
            "rank_2": trial.suggest_int("rank_2", 1, min(20, n3)),
            "local_threshold": trial.suggest_float("local_threshold", 0, 3),
            "threshold": trial.suggest_float("threshold", 0, 1),
            "anomaly_type": "spikes" if events is None else "events",
        }
        mlflow.log_params(trial_params)
        ranks = (
            trial_params["rank_0"],
            trial_params["rank_1"],
            trial_params["rank_2"],
        )

        model = MyRHOOITenDecomp(
            ranks=ranks,
            local_threshold=trial_params["local_threshold"],
            threshold=trial_params["threshold"],
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
    mlflow.set_experiment(experiment_name="RHOOI")
    mlflow.set_active_model(name="RHOOI")
    name = "RHOOI"
    seed = 42
    anomaly_type = "spikes"
    if anomaly_type == "spikes":
        data_fetch_func = create_spike_dataset_train
    else:
        data_fetch_func = create_event_dataset_train
    np.random.seed(seed)
    n_trials = 40
    with mlflow.start_run(run_name=name):
        mlflow.log_param("name", name)
        mlflow.log_param("git_hash", get_git_hash())
        mlflow.log_param("seed", seed)

        T, L, events, params = data_fetch_func()
        mlflow.log_params(params=params)

        study = optuna.create_study(direction="maximize", study_name="RHOOI")
        study.optimize(lambda trial: objective(trial, T, L, events), n_trials=n_trials)
        mlflow.log_params(study.best_params)
    print("Best trial: ", study.best_trial.number, ", with value: ", study.best_value)
    print("Best Params", study.best_params, end="\n\n")


if __name__ == "__main__":
    main()
