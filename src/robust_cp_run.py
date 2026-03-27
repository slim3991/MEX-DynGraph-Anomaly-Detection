from dataclasses import asdict
from typing import Optional, Protocol
import numpy as np
import optuna
import tensorly as tl
import mlflow

from models.RobustCp import MyRCPTenDecomp
from utils.metrics import compute_tensor_model_metrics
from utils.datasets import create_spike_dataset_std
from utils.anomaly_injector import *
import subprocess

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
    with mlflow.start_run(nested=True):
        trial_params = {
            "rank": trial.suggest_int("rank", 1, 20),
            "threshold": trial.suggest_float("threshold", 0, 2),
        }
        mlflow.log_params(trial_params)

        model = MyRCPTenDecomp(
            rank=trial_params["rank"], threshold=trial_params["threshold"]
        )
        resids = model.residuals(T)
        metrics = compute_tensor_model_metrics(resids, L)

        metrics_dict = {k: v for k, v in asdict(metrics).items() if v is not None}
        mlflow.log_metrics(metrics_dict)
        obj = metrics.f1
        return obj if obj is not None else 0


def main():

    mlflow.set_experiment(experiment_id=1)
    mlflow.set_active_model(name="R-CP")
    name = "R-CP"
    seed = 42
    np.random.seed(seed)
    n_trials = 20
    with mlflow.start_run(run_name=name):
        mlflow.log_param("name", name)
        mlflow.log_param("git_hash", get_git_hash())
        mlflow.log_param("seed", seed)

        T, L, params = create_spike_dataset_std()
        mlflow.log_params(params=params)

        study = optuna.create_study(direction="maximize", study_name="Robust-CP")
        study.optimize(lambda trial: objective(trial, T, L, None), n_trials=n_trials)
        mlflow.log_params(study.best_params)
    print("Best trial: ", study.best_trial.number, ", with value: ", study.best_value)
    print("Best Params", study.best_params, end="\n\n")


if __name__ == "__main__":
    main()
