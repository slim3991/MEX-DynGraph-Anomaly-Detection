from copy import deepcopy
from dataclasses import asdict
from functools import partial
from typing import List
import numpy as np
import numpy.typing as npt
import optuna
from sklearn.metrics import auc
import tensorly as tl
import mlflow

from models.robust_cp import robust_cp
from utils.tensor_processing import (
    make_mode_laplacian,
    preprocess,
)
from utils.anomaly_injector import *
from utils.model_eval import (
    compute_tensor_model_metrics,
    metrics_to_latex,
)
from models.lap_reg_cp import graph_regularized_als
import subprocess

type recomp_func = Callable[[npt.NDArray, int], npt.NDArray]


def get_git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def test_eval(
    T: npt.NDArray,
    L: npt.NDArray,
    rank: int,
    recomp_func: recomp_func,
):
    T_hat = recomp_func(T, rank)
    err = np.abs(T - T_hat)
    metrics = compute_tensor_model_metrics(err, L)

    return metrics


def decomp_recomp(T: tl.tensor, rank: int, threshold: float) -> npt.NDArray:
    init = "svd" if rank < 12 else "random"
    weights, factors, _ = robust_cp(
        T,
        rank=rank,
        verbose=False,
        n_iter=20,
        init=init,
        threshold=threshold,
    )
    reconst = tl.cp_to_tensor(cp_tensor=(weights, factors))
    return reconst


def parameter_search(T, L, params: dict, name: str | None = None):
    T_loc = deepcopy(T)
    del T

    def objective(trial: optuna.Trial) -> float:
        with mlflow.start_run(nested=True):
            trial_params = {
                "rank": trial.suggest_int("rank", 1, 20),
                "threshold": trial.suggest_float("threshold", 0, 2),
            }

            recomp = partial(
                decomp_recomp,
                threshold=trial_params["threshold"],
            )

            metrics = test_eval(T_loc, L, rank=trial_params["rank"], recomp_func=recomp)

            metrics_dict = {k: v for k, v in asdict(metrics).items() if v is not None}
            mlflow.log_metrics(metrics_dict)
            mlflow.log_params(trial_params)
            obj = metrics.f1

            assert obj is not None
            return obj

    # Main MLflow parent run
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params=params)
        study = optuna.create_study(direction="maximize", study_name=name)
        study.optimize(objective, n_trials=30)

        # Log best results to the parent run
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1", study.best_value)

        print(f"Best Params: {study.best_params}")
        print(f"Best F1: {study.best_value}")

        return study


def main():

    seed = 42
    np.random.seed(seed=seed)

    T = np.load("data/abiline_ten.npy")
    start = 0
    end = 5000
    T = T[:, :, start:end]

    preprocess_rank = 20
    keep_percentile = 95
    alpha = 0.4
    T = preprocess(
        T, rank=preprocess_rank, keep_percentile=keep_percentile, alpha=alpha
    )

    n_spikes = 1000
    anomaly_type = "spikes"
    if anomaly_type == "spikes":
        T, L = inject_random_spikes(T, n_spikes=n_spikes)
    elif anomaly_type == "events":
        raise NotImplementedError("not implemented")
    else:
        raise ValueError("anomaly type not supported")

    name = "Robust CP"
    params = {
        "git hash": get_git_hash(),
        "intervall": (start, end),
        "anomaly type": anomaly_type,
        "keep percentile": keep_percentile,
        "alpha": alpha,
        "seed": seed,
    }
    parameter_search(T, L, name=name, params=params)


if __name__ == "__main__":
    main()
