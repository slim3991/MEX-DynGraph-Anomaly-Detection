from copy import deepcopy
from dataclasses import asdict
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.metrics import auc, precision_recall_curve
import tensorly as tl
import mlflow

from hyperparam_tune import compute_pr_auc
from utils.tensor_processing import (
    de_anomalize_tensor,
    make_mode_laplacian,
    normalize_tensor,
    preprocess,
)
from utils.anomaly_injector import *
from utils.model_eval import (
    compute_tensor_model_metrics,
    metrics_to_latex,
    print_metrics,
)
from models.lap_reg_cp import graph_regularized_als
import subprocess


class Dataset:
    def __init__(
        self,
        start: int,
        end: int,
        preprocess_rank: int,
        keep_percentile: int,
        alpha: float,
        path: str = "data/abiline_ten.npy",
        name: str = "abilene",
    ):
        self.data = np.load(path)
        self.start = start
        self.end = end
        self.data = self.data[:, :, start:end]
        self.n_spikes = None
        self.events = None

        self.preprocess_rank = preprocess_rank
        self.keep_percentile = keep_percentile
        self.alpha = alpha
        self.data = preprocess(
            self.data,
            rank=self.preprocess_rank,
            keep_percentile=self.keep_percentile,
            alpha=self.alpha,
        )

    def inject_anomaly(self):
        raise NotImplementedError("not implemented")


def get_git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def evaluate(T, rank, recomp_func, events_based=True, test=None):
    n = 10
    metric = None
    events = None
    T_hat = preprocess(T)

    for _ in range(n):
        T_iter = deepcopy(T_hat)

        if events_based:
            T_iter, L, events = inject_random_shapes(
                T_iter,
                start_min=20,
                start_max=4000,
                min_durantion=10,
                max_duration=100,
                n_shapes=20,
                amplitude_factor=10,
            )
        else:
            T_iter, L = inject_random_spikes_normal(
                T_iter, amplitide_factor=10, n_spikes=1000
            )

        if metric is None:
            metric = test_eval(T_iter, L, events, rank, recomp_func=recomp_func)
        else:
            metric = metric + test_eval(
                T_iter, L, events, rank, recomp_func=recomp_func
            )

    ave_metrics = metric / n
    print(metrics_to_latex(ave_metrics, name=test))


def test_eval(T, L, events, rank, recomp_func):
    T_hat = recomp_func(T, rank)
    err = np.abs(T - T_hat)
    metrics = compute_tensor_model_metrics(err, L)

    return metrics


def decomp_recomp(T: tl.tensor, rank: int, laps, lambdas, threshold):
    cp_decomp, _ = graph_regularized_als(
        T,
        rank=rank,
        lmbda=lambdas,
        laps=laps,
        verbose=False,
        n_E=1000,
        n_iter=20,
        threshold=threshold,
    )
    reconst = tl.cp_to_tensor(cp_tensor=cp_decomp)
    return reconst


def parameter_search(T, L, events, params: dict, name: str | None = None):
    # Local copy to prevent side effects on the original tensor
    T_loc = deepcopy(T)
    del T

    def objective(trial: optuna.Trial):
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

            laps = [
                make_mode_laplacian(
                    T_loc,
                    mode=0,
                    k=trial_params["k1"],
                    normalize=True,
                    measure=trial_params["distance"],
                ),
                make_mode_laplacian(
                    T_loc,
                    mode=1,
                    k=trial_params["k2"],
                    normalize=True,
                    measure=trial_params["distance"],
                ),
                make_mode_laplacian(
                    T_loc,
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
            recomp = partial(
                decomp_recomp,
                laps=laps,
                lambdas=lambdas,
                threshold=trial_params["threshold"],
            )

            metrics = test_eval(
                T_loc, L, events=events, rank=trial_params["rank"], recomp_func=recomp
            )
            metrics_dict = {k: v for k, v in asdict(metrics).items() if v is not None}
            mlflow.log_metrics(metrics_dict)
            mlflow.log_params(trial_params)

            return metrics.f1

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

    name = "test"
    params = {
        "git hash": get_git_hash(),
        "name": name,
        "intervall": (start, end),
        "anomaly type": anomaly_type,
        "keep percentile": keep_percentile,
        "alpha": alpha,
        "seed": seed,
    }
    parameter_search(T, L, name=name, events=None, params=params)


if __name__ == "__main__":
    main()
