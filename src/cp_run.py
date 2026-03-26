from copy import deepcopy
from dataclasses import asdict
from typing import Optional
from tqdm import tqdm
import numpy as np
import numpy.typing as npt
import optuna
import tensorly as tl
import mlflow

from utils.tensor_processing import (
    preprocess,
)
from utils.anomaly_injector import *
from utils.model_eval import (
    compute_tensor_model_metrics,
    metrics_to_latex,
)
import subprocess

type recomp_func = Callable[[npt.NDArray, int], npt.NDArray]


def get_git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()


def test_eval(T, L, events, rank, recomp_func):
    T_hat = recomp_func(T, rank)
    err = np.abs(T - T_hat)
    if events is not None:
        metrics = compute_event_metrics(
            err,
            events=events,
            L=L,
            overlap_threshold=0.1,
        )
    else:
        metrics = compute_tensor_model_metrics(err, L)

    return metrics


def decomp_recomp(T: tl.tensor, rank: int) -> npt.NDArray:
    init = "svd" if rank < 12 else "random"
    factors = tl.decomposition.CP(
        rank=rank,
        verbose=False,
        init=init,
    ).fit_transform(T)
    reconst = tl.cp_to_tensor(factors)
    return np.array(reconst)


def parameter_search(T, L, params: dict, name: str | None = None):
    T_loc = deepcopy(T)
    del T

    def objective(trial: optuna.Trial) -> float:
        with mlflow.start_run(nested=True):
            trial_params = {
                "rank": trial.suggest_int("rank", 1, 20),
            }

            metrics = test_eval(
                T_loc,
                L,
                events=None,
                rank=trial_params["rank"],
                recomp_func=decomp_recomp,
            )

            metrics_dict = {k: v for k, v in asdict(metrics).items() if v is not None}
            mlflow.log_metrics(metrics_dict)
            mlflow.log_params(trial_params)
            obj = metrics.f1

            assert obj is not None
            return obj

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


def model_evaluation(
    T,
    data_params: dict,
    events_based: bool,
    model_params: dict,
    name: Optional[str] = None,
):
    n = 10
    metric = None
    events = None
    with mlflow.start_run(run_name=f"evaluate"):
        mlflow.log_params(data_params)
        for _ in tqdm(range(n)):
            T_iter = deepcopy(T)

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
                metric = test_eval(
                    T=T_iter,
                    L=L,
                    events=events,
                    rank=model_params["rank"],
                    recomp_func=decomp_recomp,
                )

            else:
                metric = metric + test_eval(
                    T=T_iter,
                    L=L,
                    events=events,
                    rank=model_params["rank"],
                    recomp_func=decomp_recomp,
                )

        assert metric is not None
        ave_metrics = metric / n
        print(metrics_to_latex(ave_metrics, name=name))


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

    name = "Basic CP"
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
