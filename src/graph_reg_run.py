from copy import deepcopy
from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.metrics import auc, precision_recall_curve
import tensorly as tl

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


def decomp_recomp(T: tl.tensor, rank: int, laps, lambdas):
    cp_decomp, _ = graph_regularized_als(
        T,
        rank=rank,
        lmbda=lambdas,
        laps=laps,
        verbose=False,
        n_E=1000,
        n_iter=20,
    )
    reconst = tl.cp_to_tensor(cp_tensor=cp_decomp)
    return reconst


def parameter_search(
    T,
    events_anomalies: bool,
    name: str | None = None,
):
    T_loc = deepcopy(preprocess(T))
    del T
    if events_anomalies:
        T_loc, L, events = inject_random_shapes(
            T_loc,
            start_min=20,
            start_max=4000,
            min_durantion=10,
            max_duration=100,
            n_shapes=20,
            amplitude_factor=10,
        )
    else:
        T_loc, L = inject_random_spikes_normal(
            T_loc, amplitide_factor=10, n_spikes=1000
        )
        events = None
    T_loc = np.reshape(T_loc, (12 * 12, -1))
    T_loc = np.reshape(T_loc, (12 * 24, 7, -1))
    L = np.reshape(L, (12 * 12, -1))
    L = np.reshape(L, (12 * 24, 7, -1))

    # T_loc = np.reshape(T_loc, (12 * 12, -1))
    # T_loc = np.reshape(T_loc, (12 * 12, 12 * 24, -1))
    # L = np.reshape(L, (12 * 12, -1))
    # L = np.reshape(L, (12 * 12, 12 * 24, -1))

    print(T_loc.shape)

    def objective(trial: optuna.Trial):
        rank = trial.suggest_int("rank", 1, 20)

        lmbda = (
            trial.suggest_float("lambda_0", 1e-2, 1e2, log=True),
            trial.suggest_float("lambda_1", 1e-2, 1e2, log=True),
            trial.suggest_float("lambda_2", 1e-2, 1e2, log=True),
        )
        # euclidean
        distance = trial.suggest_categorical(
            "distance", ["dot", "euclidean", "angular"]
        )
        k1 = trial.suggest_int("k1", 1, 144)
        k2 = trial.suggest_int("k2", 1, 133)
        k3 = trial.suggest_int("k3", 1, 500)
        # n_E = trial.suggest_int("n_E", 750, 1250)
        # n_E = 1000
        laps = [
            make_mode_laplacian(T_loc, mode=0, k=k1, normalize=True, measure=distance),
            make_mode_laplacian(T_loc, mode=1, k=k2, normalize=True, measure=distance),
            make_mode_laplacian(T_loc, mode=2, k=k3, normalize=True, measure=distance),
        ]

        recomp = partial(decomp_recomp, laps=laps, lambdas=lmbda)
        obj = test_eval(T_loc, L, events, rank=rank, recomp_func=recomp).f1

        return obj

    study = optuna.create_study(direction="maximize", study_name=name)
    study.optimize(objective, n_trials=70)
    print(study.best_params)
    print(study.best_value)
    # best k's: (10,9,64)
    # best lambdas: (0.1,0.01,0.2)


def main():
    T = np.load("data/abiline_ten.npy")
    # T_train = T[:, :, :5000]
    T_train = T[:, :, : 12 * 24 * 7 * 3]
    print(T_train.shape)
    # T_test = T[:, :, 10_000:15_000]
    del T
    # parameter_search(T_train, events_anomalies=True, name="events_based")
    # parameter_search(T_train, events_anomalies=False, name="spike_based")
    # exit()

    # events_rank = 18
    # events_lambda = (6.2, 2.9, 33.4)
    # events_k = (2, 10, 432)
    # events_laps = (
    #     make_mode_laplacian(T_train, mode=0, k=events_k[0], normalize=True),
    #     make_mode_laplacian(T_train, mode=1, k=events_k[1], normalize=True),
    #     make_mode_laplacian(T_train, mode=2, k=events_k[2], normalize=True),
    # )
    # events_recomp_func = partial(decomp_recomp, laps=events_laps, lambdas=events_lambda)
    # # cp_find_best_rank(T_train, events_anomalies=True, recomp_func=events_recomp_func)
    # # exit()
    # evaluate(
    #     T_train,
    #     events_rank,
    #     events_based=True,
    #     test="cp train events",
    #     recomp_func=events_recomp_func,
    # )
    # evaluate(
    #     T_test,
    #     events_rank,
    #     events_based=True,
    #     test="cp test events",
    #     recomp_func=events_recomp_func,
    # )

    spike_rank = 20
    spike_lambda = (0.2, 42, 0.6)
    spike_k = (84, 128, 477)
    spike_laps = (
        make_mode_laplacian(
            T_train, mode=0, k=spike_k[0], normalize=True, measure="angular"
        ),
        make_mode_laplacian(
            T_train, mode=1, k=spike_k[1], normalize=True, measure="angular"
        ),
        make_mode_laplacian(
            T_train, mode=2, k=spike_k[2], normalize=True, measure="angular"
        ),
    )
    spike_recomp_func = partial(decomp_recomp, laps=spike_laps, lambdas=spike_lambda)
    # cp_find_best_rank(T_train, events_anomalies=False, recomp_func=spike_recomp_func)
    # exit()
    evaluate(
        T_train,
        spike_rank,
        events_based=False,
        test="cp train spikes",
        recomp_func=spike_recomp_func,
    )


if __name__ == "__main__":
    main()
