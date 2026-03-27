from itertools import product
import numpy as np
import optuna

from copy import deepcopy

from utils.tensor_processing import de_anomalize_tensor, normalize_tensor
from utils.anomaly_injector import *
from utils.model_eval import *


def preprocess(T):
    for i in range(12):
        for j in range(12):
            T[i, j, :] = normalize_tensor(T[i, j, :], "minmax")
    T = de_anomalize_tensor(T, low_rank=20, keep_pecentile=95, alpha=0.8)
    return T


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


def cp_find_best_rank(T, events_anomalies: bool, recomp_func):
    T_loc = deepcopy(preprocess(T))
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
    best_rank = 0
    best_obj = -np.inf
    for rank in range(5, 26):
        print(f"Computing rank: {rank} ...")
        obj = test_eval(T_loc, L, events, rank=rank, recomp_func=recomp_func).f1
        if obj > best_obj:
            best_obj = obj
            best_rank = rank
            print(f"New best! obj: {best_obj}, rank: {best_rank}")
    return best_rank


def tucker_find_best_rank_grid_search(T, events_anomalies: bool, recomp_func):
    # 1. Preprocessing happens once outside the objective to save time
    T_loc = deepcopy(preprocess(T))

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

    # 2. Define the objective function for Optuna
    def objective(trial):
        # Suggest values for the rank dimensions
        i = trial.suggest_int("rank_i", 5, 12)
        j = trial.suggest_int("rank_j", 5, 12)
        k = trial.suggest_int("rank_k", 5, 20)

        # Evaluate
        result = test_eval(T_loc, L, events, (i, j, k), recomp_func=recomp_func)
        obj = result.f1

        # Optuna handles None as a failure or you can return a default low score
        return obj if obj is not None else -1.0

    # 3. Create a study and optimize
    # We use "maximize" because F1-score is better when higher
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)

    best_rank = (
        study.best_params["rank_i"],
        study.best_params["rank_j"],
        study.best_params["rank_k"],
    )

    print(f"Best score: {study.best_value}")
    print(f"Best rank found: {best_rank}, events anomalies: {events_anomalies}")

    return best_rank, study.best_value


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


def main():
    pass


if __name__ == "__main__":
    main()
