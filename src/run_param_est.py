from math import radians
import secrets
import sys
import mlflow
import optuna
import numpy as np
import subprocess
from dataclasses import asdict
from typing import Callable, Any, Literal
from models.GRTucker import MyGRTuckerDecomp
from utils.datasets import create_event_dataset_train, create_spike_dataset_train
from utils.metrics import Metrics, compute_metrics_with_threshold
from models import MyGRTenDecomp
from models.BasicCP import MyCPTenDecomp
from models.BasicTucker import MyTuckerTenDecomp
from models.RHOOI_model import MyRHOOITenDecomp
from models.RobustCp import MyRCPTenDecomp


has_asked = False

type anomaly_types = Literal["spikes", "events"]


def get_git_hash():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except:
        return "unknown"


def check_git_status():
    """Checks if there are uncommitted changes and warns the user."""
    global has_asked
    if has_asked:
        return
    try:
        # --porcelain gives a script-friendly output; empty means clean
        status = (
            subprocess.check_output(["git", "status", "--porcelain"])
            .decode("utf-8")
            .strip()
        )
        if status:
            print("\n" + "!" * 60)
            print("⚠️  WARNING: UNCOMMITTED CHANGES DETECTED")
            print("The current Git hash will NOT accurately represent this run.")
            print("Files changed:")
            print(status)
            print("!" * 60 + "\n")

            # Optional: Force a confirmation
            response = input("Do you want to proceed anyway? (y/n): ")
            if response.lower() != "y":
                print("Aborting run. Commit your changes first!")
                sys.exit(1)
            has_asked = True
        else:
            print("✅ Git repository is clean. Proceeding...")
    except subprocess.CalledProcessError:
        print("❌ Error: Could not verify Git status. Is this a git repo?")


#################################################################################
#################################################################################


def run_tensor_experiment(
    experiment_name: str,
    model_name: str,
    suggest_and_build_model: Callable[[optuna.Trial, Any], Any],
    anomaly_type: Literal["events", "spikes"] = "events",
    n_trials: int = 40,
    seed: int = 42,
    tag=None,
):
    check_git_status()
    model_tag = secrets.token_hex(4)

    np.random.seed(seed)
    mlflow.set_experiment(experiment_name)
    data_fetch_funcs = {
        "spikes": create_spike_dataset_train,
        "events": create_event_dataset_train,
    }

    # Data Setup
    data_fetch_func = data_fetch_funcs[anomaly_type]

    def objective(trial: optuna.Trial):
        trial_number = trial.number
        with mlflow.start_run(
            run_name=f"{model_name}-({model_tag})-run{trial_number}-({anomaly_type})",
            nested=True,
            tags={"group_tag": tag},
        ):
            n = 3
            ave_metrics = None
            for _ in range(n):
                T, L, events, data_params = data_fetch_func()
                model, trial_params = suggest_and_build_model(trial, T)
                model.tol = 1e-4

                trial_params["anomaly_type"] = anomaly_type
                mlflow.log_params(
                    {
                        **trial_params,
                        "trial_num": trial.number,
                        "model_tag": model_tag,
                    }
                )
                model.fit(T, L)
                resids = model.residuals(T)
                metrics = compute_metrics_with_threshold(
                    resids, L, model.threshold_, events=events
                )
                if ave_metrics is None:
                    ave_metrics = metrics
                else:
                    ave_metrics += metrics

                # ave_metric = metrics if ave_metrics is None else ave_metrics + metrics
            assert ave_metrics is not None
            ave_metrics = ave_metrics / n

            metrics_dict = {
                k: v for k, v in asdict(ave_metrics).items() if v is not None
            }
            mlflow.log_metrics(metrics_dict)
            return ave_metrics.pr_auc if ave_metrics.pr_auc is not None else 0

    with mlflow.start_run(
        run_name=f"{model_name}-({model_tag})-({anomaly_type})", tags={"run_tag": tag}
    ):
        mlflow.log_params(
            {
                "model_name": model_name,
                "model_tag": model_tag,
                "group_tag": tag,
                "git_hash": get_git_hash(),
                "seed": seed,
                "anomaly_type": anomaly_type,
            }
        )

        study = optuna.create_study(
            direction="maximize",
            study_name=f"study: ({tag}), {model_name}-({model_tag})",
        )
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("pr_auc", study.best_value)

    print(f"Best value: {study.best_value} for {model_name}")


#################################################################################
#################################################################################


def grTucker_builder_no_robust(trial, T):
    trial_params = {
        "rank_0": trial.suggest_int("rank_0", 7, 12),
        "rank_1": trial.suggest_int("rank_1", 7, 12),
        "rank_2": trial.suggest_int("rank_2", 7, 30),
        "lambda_0": trial.suggest_float("lambda_0", 1e-4, 1e2, log=True),
        "lambda_1": trial.suggest_float("lambda_1", 1e-4, 1e2, log=True),
        "lambda_2": trial.suggest_float("lambda_2", 1e-4, 1e2, log=True),
        "distance": trial.suggest_categorical(
            "distance", ["dot", "euclidean", "angular"]
        ),
        "k1": trial.suggest_int("k1", 0, min(T.shape[0], 50)),
        "k2": trial.suggest_int("k2", 0, min(T.shape[1], 50)),
        "k3": trial.suggest_int("k3", 0, min(T.shape[2], 200)),
    }

    lambdas = [
        trial_params["lambda_0"],
        trial_params["lambda_1"],
        trial_params["lambda_2"],
    ]
    ks = [
        trial_params["k1"],
        trial_params["k2"],
        trial_params["k3"],
    ]
    ranks = (
        trial_params["rank_0"],
        trial_params["rank_1"],
        trial_params["rank_2"],
    )

    model = MyGRTuckerDecomp(
        rank=ranks,
        lambdas=lambdas,
        ks=ks,
        local_threshold=0,
        measure=trial_params["distance"],
        tol=1e-4,
    )
    return model, trial_params


def grTucker_builder(trial, T):
    trial_params = {
        "rank_0": trial.suggest_int("rank_0", 7, 13),
        "rank_1": trial.suggest_int("rank_1", 7, 13),
        "rank_2": trial.suggest_int("rank_2", 7, 30),
        "lambda_0": trial.suggest_float("lambda_0", 1e-4, 1e4),
        "lambda_1": trial.suggest_float("lambda_1", 1e-4, 1e4),
        "lambda_2": trial.suggest_float("lambda_2", 1e-4, 1e4),
        "distance": trial.suggest_categorical(
            "distance", ["dot", "euclidean", "angular"]
        ),
        "k1": trial.suggest_int("k1", 0, min(T.shape[0], 50)),
        "k2": trial.suggest_int("k2", 0, min(T.shape[1], 50)),
        "k3": trial.suggest_int("k3", 0, min(T.shape[2], 200)),
        "local_threshold": trial.suggest_float("local_threshold", 0, 3),
    }

    lambdas = [
        trial_params["lambda_0"],
        trial_params["lambda_1"],
        trial_params["lambda_2"],
    ]
    ks = [
        trial_params["k1"],
        trial_params["k2"],
        trial_params["k3"],
    ]
    ranks = (
        trial_params["rank_0"],
        trial_params["rank_1"],
        trial_params["rank_2"],
    )

    model = MyGRTuckerDecomp(
        rank=ranks,
        lambdas=lambdas,
        ks=ks,
        local_threshold=trial_params["local_threshold"],
        measure=trial_params["distance"],
        tol=1e-4,
    )
    return model, trial_params


def grten_builder(trial, T):
    trial_params = {
        "rank": trial.suggest_int("rank", 7, 30),
        "lambda_0": trial.suggest_float("lambda_0", 1e-4, 1e4),
        "lambda_1": trial.suggest_float("lambda_1", 1e-4, 1e4),
        "lambda_2": trial.suggest_float("lambda_2", 1e-4, 1e4),
        "distance": trial.suggest_categorical(
            "distance", ["dot", "euclidean", "angular"]
        ),
        "k1": trial.suggest_int("k1", 0, min(T.shape[0], 50)),
        "k2": trial.suggest_int("k2", 0, min(T.shape[1], 50)),
        "k3": trial.suggest_int("k3", 0, min(T.shape[2], 200)),
        "local_threshold": trial.suggest_float("local_threshold", 0, 3),
    }

    lambdas = [
        trial_params["lambda_0"],
        trial_params["lambda_1"],
        trial_params["lambda_2"],
    ]
    ks = [
        trial_params["k1"],
        trial_params["k2"],
        trial_params["k3"],
    ]

    model = MyGRTenDecomp(
        rank=trial_params["rank"],
        lambdas=lambdas,
        ks=ks,
        local_threshold=trial_params["local_threshold"],
        measure=trial_params["distance"],
        tol=1e-4,
    )
    return model, trial_params


def grten_builder_no_robust(trial, T):
    trial_params = {
        "rank": trial.suggest_int("rank", 7, 20),
        "lambda_0": trial.suggest_float("lambda_0", 1e-4, 1e2, log=True),
        "lambda_1": trial.suggest_float("lambda_1", 1e-4, 1e2, log=True),
        "lambda_2": trial.suggest_float("lambda_2", 1e-4, 1e2, log=True),
        "distance": trial.suggest_categorical(
            "distance", ["dot", "euclidean", "angular"]
        ),
        "k1": trial.suggest_int("k1", 0, min(T.shape[0], 50)),
        "k2": trial.suggest_int("k2", 0, min(T.shape[1], 50)),
        "k3": trial.suggest_int("k3", 0, min(T.shape[2], 200)),
    }

    lambdas = [
        trial_params["lambda_0"],
        trial_params["lambda_1"],
        trial_params["lambda_2"],
    ]
    ks = [
        trial_params["k1"],
        trial_params["k2"],
        trial_params["k3"],
    ]

    model = MyGRTenDecomp(
        rank=trial_params["rank"],
        lambdas=lambdas,
        ks=ks,
        local_threshold=0,
        measure=trial_params["distance"],
        tol=1e-4,
    )
    return model, trial_params


def cp_builder(trial, T):
    params = {
        "rank": trial.suggest_int("rank", 1, 20),
    }
    model = MyCPTenDecomp(rank=params["rank"], tol=1e-4)
    return model, params


def tucker_builder(trial, T):
    n1, n2, n3 = T.shape
    params = {
        "rank_0": trial.suggest_int("rank_0", 1, min(20, n1)),
        "rank_1": trial.suggest_int("rank_1", 1, min(20, n2)),
        "rank_2": trial.suggest_int("rank_2", 1, min(20, n3)),
    }
    ranks = (params["rank_0"], params["rank_1"], params["rank_2"])
    model = MyTuckerTenDecomp(ranks=ranks, tol=1e-4)
    return model, params


def rhooi_builder(trial, T):
    n1, n2, n3 = T.shape
    params = {
        "rank_0": trial.suggest_int("rank_0", 1, min(20, n1)),
        "rank_1": trial.suggest_int("rank_1", 1, min(20, n2)),
        "rank_2": trial.suggest_int("rank_2", 1, min(20, n3)),
        "local_threshold": trial.suggest_float("local_threshold", 0, 3),
    }

    ranks = (params["rank_0"], params["rank_1"], params["rank_2"])
    model = MyRHOOITenDecomp(
        ranks=ranks, local_threshold=params["local_threshold"], tol=1e-4
    )
    return model, params


def robust_cp_builder(trial, T):
    params = {
        "rank": trial.suggest_int("rank", 1, 20),
        "local_threshold": trial.suggest_float("local_threshold", 0, 2),
    }

    model = MyRCPTenDecomp(
        rank=params["rank"], local_threshold=params["local_threshold"], tol=1e-4
    )

    return model, params


def main():
    tag = secrets.token_hex(4)
    anomaly_type = "spikes"
    run_tensor_experiment(
        experiment_name="Tensor_Decomp",
        model_name="GRTucker",
        suggest_and_build_model=grTucker_builder,
        anomaly_type=anomaly_type,
        tag=tag,
    )

    # run_tensor_experiment(
    #     experiment_name="Tensor_Decomp",
    #     model_name="GRTucker_no_robust",
    #     suggest_and_build_model=grTucker_builder_no_robust,
    #     anomaly_type=anomaly_type,
    #     tag=tag,
    # )
    # run_tensor_experiment(
    #     experiment_name="Tensor_Decomp",
    #     model_name="GRTen no Robust",
    #     suggest_and_build_model=grten_builder_no_robust,
    #     anomaly_type=anomaly_type,
    #     tag=tag,
    # )
    run_tensor_experiment(
        experiment_name="Tensor_Decomp",
        model_name="GRTen",
        suggest_and_build_model=grten_builder,
        anomaly_type=anomaly_type,
        tag=tag,
    )

    # run_tensor_experiment(
    #     experiment_name="Tensor_Decomp",
    #     model_name="Robust CP",
    #     suggest_and_build_model=robust_cp_builder,
    #     anomaly_type=anomaly_type,
    #     tag=tag,
    # )
    #
    # run_tensor_experiment(
    #     experiment_name="Tensor_Decomp",
    #     model_name="RHOOI",
    #     suggest_and_build_model=rhooi_builder,
    #     anomaly_type=anomaly_type,
    #     tag=tag,
    # )
    # run_tensor_experiment(
    #     experiment_name="Tensor_Decomp",
    #     model_name="BasicCP",
    #     suggest_and_build_model=cp_builder,
    #     anomaly_type=anomaly_type,
    #     tag=tag,
    # )
    #
    # run_tensor_experiment(
    #     experiment_name="Tensor_Decomp",
    #     model_name="Basic Tucker",
    #     suggest_and_build_model=tucker_builder,
    #     anomaly_type=anomaly_type,
    #     tag=tag,
    # )


if __name__ == "__main__":
    main()
