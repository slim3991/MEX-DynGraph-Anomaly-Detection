import secrets
import sys
import mlflow
import optuna
import numpy as np
import subprocess
from dataclasses import asdict
from typing import Callable, Any, Literal

from sklearn.metrics import auc, precision_recall_curve
from models.GRTucker import MyGRTuckerDecomp
from utils.datasets import (
    create_ddos_dataset,
    create_event_dataset,
    create_outage_dataset,
    create_spike_dataset,
)
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


#################################################################################
#################################################################################


def run_tensor_experiment(
    experiment_name: str,
    model_name: str,
    anomaly_type: Literal["events", "spikes", "ddos", "outage"],
    suggest_and_build_model: Callable[[optuna.Trial, Any], Any],
    n_trials: int = 30,
    tag=None,
):
    model_tag = secrets.token_hex(4)

    mlflow.set_experiment(experiment_name)
    data_fetch_funcs = {
        "spikes": create_spike_dataset,
        "events": create_event_dataset,
        "ddos": create_ddos_dataset,
        "outage": create_outage_dataset,
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
            prauc_sum = 0.0
            for _ in range(n):

                T, L, _, _ = data_fetch_func(train_test="train")
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
                precision, recall, _ = precision_recall_curve(L.ravel(), resids.ravel())
                recall, precision = zip(*sorted(zip(recall, precision)))
                pr_auc = float(auc(recall, precision))
                prauc_sum += pr_auc

                # ave_metric = metrics if ave_metrics is None else ave_metrics + metrics
            ave_prauc = prauc_sum / n

            return ave_prauc

    with mlflow.start_run(
        run_name=f"{model_name}-({model_tag})-({anomaly_type})", tags={"run_tag": tag}
    ):
        mlflow.log_params(
            {
                "model_name": model_name,
                "model_tag": model_tag,
                "group_tag": tag,
                "git_hash": get_git_hash(),
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
    lap_params = {
        "lambda_1": 0,  # trial.suggest_float("lambda_1", 1e-4, 1e4, log=True),
        "lambda_2": 0,  # trial.suggest_float("lambda_2", 1e-4, 1e4, log=True),
        "lambda_smooth": trial.suggest_float("lambda_smooth", 0, 1e3),
        "lambda_interval": trial.suggest_float("lambda_intervall", 0, 1e3),
        # "distance": trial.suggest_categorical(
        #     "distance", ["dot", "euclidean", "angular"]
        # ),
        "k1": 0,  # trial.suggest_int("k1", 0, min(T.shape[0], 50)),
        "k2": 0,  # trial.suggest_int("k2", 0, min(T.shape[1], 50)),
    }

    model = MyGRTuckerDecomp(
        rank=15,
        local_threshold=0,
        tol=1e-4,
        laplacian_parameters=lap_params,
    )
    return model, lap_params


def grten_builder_no_robust(trial, T):
    lap_params = {
        "lambda_1": 0,  # trial.suggest_float("lambda_1", 1e-4, 1e4, log=True),
        "lambda_2": 0,  # trial.suggest_float("lambda_2", 1e-4, 1e4, log=True),
        "lambda_smooth": trial.suggest_float("lambda_smooth", 0, 1e4),
        "lambda_interval": trial.suggest_float("lambda_intervall", 0, 1e4),
        # "distance": trial.suggest_categorical(
        #     "distance", ["dot", "euclidean", "angular"]
        # ),
        "k1": 0,  # trial.suggest_int("k1", 0, min(T.shape[0], 50)),
        "k2": 0,  # trial.suggest_int("k2", 0, min(T.shape[1], 50)),
    }

    model = MyGRTenDecomp(
        rank=15,
        local_threshold=0,
        tol=1e-4,
        laplacian_parameters=lap_params,
    )
    return model, lap_params


def main():
    tag = secrets.token_hex(4)
    anomaly_type = "ddos"

    anomalies = ("spikes", "events", "ddos", "outage")
    for anomaly in anomalies:
        # run_tensor_experiment(
        #     experiment_name="Tensor_Decomp",
        #     model_name="GRTen",
        #     suggest_and_build_model=grten_builder_no_robust,
        #     anomaly_type=anomaly,
        #     tag=tag,
        # )

        run_tensor_experiment(
            experiment_name="Tensor_Decomp",
            model_name="GRTucker",
            suggest_and_build_model=grTucker_builder_no_robust,
            anomaly_type=anomaly,
            tag=tag,
        )


if __name__ == "__main__":
    main()
