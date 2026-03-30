import secrets
import sys
import mlflow
import optuna
import numpy as np
import subprocess
from dataclasses import asdict
from typing import Callable, Any
from utils.datasets import create_event_dataset_train, create_spike_dataset_train
from utils.metrics import compute_metrics_with_threshold
from models import MyGRTenDecomp
from models.BasicCP import MyCPTenDecomp
from models.BasicTucker import MyTuckerTenDecomp
from models.RHOOI_model import MyRHOOITenDecomp
from models.RobustCp import MyRCPTenDecomp
from utils.tensor_processing import make_mode_laplacian


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
    anomaly_type: str = "spike",
    n_trials: int = 40,
    seed: int = 42,
    tag=None,
):
    check_git_status()

    np.random.seed(seed)
    mlflow.set_experiment(experiment_name)

    # Data Setup
    data_fetch_func = (
        create_spike_dataset_train
        if anomaly_type == "spike"
        else create_event_dataset_train
    )
    T, L, events, data_params = data_fetch_func()

    def objective(trial):
        with mlflow.start_run(
            nested=True,
            tags={
                "group_tag": tag,
                "model_name": model_name,
                "level": "model",
            },
        ):
            model, trial_params = suggest_and_build_model(trial, T)

            trial_params["anomaly_type"] = anomaly_type
            mlflow.log_params(trial_params)
            model.fit(T, L)
            resids = model.residuals(T)
            metrics = compute_metrics_with_threshold(
                resids, L, model.threshold_, events=events
            )

            metrics_dict = {k: v for k, v in asdict(metrics).items() if v is not None}
            mlflow.log_metrics(metrics_dict)
            return metrics.pr_auc if metrics.pr_auc is not None else 0

    mlflow.set_model(model_name)
    with mlflow.start_run(
        nested=True,
        run_name=model_name,
        tags={"run_tag": tag, "model_name": model_name, "level": "trial"},
    ):
        mlflow.log_params(
            {
                "model_name": model_name,
                "git_hash": get_git_hash(),
                "seed": seed,
                "anomaly_type": anomaly_type,
                **data_params,
            }
        )

        study = optuna.create_study(
            direction="maximize", study_name=model_name + f"-{tag}"
        )
        study.optimize(objective, n_trials=n_trials)

        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_pr_auc", study.best_value)

    print(f"Best value: {study.best_value} for {model_name}")


#################################################################################
#################################################################################


def grten_builder(trial, T):
    trial_params = {
        "rank": trial.suggest_int("rank", 1, 20),
        "lambda_0": trial.suggest_float("lambda_0", 1e-2, 1e2, log=True),
        "lambda_1": trial.suggest_float("lambda_1", 1e-2, 1e2, log=True),
        "lambda_2": trial.suggest_float("lambda_2", 1e-2, 1e2, log=True),
        "distance": trial.suggest_categorical(
            "distance", ["dot", "euclidean", "angular"]
        ),
        "k1": trial.suggest_int("k1", 1, min(T.shape[0], 500)),
        "k2": trial.suggest_int("k2", 1, min(T.shape[0], 500)),
        "k3": trial.suggest_int("k3", 1, min(T.shape[0], 500)),
        "local_threshold": trial.suggest_float("local_threshold", 0, 2),
    }

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
        local_threshold=trial_params["local_threshold"],
        laps=laps,
    )
    return model, trial_params


def cp_builder(trial, T):
    params = {
        "rank": trial.suggest_int("rank", 1, 20),
    }
    model = MyCPTenDecomp(rank=params["rank"])
    return model, params


def tucker_builder(trial, T):
    n1, n2, n3 = T.shape
    params = {
        "rank_0": trial.suggest_int("rank_0", 1, min(20, n1)),
        "rank_1": trial.suggest_int("rank_1", 1, min(20, n2)),
        "rank_2": trial.suggest_int("rank_2", 1, min(20, n3)),
    }
    ranks = (params["rank_0"], params["rank_1"], params["rank_2"])
    model = MyTuckerTenDecomp(ranks=ranks)
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
        ranks=ranks,
        local_threshold=params["local_threshold"],
    )
    return model, params


def robust_cp_builder(trial, T):
    params = {
        "rank": trial.suggest_int("rank", 1, 20),
        "local_threshold": trial.suggest_float("local_threshold", 0, 2),
    }

    model = MyRCPTenDecomp(
        rank=params["rank"],
        local_threshold=params["local_threshold"],
    )

    return model, params


def main():
    tag = secrets.token_hex(4)

    with mlflow.start_run(run_name=f"Group_experiment_tag{tag}"):

        # run_tensor_experiment(
        #     experiment_name="Tensor_Decomp",
        #     model_name="BasicCP",
        #     suggest_and_build_model=cp_builder,
        #     anomaly_type="events",
        #     tag=tag,
        # )

        # run_tensor_experiment(
        #     experiment_name="Tensor_Decomp",
        #     model_name="Basic Tucker",
        #     suggest_and_build_model=tucker_builder,
        #     anomaly_type="spike",
        #     tag=tag,
        # )
        # run_tensor_experiment(
        #     experiment_name="Tensor_Decomp",
        #     model_name="Robust CP",
        #     suggest_and_build_model=robust_cp_builder,
        #     anomaly_type="spike",
        #     tag=tag,
        # )
        # run_tensor_experiment(
        #     experiment_name="Tensor_Decomp",
        #     model_name="RHOOI",
        #     suggest_and_build_model=rhooi_builder,
        #     anomaly_type="spike",
        #     tag=tag,
        # )
        run_tensor_experiment(
            experiment_name="Tensor_Decomp",
            model_name="GRTen",
            suggest_and_build_model=grten_builder,
            anomaly_type="spike",
            tag=tag,
        )


if __name__ == "__main__":
    main()
