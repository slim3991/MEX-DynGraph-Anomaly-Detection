import mlflow
from utils.model_eval import evaluate_model
from models import (
    MyRCPTenDecomp,
    MyCPTenDecomp,
    MyGRTenDecomp,
    MyRHOOITenDecomp,
    MyTuckerTenDecomp,
)
import secrets


def main():
    models = [
        MyCPTenDecomp(rank=17),
        MyTuckerTenDecomp(ranks=(9, 11, 8)),
        MyRCPTenDecomp(rank=17, local_threshold=0.74),
        MyRHOOITenDecomp(ranks=(9, 12, 9), local_threshold=0.21),
        MyGRTenDecomp(
            rank=17,
            lambdas=(0.9, 0.04, 2),
            ks=(12, 1, 11),
            laps=None,
            measure="angular",
            local_threshold=0.5,
        ),
    ]
    tag = secrets.token_hex(4)
    tag = {"eval_run": tag}
    anomaly_type = "train"
    train_test = "spikes"

    mlflow.set_experiment(f"Evaluate Models")

    with mlflow.start_run(run_name=f"model evals, {tag['eval_run']}", tags=tag):
        for model in models:
            with mlflow.start_run(nested=True, tags=tag):
                mlflow.log_params(model.get_params())
                mlflow.log_param("name", model.name)
                mlflow.set_active_model(name=model.name)
                mlflow.log_params(
                    {"anomaly_type": anomaly_type, "train_test": train_test}
                )
                evaluate_model(
                    model, n_runs=10, anomaly_type=anomaly_type, train_test=train_test
                )


if __name__ == "__main__":
    main()
