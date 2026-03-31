from typing import Dict
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
        MyCPTenDecomp(rank=17, threshold=0.4),
        MyTuckerTenDecomp(ranks=(1, 8, 8), threshold=0.65),
        MyRCPTenDecomp(rank=20, local_threshold=0.05),
        MyRHOOITenDecomp(ranks=(11, 12, 7), local_threshold=0.61),
        MyGRTenDecomp(
            rank=7,
            lambdas=(0.3, 0.01, 0.04),
            ks=(5, 5, 1),
            laps=None,
            measure="angular",
            local_threshold=1.5,
            threshold=0.71,
        ),
    ]
    tag = secrets.token_hex(4)
    tag = {"eval_run": tag}
    anomaly_type = "spikes"
    train_test = "test"

    mlflow.set_experiment(f"Evaluate Models")

    with mlflow.start_run(run_name=f"model evals, {tag['eval_run']}", tags=tag):
        mlflow.log_params(
            {
                "anomaly_type": anomaly_type,
                "train_test": train_test,
            }
        )
        for model in models:

            with mlflow.start_run(nested=True, tags=tag):
                mlflow.log_params(model.get_params())
                mlflow.log_param("name", model.name)
                mlflow.set_active_model(name=model.name)
                mlflow.log_params(
                    {"anomaly_type": anomaly_type, "train_test": train_test}
                )
                metrics: Dict[str, float] = evaluate_model(
                    model,
                    n_runs=10,
                    anomaly_type=anomaly_type,
                    train_test=train_test,
                    threshold=model.threshold,
                )
                mlflow.log_metrics(metrics=metrics)


if __name__ == "__main__":
    main()
