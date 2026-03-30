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
        MyCPTenDecomp(rank=17, threshold=0.21),
        MyTuckerTenDecomp(ranks=(5, 11, 9), threshold=0.34),
        MyRCPTenDecomp(rank=20, local_threshold=0.74, threshold=0.16),
        MyRHOOITenDecomp(ranks=(5, 12, 13), threshold=0.35, local_threshold=0.39),
        MyGRTenDecomp(
            rank=17,
            lambdas=(38, 0.5, 0.08),
            ks=(8, 7, 5),
            laps=None,
            threshold=0.2,
            measure="dot",
            local_threshold=1.01,
        ),
    ]
    tag = secrets.token_hex(4)
    tag = {"eval_run": tag}
    anomaly_type = "train"
    train_test = "spikes"

    mlflow.set_experiment(f"Evaluate Models (w. events)")

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
