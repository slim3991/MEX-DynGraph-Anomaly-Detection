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
        # MyCPTenDecomp(rank=20, threshold=0.17),
        # MyTuckerTenDecomp(ranks=(11, 10, 10), threshold=0.26),
        # MyRCPTenDecomp(rank=20, local_threshold=0.88, threshold=0.19),
        # MyRHOOITenDecomp(ranks=(11, 11, 15), threshold=0.3, local_threshold=1.9),
        MyGRTenDecomp(
            rank=17,
            lambdas=(0.03, 0.02, 24),
            ks=(4, 5, 8),
            laps=None,
            threshold=0.002,
            measure="dot",
            local_threshold=0.03,
        ),
    ]
    tag = secrets.token_hex(4)
    tag = {"eval_run": tag}
    anomaly_type = "events"
    train_test = "eval"

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
