from typing import Dict
import mlflow
from models import (
    MyRCPTenDecomp,
    MyCPTenDecomp,
    MyGRTenDecomp,
    MyRHOOITenDecomp,
    MyTuckerTenDecomp,
)
import secrets
from dataclasses import asdict
from typing import Callable, Dict, Literal, Optional, Protocol, Tuple
import mlflow
from tqdm import tqdm
import numpy.typing as npt

from utils.metrics import compute_metrics_with_threshold, print_metrics
from utils.datasets import (
    create_event_dataset_train,
    create_event_dataset_test,
    create_spike_dataset_train,
    create_spike_dataset_test,
)

dataset_fetch_func = Callable[[], Tuple[npt.NDArray, npt.NDArray, Optional[list], dict]]


def main():
    models = [
        MyCPTenDecomp(
            rank=7,
            threshold=0.7,
        ),
        MyTuckerTenDecomp(
            ranks=(7, 3, 20),
            threshold=0.76,
        ),
        MyRCPTenDecomp(
            rank=20,
            local_threshold=0.02,
            threshold=0.5,
        ),
        MyRHOOITenDecomp(
            ranks=(8, 10, 10),
            local_threshold=0.03,
            threshold=0.4,
        ),
        MyGRTenDecomp(
            rank=15,
            lambdas=(0.6, 0.01, 1.1),
            ks=(5, 1, 4),
            laps=None,
            measure="angular",
            local_threshold=1.8,
            threshold=0.31,
        ),
    ]
    tag = secrets.token_hex(4)
    tag = {"eval_run": tag}
    anomaly_type = "events"
    train_test = "train"

    mlflow.set_experiment(f"Evaluate Models")

    with mlflow.start_run(
        run_name=f"eval-{tag['eval_run']}({anomaly_type}-{train_test})", tags=tag
    ):
        mlflow.log_params(
            {"anomaly_type": anomaly_type, "train_test": train_test, "eval_tag": tag}
        )
        for model in models:

            with mlflow.start_run(
                run_name=f"eval-{tag['eval_run']}_model-{model.name}({anomaly_type}-{train_test})",
                nested=True,
                tags=tag,
            ):
                mlflow.log_params(model.get_params())
                mlflow.log_param("name", model.name)
                mlflow.set_active_model(name=model.name)
                mlflow.log_params(
                    {
                        "anomaly_type": anomaly_type,
                        "train_test": train_test,
                        "eval_tag": tag,
                    }
                )
                metrics: Dict[str, float] = evaluate_model(
                    model,
                    n_runs=10,
                    anomaly_type=anomaly_type,
                    train_test=train_test,
                    threshold=model.threshold,
                )
                mlflow.log_metrics(metrics=metrics)


def fetch_dataset_fetch_func(
    anomaly_type: Literal["spikes", "events"] = "spikes",
    train_test: Literal["train", "test"] = "test",
) -> dataset_fetch_func:
    if anomaly_type == "spikes":
        return (
            create_spike_dataset_train
            if train_test == "train"
            else create_spike_dataset_test
        )
    else:
        return (
            create_event_dataset_train
            if train_test == "train"
            else create_event_dataset_test
        )


class Transformer(Protocol):
    def fit_transform(self, X, y): ...
    @property
    def name(self) -> str: ...
    def get_params(self) -> dict: ...


def evaluate_model(
    model: Transformer,
    n_runs: int = 10,
    anomaly_type: Literal["spikes", "events"] = "spikes",
    train_test: Literal["train", "test"] = "test",
    threshold: Optional[float] = None,
) -> Dict[str, float]:
    metric_sum = None
    dataset_func = fetch_dataset_fetch_func(
        anomaly_type=anomaly_type, train_test=train_test
    )
    for _ in tqdm(range(n_runs)):
        T, L, events, _ = dataset_func()
        T_hat = model.fit_transform(T, L)

        resids = T - T_hat
        metrics = compute_metrics_with_threshold(
            resids, L, threshold=threshold, events=events
        )
        tqdm.write("PR_AUC: " + str(metrics.pr_auc))
        metric_sum = metrics if metric_sum is None else metric_sum + metrics
    ave_metrics = metric_sum / n_runs
    ave_dict = asdict(ave_metrics)
    ave_dict = {k: v for k, v in ave_dict.items() if v is not None}

    mlflow.log_metrics(ave_dict)

    print_metrics(ave_metrics)
    return ave_dict


if __name__ == "__main__":
    main()
