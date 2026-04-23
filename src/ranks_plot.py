from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Protocol, Tuple
import json
import numpy as np
import numpy.typing as npt
from scipy.sparse.csgraph import johnson
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

from models.BasicCP import MyCPTenDecomp
from models.BasicTucker import MyTuckerTenDecomp
from models.GRTenDecomp import MyGRTenDecomp
from models.GRTucker import MyGRTuckerDecomp
from models.RHOOI_model import MyRHOOITenDecomp
from models.RobustCp import MyRCPTenDecomp
from utils.datasets import (
    create_ddos_dataset,
    create_event_dataset,
    create_outage_dataset,
    create_spike_dataset,
)

ANOMALY_TYPE = "ddos"


with open("src/model_config.yaml") as f:
    m_conf = yaml.safe_load(f)
model_confs = m_conf[f"{ANOMALY_TYPE}_configs"]

model_specs = [
    # {
    #     "class": MyGRTenDecomp,
    #     "kwargs": {
    #         "local_threshold": 0,
    #         "laplacian_parameters": model_confs["GRRCP_no_robust"]["laps_params"],
    #     },
    # },
    # {
    #     "class": MyGRTenDecomp,
    #     "kwargs": {
    #         "laplacian_parameters": model_confs["GRRCP_no_robust"]["laps_params"]
    #     },
    # },
    # {
    #     "class": MyGRTuckerDecomp,
    #     "kwargs": {
    #         "local_threshold": 0,
    #         "laplacian_parameters": model_confs["GRRTucker_no_robust"]["laps_params"],
    #     },
    # },
    # {
    #     "class": MyGRTuckerDecomp,
    #     "kwargs": {
    #         "local_threshold": None,
    #         "laplacian_parameters": model_confs["GRRTucker_no_robust"]["laps_params"],
    #     },
    # },
    # {"class": MyTuckerTenDecomp, "kwargs": {}},
    {"class": MyRHOOITenDecomp, "kwargs": {}},
    # {"class": MyCPTenDecomp, "kwargs": {}},
    {"class": MyRCPTenDecomp, "kwargs": {}},  # Robust CP
]


def plotting(anomaly_type: Optional[str] = None):
    if anomaly_type is None:
        anomaly_type = "ddos"

    with open(f"figures/results_rank_sensitivity_{anomaly_type}.json", "r") as f:
        results = json.load(f)

    _, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)

    for model_name, data in results.items():
        if model_name.startswith("_"):
            continue

        plot_params = {
            "x": data["x"],
            "y": data["mean_auc"],
            "yerr": data["std_auc"],
            "label": model_name,
            "marker": "o",
            "capsize": 5,
            "linestyle": "--",
            "alpha": 0.8,
        }

        if "Tucker" in model_name or "RHOOI" in model_name:
            ax1.errorbar(**plot_params)
        else:
            ax2.errorbar(**plot_params)

    anomaly_type = "low-rank"
    ax1.set_ylabel("Average PR AUC")
    ax1.set_title(f"Rank Sensitivity Tucker ({anomaly_type})")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Tensor Rank")
    ax2.set_ylabel("Average PR AUC")
    ax2.set_title(f"Rank Sensitivity CP ({anomaly_type})")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


plotting(ANOMALY_TYPE)
exit()

ranks_to_test = [10, 15, 20, 25, 30]
fixed_ampf = 8
n_runs = 3


dataset_fetchers = {
    "ddos": create_ddos_dataset,
    "spikes": create_spike_dataset,
    "events": create_event_dataset,
    "outage": create_outage_dataset,
}

results = {}
results["_meta"] = {
    "anomaly_type": ANOMALY_TYPE,
    "n_runs": n_runs,
    "ranks": ranks_to_test,
}

print(ANOMALY_TYPE)
for spec in model_specs:
    model_name = spec["class"].name()
    if spec["kwargs"].get("local_threshold") != 0:
        model_name += "-thresholded"
    print(model_name)

    results[model_name] = {
        "x": [],
        "mean_auc": [],
        "std_auc": [],
    }

    for r in ranks_to_test:
        aucs = []

        for i in range(n_runs):
            model = spec["class"](rank=r, tol=1e-4, **spec["kwargs"])

            T, L, _, _ = dataset_fetchers[ANOMALY_TYPE]("test")
            T_hat = model.fit_transform(T, L)
            resids = np.abs(T - T_hat)

            precision, recall, _ = precision_recall_curve(L.ravel(), resids.ravel())
            pr_auc = auc(recall, precision)
            aucs.append(pr_auc)

        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))

        results[model_name]["x"].append(r)
        results[model_name]["mean_auc"].append(mean_auc)
        results[model_name]["std_auc"].append(std_auc)

        print(
            f"Model: {model_name} | Rank: {r} | Mean AUC: {mean_auc:.4f}±{std_auc:.4f}"
        )

output_path = Path(f"figures/results_rank_sensitivity_{ANOMALY_TYPE}.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

plotting(ANOMALY_TYPE)

print(f"Saved results to {output_path}")
