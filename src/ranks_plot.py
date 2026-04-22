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
    # Graph Regularized Tucker (Robust)
    # Graph Regularized Tucker (Non-Robust version)
    {
        "class": MyGRTuckerDecomp,
        "kwargs": {
            "local_threshold": 0,
            "laplacian_parameters": model_confs["GRRTucker_no_robust"]["laps_params"],
        },
    },
    {
        "class": MyGRTuckerDecomp,
        "kwargs": {
            "local_threshold": None,
            "laplacian_parameters": model_confs["GRRTucker_no_robust"]["laps_params"],
        },
    },
    # Graph Regularized CP (Robust)
    {
        "class": MyGRTenDecomp,
        "kwargs": {
            "laplacian_parameters": model_confs["GRRCP_no_robust"]["laps_params"]
        },
    },
    # Graph Regularized CP (Non-Robust version)
    {
        "class": MyGRTenDecomp,
        "kwargs": {
            "local_threshold": 0,
            "laplacian_parameters": model_confs["GRRCP_no_robust"]["laps_params"],
        },
    },
    # Standard Models
    {"class": MyTuckerTenDecomp, "kwargs": {}},
    {"class": MyRHOOITenDecomp, "kwargs": {}},
    {"class": MyCPTenDecomp, "kwargs": {}},
    {"class": MyRCPTenDecomp, "kwargs": {}},  # Robust CP
]

# ... (imports and dataset functions remain the same)


def plotting():
    with open("results_rank_sensitivity_ddos.json", "r") as f:
        results = json.load(f)

    plt.figure(figsize=(10, 7))

    for model_name, data in results.items():
        if model_name[0] == "_":
            continue
        plt.errorbar(
            data["x"],
            data["mean_auc"],
            yerr=data["std_auc"],
            label=model_name,
            marker="o",
            capsize=5,
            linestyle="--",
            alpha=0.8,
        )

    plt.xlabel("Tensor Rank")
    plt.ylabel("Average PR AUC")
    plt.title("Rank Sensitivity Analysis")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


plotting()
exit()

ranks_to_test = [5, 10, 15, 20, 25]
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
    if spec["kwargs"].get("local_threshold") == 0:
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

            indices = np.argsort(recall)
            pr_auc = auc(recall[indices], precision[indices])
            aucs.append(pr_auc)

        mean_auc = float(np.mean(aucs))
        std_auc = float(np.std(aucs))

        results[model_name]["x"].append(r)
        results[model_name]["mean_auc"].append(mean_auc)
        results[model_name]["std_auc"].append(std_auc)

        print(f"Model: {model_name} | Rank: {r} | Mean AUC: {mean_auc:.4f}")

output_path = Path(f"results_rank_sensitivity_{ANOMALY_TYPE}.json")
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"Saved results to {output_path}")
