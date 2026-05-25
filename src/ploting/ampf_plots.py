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


def model_specs_gen(anomaly_type):
    with open("src/model_config.yaml") as f:
        m_conf = yaml.safe_load(f)
    model_confs = m_conf[f"{anomaly_type}_configs"]
    model_specs = [
        {
            "class": MyGRTenDecomp,
            "kwargs": {
                "local_threshold": 0,
                "laplacian_parameters": model_confs["GRRCP_no_robust"]["laps_params"],
            },
        },
        {
            "class": MyGRTenDecomp,
            "kwargs": {
                "laplacian_parameters": model_confs["GRRCP_no_robust"]["laps_params"]
            },
        },
        {
            "class": MyGRTuckerDecomp,
            "kwargs": {
                "local_threshold": 0,
                "laplacian_parameters": model_confs["GRRTucker_no_robust"][
                    "laps_params"
                ],
            },
        },
        {
            "class": MyGRTuckerDecomp,
            "kwargs": {
                "laplacian_parameters": model_confs["GRRTucker_no_robust"][
                    "laps_params"
                ],
            },
        },
        {"class": MyTuckerTenDecomp, "kwargs": {}},
        {"class": MyRHOOITenDecomp, "kwargs": {}},
        {"class": MyCPTenDecomp, "kwargs": {}},
        {"class": MyRCPTenDecomp, "kwargs": {}},  # Robust CP
    ]
    return model_specs


def plotting(anomaly_type: str):
    # Updated to load 'amplitude' results
    with open(f"figures/results_ampf_sensitivity_{anomaly_type}.json", "r") as f:
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

    anomaly_name = "low-rank" if anomaly_type == "ddos" else anomaly_type
    ax1.set_ylabel("Average PR AUC")
    ax1.set_title(f"Amplitude Sensitivity Tucker ({anomaly_name})")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Amplitude Factor")
    ax2.set_ylabel("Average PR AUC")
    ax2.set_title(f"Amplitude Sensitivity CP ({anomaly_name})")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


plotting("ddos")
exit()


# --- Configuration ---
ampf_to_test = [2, 4, 6, 8, 10, 12]
fixed_rank = 20  # Rank is now held constant
n_runs = 3

dataset_fetchers = {
    "ddos": create_ddos_dataset,
    "spikes": create_spike_dataset,
    "events": create_event_dataset,
    "outage": create_outage_dataset,
}

anomaly_types = ["spikes", "events", "ddos"]

for anomaly_type in anomaly_types:
    print(f"Starting Amplitude experiment for: {anomaly_type}")

    results = {}
    results["_meta"] = {
        "anomaly_type": anomaly_type,
        "n_runs": n_runs,
        "ampf_range": ampf_to_test,
        "fixed_rank": fixed_rank,
    }

    model_specs = model_specs_gen(anomaly_type)

    for spec in model_specs:
        model_name = spec["class"].name()
        if spec["kwargs"].get("local_threshold") != 0:
            model_name += "-thresholded"

        print(f"  Running model: {model_name}")

        results[model_name] = {"x": [], "mean_auc": [], "std_auc": []}

        for a in ampf_to_test:
            aucs = []
            for i in range(n_runs):
                # Initialize model with FIXED rank
                model = spec["class"](rank=fixed_rank, tol=1e-4, **spec["kwargs"])

                # Pass the variable ampf (a) to the dataset generator
                # Note: Assuming your fetcher signature is (split, ampf=...)
                T, L, _, _ = dataset_fetchers[anomaly_type](train_test="test", ampf=a)

                T_hat = model.fit_transform(T, L)
                resids = np.abs(T - T_hat)

                precision, recall, _ = precision_recall_curve(L.ravel(), resids.ravel())
                pr_auc = auc(recall, precision)
                aucs.append(pr_auc)

            mean_auc = float(np.mean(aucs))
            std_auc = float(np.std(aucs))

            results[model_name]["x"].append(a)
            results[model_name]["mean_auc"].append(mean_auc)
            results[model_name]["std_auc"].append(std_auc)

            print(f"    Ampf: {a} | Mean AUC: {mean_auc:.4f}")

    # Updated file name to reflect amplitude factor
    output_path = Path(f"figures/results_ampf_sensitivity_{anomaly_type}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Successfully saved {anomaly_type} to {output_path}\n")
