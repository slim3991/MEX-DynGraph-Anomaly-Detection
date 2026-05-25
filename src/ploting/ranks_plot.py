from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Protocol, Tuple
import json
import os
import numpy as np
import numpy.typing as npt
import time
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


def plot_runtime_vs_rank(anomaly_type: str):
    with open(f"figures/results_rank_sensitivity_{anomaly_type}.json", "r") as f:
        results = json.load(f)

    # Create subplots to match the AUC plotting style
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)

    for model_name, data in results.items():
        if model_name.startswith("_"):
            continue

        plot_params = {
            "x": data["x"],
            "y": data["mean_time"],
            "yerr": data["std_time"],
            "label": model_name,
            "marker": "o",
            "linestyle": "--",
            "capsize": 5,
            "alpha": 0.8,
        }

        # Split logic: Tucker-based vs CP-based
        if "Tucker" in model_name or "RHOOI" in model_name:
            ax1.errorbar(**plot_params)
        else:
            ax2.errorbar(**plot_params)

    anomaly_name = "low-rank" if anomaly_type == "ddos" else anomaly_type

    # Formatting Top Plot (Tucker)
    ax1.set_ylabel("Runtime (seconds)")
    ax1.set_title(f"Runtime Sensitivity Tucker ({anomaly_name})")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)
    # ax1.set_yscale("log")  # Useful for runtime variance

    # Formatting Bottom Plot (CP)
    ax2.set_xlabel("Tensor Rank")
    ax2.set_ylabel("Runtime (seconds)")
    ax2.set_title(f"Runtime Sensitivity CP ({anomaly_name})")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)
    # ax2.set_yscale("log")

    plt.tight_layout()
    plt.show()


def plot_efficiency_frontier(anomaly_type: str, save=False):
    with open(f"figures/results_rank_sensitivity_{anomaly_type}.json", "r") as f:
        results = json.load(f)

    plt.figure(figsize=(14, 7))

    for model_name, data in results.items():
        if model_name.startswith("_"):
            continue

        # We plot time on X and AUC on Y
        # Since 'x' (Rank) varies, we get a trajectory for each model
        plt.plot(
            data["mean_time"],
            data["mean_auc"],
            label=model_name,
            marker="o",
            linestyle="-",
            alpha=0.7,
        )

        # Optional: Label the start and end ranks to show direction
        plt.text(
            data["mean_time"][0], data["mean_auc"][0], f"R={data['x'][0]}", fontsize=8
        )
        plt.text(
            data["mean_time"][-1],
            data["mean_auc"][-1],
            f"R={data['x'][-1]}",
            fontsize=8,
        )

    anomaly_name = "low-rank" if anomaly_type == "ddos" else anomaly_type

    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Average PR AUC")
    plt.title(f"Efficiency Frontier: Runtime vs Performance ({anomaly_name})")

    plt.xscale("log")  # Highly recommended for runtime vs performance
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if save:
        path = os.path.expanduser(f"~/Desktop/pareto_frontier_{anomaly_type}.png")

        # bbox_inches="tight" ensures the legend isn't cropped out
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {path}")
    else:
        plt.show()


def plotting(anomaly_type: str, save=False):

    with open(f"figures/results_rank_sensitivity_{anomaly_type}.json", "r") as f:
        results = json.load(f)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True, sharey=True)
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
    ax1.set_title(f"Rank Sensitivity Tucker ({anomaly_name})")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Tensor Rank")
    ax2.set_ylabel("Average PR AUC")
    ax2.set_title(f"Rank Sensitivity CP ({anomaly_name})")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(fig, f"~/Desktop/rank_sensitivity_{anomaly_type}.png")
    else:
        plt.show()


plot_runtime_vs_rank("spikes")
plot_runtime_vs_rank("events")
plot_runtime_vs_rank("ddos")
exit()
# plot_efficiency_frontier("spikes", True)
# plot_efficiency_frontier("events", True)
# plot_efficiency_frontier("ddos", True)
plotting("spikes")
plotting("events")
plotting("ddos")
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


anomaly_types = "spikes", "events", "ddos"
for anomaly_type in anomaly_types:
    print(f"Starting experiment for: {anomaly_type}")

    results = {}
    results["_meta"] = {
        "anomaly_type": anomaly_type,
        "n_runs": n_runs,
        "ranks": ranks_to_test,
    }

    model_specs = model_specs_gen(anomaly_type)
    for spec in model_specs:
        # Determine base name
        model_name = spec["class"].name()

        # Apply suffix logic
        if spec["kwargs"].get("local_threshold") != 0:
            model_name += "-thresholded"

        print(f"  Running model: {model_name}")

        results[model_name] = {
            "x": [],
            "mean_auc": [],
            "std_auc": [],
            "mean_time": [],
            "std_time": [],
        }

        for r in ranks_to_test:
            aucs = []
            times = []

            for i in range(n_runs):
                model = spec["class"](rank=r, tol=1e-4, **spec["kwargs"])

                T, L, _, _ = dataset_fetchers[anomaly_type]("test")

                start = time.perf_counter()
                T_hat = model.fit_transform(T, L)
                elapsed = time.perf_counter() - start

                times.append(elapsed)

                resids = np.abs(T - T_hat)
                precision, recall, _ = precision_recall_curve(L.ravel(), resids.ravel())
                pr_auc = auc(recall, precision)
                aucs.append(pr_auc)

            mean_auc = float(np.mean(aucs))
            std_auc = float(np.std(aucs))

            mean_time = float(np.mean(times))
            std_time = float(np.std(times))

            results[model_name]["x"].append(r)
            results[model_name]["mean_auc"].append(mean_auc)
            results[model_name]["std_auc"].append(std_auc)

            # NEW
            results[model_name].setdefault("mean_time", []).append(mean_time)
            results[model_name].setdefault("std_time", []).append(std_time)

            print(f"    Rank: {r} | AUC: {mean_auc:.4f} | Time: {mean_time:.3f}s")

    # Save inside the anomaly_type loop
    output_path = Path(f"figures/results_rank_sensitivity_{anomaly_type}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure folder exists

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Successfully saved {anomaly_type} to {output_path}\n")
