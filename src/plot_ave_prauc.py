from typing import Tuple, List
import yaml
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

from models.BasicCP import MyCPTenDecomp
from models.BasicTucker import MyTuckerTenDecomp
from models.GRTenDecomp import MyGRTenDecomp
from models.GRTucker import MyGRTuckerDecomp
from models.RHOOI_model import MyRHOOITenDecomp
from models.RobustCp import MyRCPTenDecomp
from utils.anomaly_injector import inject_random_spikes_normal
from utils.datasets import create_event_dataset_train, get_train_dataset


def create_spike_dataset_train(ampf) -> Tuple[npt.NDArray, npt.NDArray, None, dict]:
    T, data_param = get_train_dataset()
    n_spikes = 1000
    amplitude_factor = ampf

    T, L = inject_random_spikes_normal(
        T, amplitude_factor=amplitude_factor, n_spikes=n_spikes
    )
    params = {"amplitude_factor": amplitude_factor, "n_spikes": n_spikes}

    return T, L, None, params | data_param


# Load configuration
with open("src/model_config.yaml") as f:
    m_conf = yaml.safe_load(f)
model_confs = m_conf["events_parameters"]

models = [
    MyCPTenDecomp(**model_confs["basic_cp"]),
    MyTuckerTenDecomp(**model_confs["basic_tucker"]),
    MyRCPTenDecomp(**model_confs["robust_cp"]),
    MyRHOOITenDecomp(**model_confs["robust_tucker"]),
    MyGRTenDecomp(**model_confs["GRRCP"]),
    MyGRTenDecomp(**model_confs["GRRCP_no_robust"]),
    MyGRTuckerDecomp(**model_confs["GRRTucker"]),
    MyGRTuckerDecomp(**model_confs["GRRTucker_no_robust"]),
]

# Containers for results
model_names = []
avg_aucs = []
std_aucs = []

print("Starting evaluation...")

for model in models:
    model_names.append(model.name)
    model.tol = 1e-4
    current_model_aucs = []

    # Run multiple iterations to get an average
    for i in range(3):
        print(f"Evaluating {model.name} - Iteration {i+1}")
        T, L, _, _ = create_event_dataset_train()
        T_hat = model.fit_transform(T, L)
        resids = T - T_hat

        # Calculate PR Curve
        precision, recall, _ = precision_recall_curve(L.ravel(), resids.ravel())

        pr_auc = auc(recall, precision)
        current_model_aucs.append(pr_auc)

    avg_aucs.append(np.mean(current_model_aucs))
    std_aucs.append(np.std(current_model_aucs))

# --- Plotting Section ---
plt.figure(figsize=(12, 7))

bars = plt.bar(
    model_names,
    avg_aucs,
    yerr=std_aucs,
    capsize=5,
    color="skyblue",
    edgecolor="navy",
    alpha=0.8,
)

# Formatting
plt.ylabel("Average PR-AUC", fontweight="bold")
plt.title("Model Performance Comparison: Average PR-AUC", fontsize=14)
plt.xticks(rotation=45, ha="right")
plt.ylim(0, 1.1)
plt.grid(axis="y", linestyle="--", alpha=0.7)

for bar, ste in zip(bars, std_aucs):
    yval = bar.get_height()

    plt.text(
        bar.get_x() + bar.get_width() / 2,
        yval + ste + 0.01,
        f"{yval:.3f}",
        ha="center",
        va="bottom",
        fontsize=10,
    )

plt.tight_layout()
plt.show()
