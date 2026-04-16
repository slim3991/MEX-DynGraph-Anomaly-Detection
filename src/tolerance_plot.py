import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

from models.BasicCP import MyCPTenDecomp
from models.BasicTucker import MyTuckerTenDecomp
from models.GRTenDecomp import MyGRTenDecomp
from models.GRTucker import MyGRTuckerDecomp
from models.RHOOI_model import MyRHOOITenDecomp
from models.RobustCp import MyRCPTenDecomp
from utils.anomaly_injector import inject_random_shapes, inject_random_spikes_normal
from utils.datasets import get_train_dataset


def create_spike_dataset_train(ampf):
    T, data_param = get_train_dataset()
    n_spikes = 1000
    amplitude_factor = ampf

    T, L = inject_random_spikes_normal(
        T, amplitude_factor=amplitude_factor, n_spikes=n_spikes
    )
    params = {"amplitude_factor": amplitude_factor, "n_spikes": n_spikes}

    return T, L, None, params | data_param


def create_event_dataset_train(ampf):
    T, data_param = get_train_dataset()

    params = {
        "start_min": 20,
        "start_max": 4000,
        "min_duration": 10,
        "max_duration": 100,
        "n_shapes": 20,
        "amplitude_factor": ampf,
    }
    T, L, events = inject_random_shapes(T, **params)

    return T, L, events, params | data_param


with open("src/model_config.yaml") as f:
    m_conf = yaml.safe_load(f)
model_confs = m_conf["spikes_parameters"]
models = [
    # MyCPTenDecomp(**model_confs["basic_cp"]),
    MyTuckerTenDecomp(**model_confs["basic_tucker"]),
    # MyRCPTenDecomp(**model_confs["robust_cp"]),
    MyRHOOITenDecomp(**model_confs["robust_tucker"]),
    # MyGRTenDecomp(**model_confs["GRRCP"]),
    # MyGRTenDecomp(**model_confs["GRRCP_no_robust"]),
    MyGRTuckerDecomp(**model_confs["GRRTucker"]),
    # MyGRTuckerDecomp(**model_confs["GRRTucker_no_robust"]),
]

n_runs = 5
tols = [1e-1, 1e-3, 1e-5, 1e-7, 1e-9]

plt.figure(figsize=(10, 7))

for model in models:
    mean_aucs = []
    std_aucs = []

    for tol in tols:
        run_aucs = []
        model.tol = tol

        for i in range(n_runs):
            print(f"Model: {model.name} | Tol: {tol} | Run: {i+1}/{n_runs}")

            # Generate dataset
            T, L, _, _ = create_spike_dataset_train(6)

            # Decompose
            T_hat = model.fit_transform(T, L)
            resids = np.abs(T - T_hat)  # Use absolute residuals for anomaly detection

            # PR AUC Calculation
            precision, recall, _ = precision_recall_curve(L.ravel(), resids.ravel())

            # Ensure sorting for AUC calculation
            sorted_indices = np.argsort(recall)
            pr_auc = auc(recall[sorted_indices], precision[sorted_indices])
            run_aucs.append(pr_auc)

        # Statistics for the current tolerance level
        mean_aucs.append(np.mean(run_aucs))
        std_aucs.append(np.std(run_aucs))

    # Plotting with Error Bars
    plt.errorbar(
        tols,
        mean_aucs,
        yerr=std_aucs,
        label=model.name,
        marker="o",
        capsize=5,  # Adds horizontal caps to the error bars
        linestyle="-",  # Connects the dots
        alpha=0.8,  # Slight transparency to see overlapping lines
    )

# Formatting the plot
plt.xscale("log")  # Explicitly set log scale for the x-axis
plt.xlabel("Tolerance (log scale)")
plt.ylabel("Average PR AUC")
plt.title("Effect of Tolerance on PR AUC (Mean ± SD)")
plt.legend(
    bbox_to_anchor=(1.05, 1), loc="upper left"
)  # Move legend outside if it overlaps
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.tight_layout()

try:
    plt.savefig("./figures/tolEffects.png")
except FileNotFoundError:
    print("Path not found, displaying plot instead.")

plt.show()
