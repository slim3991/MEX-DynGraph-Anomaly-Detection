import numpy as np
import yaml
import matplotlib.pyplot as plt
import time
from sklearn.metrics import precision_recall_curve, auc

from models.BasicCP import MyCPTenDecomp
from models.BasicTucker import MyTuckerTenDecomp
from models.GRTenDecomp import MyGRTenDecomp
from models.GRTucker import MyGRTuckerDecomp
from models.RHOOI_model import MyRHOOITenDecomp
from models.RobustCp import MyRCPTenDecomp
from utils.anomaly_injector import inject_random_shapes, inject_random_spikes_normal
from utils.datasets import create_event_dataset_train, get_train_dataset


def create_spike_dataset_train(ampf):
    T, data_param = get_train_dataset()
    n_spikes = 1000
    amplitude_factor = ampf

    T, L = inject_random_spikes_normal(
        T, amplitude_factor=amplitude_factor, n_spikes=n_spikes
    )
    params = {"amplitude_factor": amplitude_factor, "n_spikes": n_spikes}

    return T, L, None, params | data_param


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
plt.figure(figsize=(10, 7))

colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

for idx, model in enumerate(models):
    aucs = []
    times = []
    model.tol = 1e-4

    for i in range(3):
        print(f"{model.name} | run={i}")

        T, L, _, _ = create_event_dataset_train()

        start = time.time()
        T_hat = model.fit_transform(T, L)
        elapsed = time.time() - start

        resids = T - T_hat

        precision, recall, _ = precision_recall_curve(L.ravel(), resids.ravel())
        pr_auc = auc(recall, precision)

        aucs.append(pr_auc)
        times.append(elapsed)

    mean_time = np.mean(times)
    mean_auc = np.mean(aucs)

    plt.scatter(
        mean_time,
        mean_auc,
        s=120,  # bigger points
        color=colors[idx],
        edgecolor="black",
        linewidth=0.8,
        label=model.name,
    )

    # nice offset label (prevents overlap)
    plt.annotate(
        model.name,
        (mean_time, mean_auc),
        textcoords="offset points",
        xytext=(8, 6),
        ha="left",
        fontsize=9,
    )

plt.xlabel("Average Time (seconds)")
plt.ylabel("Average PR AUC")
plt.title("Model Performance vs Runtime")
plt.grid(True, alpha=0.3)

# adds breathing room so points don't sit on edges
plt.margins(0.15)

plt.tight_layout()
plt.show()

try:
    plt.savefig("./figures/time_vs_pr_auc.png", bbox_inches="tight")
except FileNotFoundError:
    print("Could not save figure (folder missing)")

plt.show()
