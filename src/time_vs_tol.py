import numpy as np
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


models = [
    MyCPTenDecomp(rank=14, threshold=0.7),
    MyTuckerTenDecomp(ranks=(9, 10, 4), threshold=0.42),
    MyRCPTenDecomp(rank=14, local_threshold=1.3, threshold=0.8),
    MyRHOOITenDecomp(ranks=(7, 9, 6), local_threshold=0.6, threshold=0.98),
    MyGRTenDecomp(
        rank=20,
        lambdas=(46, 0.001, 0.04),
        ks=(8, 5, 4),
        measure="euclidean",
        local_threshold=2.9,
        threshold=0.6,
    ),
    MyGRTuckerDecomp(
        rank=(12, 17, 20),
        lambdas=(0.0096, 0.56, 0.00049),
        ks=(0, 1, 1),
        measure="euclidean",
        local_threshold=2.6,
        threshold=0.6,
    ),
]

plt.figure(figsize=(10, 7))

colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

for idx, model in enumerate(models):
    aucs = []
    times = []
    model.tol = 1e-4

    for i in range(3):
        print(f"{model.name} | run={i}")

        T, L, _, _ = create_spike_dataset_train(6)

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
