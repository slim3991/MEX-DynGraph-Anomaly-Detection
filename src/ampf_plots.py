import numpy as np
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

# Range of amplitude factors to test
amplitude_factors = [1, 3, 5, 7, 9, 12]

plt.figure(figsize=(8, 6))

for model in models:
    mean_aucs = []
    model.tol = 1e-4

    for ampf in amplitude_factors:
        aucs = []

        for i in range(3):
            print(f"{model.name} | ampf={ampf} | run={i}")

            T, L, _, _ = create_spike_dataset_train(ampf)
            T_hat = model.fit_transform(T, L)
            resids = T - T_hat

            precision, recall, _ = precision_recall_curve(L.ravel(), resids.ravel())

            # Sort recall before AUC
            recall, precision = zip(*sorted(zip(recall, precision)))

            pr_auc = auc(recall, precision)
            aucs.append(pr_auc)

        # Average AUC for this amplitude
        mean_aucs.append(np.mean(aucs))

    # Plot AUC vs amplitude
    plt.plot(amplitude_factors, mean_aucs, marker="o", label=model.name)

# Final touches
plt.xlabel("Amplitude Factor")
plt.ylabel("Average PR AUC")
plt.title("Effect of Amplitude on PR AUC")
plt.legend()
plt.grid()
try:
    plt.savefig("./figures/ampfEffects.png")
except FileNotFoundError as e:
    print("path not found")


plt.show()
