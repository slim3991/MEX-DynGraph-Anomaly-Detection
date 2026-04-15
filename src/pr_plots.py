from typing import Tuple
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

from models.BasicCP import MyCPTenDecomp
from models.BasicTucker import MyTuckerTenDecomp
from models.GRTenDecomp import MyGRTenDecomp
from models.GRTucker import MyGRTuckerDecomp
from models.RHOOI_model import MyRHOOITenDecomp
from models.RobustCp import MyRCPTenDecomp
from utils.anomaly_injector import inject_random_spikes_normal
from utils.datasets import (
    get_train_dataset,
)


def create_spike_dataset_train(ampf) -> Tuple[npt.NDArray, npt.NDArray, None, dict]:
    T, data_param = get_train_dataset()
    n_spikes = 1000
    amplitude_factor = ampf

    T, L = inject_random_spikes_normal(
        T, amplitude_factor=amplitude_factor, n_spikes=n_spikes
    )
    params = {"amplitude_factor": amplitude_factor, "n_spikes": n_spikes}

    return T, L, None, params | data_param


models = [
    MyCPTenDecomp(
        rank=14,
        threshold=0.7,
    ),
    MyTuckerTenDecomp(
        ranks=(9, 10, 4),
        threshold=0.42,
    ),
    MyRCPTenDecomp(
        rank=14,
        local_threshold=1.3,
        threshold=0.8,
    ),
    MyRHOOITenDecomp(
        ranks=(7, 9, 6),
        local_threshold=0.6,
        threshold=0.98,
    ),
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


mean_recall = np.linspace(0, 1, 100)

plt.figure(figsize=(8, 6))


for model in models:
    precisions = []

    for i in range(3):
        print(i)
        T, L, _, _ = create_spike_dataset_train(7)
        T_hat = model.fit_transform(T, L)
        resids = T - T_hat

        precision, recall, _ = precision_recall_curve(L.ravel(), resids.ravel())

        # Sort recall (just in case)
        recall, precision = zip(*sorted(zip(recall, precision)))

        # Interpolate precision onto common recall axis
        interp_precision = np.interp(mean_recall, recall, precision)
        precisions.append(interp_precision)

    # Average precision across runs
    mean_precision = np.mean(precisions, axis=0)

    # Plot
    plt.plot(mean_recall, mean_precision, label=model.name)

# Final touches
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Average Precision-Recall Curve")
plt.legend()
plt.grid()

plt.show()
