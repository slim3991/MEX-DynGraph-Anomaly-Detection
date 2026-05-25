from typing import Tuple
import yaml
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
