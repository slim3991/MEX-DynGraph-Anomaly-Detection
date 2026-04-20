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
from utils.anomaly_injector import (
    inject_DDoS,
    inject_random_shapes,
    inject_random_spikes_normal,
)
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


def create_ddos_dataset_train(ampf):
    T, data_param = get_train_dataset()
    n_spikes = 1000
    amplitude_factor = 6

    L = np.zeros_like(T)
    for _ in range(100):
        a = np.random.randint(0, 12)
        T, Lp = inject_DDoS(
            T, duration=10, n_senders=7, amplitude_factor=ampf, target=a
        )
        L += Lp
    L = np.where(L > 0, 1, 0)
    params = {"amplitude_factor": amplitude_factor, "n_spikes": n_spikes}
    return T, L, None, params | data_param


with open("src/model_config.yaml") as f:
    m_conf = yaml.safe_load(f)
model_confs = m_conf["ddos_parameters"]
models = [
    MyGRTenDecomp(**model_confs["GRRCP"]),
    MyGRTenDecomp(**model_confs["GRRCP_no_robust"]),
    MyCPTenDecomp(**model_confs["basic_cp"]),
    MyRCPTenDecomp(**model_confs["robust_cp"]),
    # MyTuckerTenDecomp(**model_confs["basic_tucker"]),
    # MyRHOOITenDecomp(**model_confs["robust_tucker"]),
    # MyGRTuckerDecomp(**model_confs["GRRTucker"]),
    # MyGRTuckerDecomp(**model_confs["GRRTucker_no_robust"]),
]

# Range of amplitude factors to test
amplitude_factors = [6, 7, 8, 9, 10, 11]

plt.figure(figsize=(8, 6))

for model in models:
    mean_aucs = []
    model.tol = 1e-3

    for ampf in amplitude_factors:
        aucs = []

        for i in range(2):
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
        print("aucs: ", aucs)

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
