from itertools import product
import yaml
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve
from sklearn.preprocessing import normalize
import tensorly as tl
import optuna

from models.GRTenDecomp import MyGRTenDecomp
from models.GRTucker import MyGRTuckerDecomp
from models.RHOOI_model import MyRHOOITenDecomp
from models.RobustCp import MyRCPTenDecomp
from utils.anomaly_injector import (
    inject_DDoS,
    inject_outage,
    inject_random_spikes_normal,
)
from utils.datasets import (
    create_ddos_dataset,
    create_event_dataset,
    create_outage_dataset,
    create_spike_dataset,
)
from utils.metrics import compute_metrics_with_optimal_threshold

#
# source, dest = np.random.randint(0, 11), np.random.randint(0, 11)
source, dest = (5, 8)

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True, sharey=True)


SPIKES_AMPF = 8
EVENTS_AMPF = 6  # ampf <-> amplitude factor
DDOS_AMPF = 7


def fit_by_anomaly_cp(anomaly_type, T, L):
    with open("src/model_config.yaml") as f:
        m_conf = yaml.safe_load(f)
    model_confs = m_conf[f"{anomaly_type}_configs"]
    cp_lap_params = model_confs["GRRCP_no_robust"]["laps_params"]
    rank = 20
    X_hat = MyGRTenDecomp(
        rank=rank,
        laplacian_parameters=cp_lap_params,
        threshold=None,
        tol=1e-4,
    ).fit_transform(T, L)
    X_hat_basic = tl.cp_to_tensor(
        tl.decomposition.parafac(
            T,
            tol=1e-4,
            rank=rank,
            init="random",
        )
    )
    return X_hat, X_hat_basic


def fit_by_anomaly_tucker(anomaly_type, T, L):
    with open("src/model_config.yaml") as f:
        m_conf = yaml.safe_load(f)
    model_confs = m_conf[f"{anomaly_type}_configs"]
    tucker_lap_params = model_confs["GRRTucker_no_robust"]["laps_params"]
    rank = 20
    X_hat = MyGRTuckerDecomp(
        rank=(rank, rank, rank),
        laplacian_parameters=tucker_lap_params,
        tol=1e-4,
        threshold=0,
    ).fit_transform(T, L)
    X_hat_basic = tl.tucker_to_tensor(
        tl.decomposition.tucker(T, rank=(rank, rank, rank), tol=1e-4, init="random")
    )
    return X_hat, X_hat_basic


def plot_fitted_and_anomalies_cp():
    # --- Spikes ---
    T, L, _, _ = create_spike_dataset("train", ampf=SPIKES_AMPF)
    ax1.plot(T[source, dest, :], label="Original with anomalies")

    X_hat, X_hat_basic = fit_by_anomaly_cp("spikes", T, L)
    ax1.plot(X_hat_basic[source, dest, :], label="Basic", alpha=0.8)
    ax1.plot(X_hat[source, dest, :], label="Graph-Regularized", alpha=0.5)

    ax1.set_title("Random Spikes Anomalies")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Events ---
    T, L, _, _ = create_event_dataset("train", EVENTS_AMPF)
    ax2.plot(T[source, dest, :], label="Original with anomalies")

    X_hat, X_hat_basic = fit_by_anomaly_cp("events", T, L)
    ax2.plot(X_hat_basic[source, dest, :], label="Basic", alpha=0.8)
    ax2.plot(X_hat[source, dest, :], label="Graph-Regularized", alpha=0.5)

    ax2.set_title("Random Events Anomalies")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- DDoS ---
    T, L, _, _ = create_ddos_dataset("train", ampf=DDOS_AMPF)
    ax3.plot(T[source, dest, :], label="Original with anomalies")

    X_hat, X_hat_basic = fit_by_anomaly_cp("ddos", T, L)
    ax3.plot(X_hat_basic[source, dest, :], label="Basic", alpha=0.8)
    ax3.plot(X_hat[source, dest, :], label="Graph-Regularized", alpha=0.5)

    ax3.set_title("Random Low-Rank Events Anomalies (DDoS)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.show()


def plot_fitted_and_anomalies_tucker():
    # --- Spikes ---
    T, L, _, _ = create_spike_dataset("train", ampf=SPIKES_AMPF)
    ax1.plot(T[source, dest, :], label="Original with anomalies")

    X_hat, X_hat_basic = fit_by_anomaly_tucker("spikes", T, L)
    ax1.plot(X_hat_basic[source, dest, :], label="Basic", alpha=0.8)
    ax1.plot(X_hat[source, dest, :], label="Graph-Regularized", alpha=0.5)

    ax1.set_title("Random Spikes Anomalies")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Events ---
    T, L, _, _ = create_event_dataset("train", EVENTS_AMPF)
    ax2.plot(T[source, dest, :], label="Original with anomalies")

    X_hat, X_hat_basic = fit_by_anomaly_tucker("events", T, L)
    ax2.plot(X_hat_basic[source, dest, :], label="Basic", alpha=0.8)
    ax2.plot(X_hat[source, dest, :], label="Graph-Regularized", alpha=0.5)

    ax2.set_title("Random Events Anomalies")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # --- DDoS ---
    T, L, _, _ = create_ddos_dataset("train", ampf=DDOS_AMPF)
    ax3.plot(T[source, dest, :], label="Original with anomalies")

    X_hat, X_hat_basic = fit_by_anomaly_tucker("ddos", T, L)
    ax3.plot(X_hat_basic[source, dest, :], label="Basic", alpha=0.8)
    ax3.plot(X_hat[source, dest, :], label="Graph-Regularized", alpha=0.5)

    ax3.set_title("Random Low-Rank Events Anomalies (DDoS)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.show()


def plot_injected_anomalies():
    source, dest = np.random.randint(0, 11), np.random.randint(0, 11)
    T, L, _, _ = create_spike_dataset("train", ampf=SPIKES_AMPF)
    ax1.plot(L[source, dest, :], label="Anomalies", alpha=0.4)
    ax1.plot(T[source, dest, :], label="Data")
    ax1.set_title("Random Spikes Anomalies")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    T, L, _, _ = create_event_dataset("train", ampf=EVENTS_AMPF)
    ax2.plot(L[source, dest, :], label="Anomalies", alpha=0.4)
    ax2.plot(T[source, dest, :], label="Data")
    ax2.set_title("Random events Anomalies")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    T, L, _, _ = create_ddos_dataset("train", ampf=DDOS_AMPF)
    ax3.plot(L[source, dest, :], label="Anomalies", alpha=0.4)
    ax3.plot(T[source, dest, :], label="Data", alpha=0.8)
    ax3.set_title("Random Low-Rank Events Anomalies (DDoS)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.show()


# plot_fitted_and_anomalies_cp()
# plot_fitted_and_anomalies_tucker()
plot_injected_anomalies()


# alkskdjf
