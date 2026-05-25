from itertools import product
from matplotlib import colors
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
    create_spike_dataset,
)
from utils.metrics import compute_metrics_with_optimal_threshold
from utils.tensor_processing import (
    de_anomalize_tensor,
    make_mode_knn,
    make_mode_laplacian,
    normalize_tensor,
    preprocess,
)


Tp = np.load("data/abiline_ten.npy")
Tp = Tp[:, :, :4500]

# T = T[:, :, 10_000:15_000]
for i, j in product(range(12), repeat=2):
    Tp[i, j, :] = normalize_tensor(Tp[i, j, :], "minmax")
# T = normalize_tensor(T, "minmax")
T = de_anomalize_tensor(Tp, 20)

# T = normalize_tensor(T, "minmax")
source, dest = np.random.randint(0, 11), np.random.randint(0, 11)
# source, dest = 5, 8

# T, L, _, _ = create_event_dataset_train()

# ddos injection
L = np.zeros_like(T)
for _ in range(100):
    a = np.random.randint(0, 12)
    T, Lp = inject_DDoS(T, duration=10, n_senders=7, target=a, amplitude_factor=10)
    L += Lp
L = np.where(L > 0, 1, 0)

# #outage injection
# L = np.zeros_like(T)
# for _ in range(70):
#     T, Lp = inject_outage(T, duration=12 * 2, n_nodes=1)
#     L += Lp
# L = np.where(L > 0, 1, 0)

# lap_parms = {
#     "lambda_1": 5,
#     "lambda_2": 0.0007,
#     "lambda_smooth": 2900,
#     "lambda_interval": 8300,
#     "measure": "dot",
#     "ks_1": 0,
#     "ks_2": 5,
# }
# tucker_lap_params = {
#     "lambda_1": 5,
#     "lambda_2": 0.0007,
#     "lambda_smooth": 29,
#     "lambda_interval": 100,
#     "measure": "dot",
#     "ks_1": 8,
#     "ks_2": 8,
# }
# X_hat_basic = tl.tucker_to_tensor(
#     tl.decomposition.tucker(T, rank=(10, 10, 10), tol=1e-4, init="random")
# )


def plot_robust_tucker():
    models = [MyRHOOITenDecomp(rank=20, tol=1e-4), MyRCPTenDecomp(rank=20, tol=1e-4)]

    # Create a vertical stack of subplots
    fig, axes = plt.subplots(
        len(models), 1, figsize=(10, 3 * len(models)), sharex=True, sharey=True
    )
    fig.suptitle("Soft-Threshold Based Models", fontsize=16)

    T, L, _, _ = create_ddos_dataset(train_test="test")

    basic_models = [
        lambda t: tl.cp_to_tensor(
            tl.decomposition.parafac(t, rank=20, tol=1e-4, init="random")
        ),
        lambda t: tl.tucker_to_tensor(
            tl.decomposition.tucker(t, rank=(20, 20, 20), init="random", tol=1e-4)
        ),
    ]

    for i, model in enumerate(models):
        print(f"model : {model.name()} ...")

        X_hat = model.fit_transform(T, L)

        # Plot on the specific subplot axis
        ax = axes[i]
        ax.plot(T[source, dest, :], "--", alpha=0.5, label="original")
        ax.plot(basic_models[i](T)[source, dest, :], label="Basic")
        ax.plot(X_hat[source, dest, :], label=f"{model.name()}")

        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.xlabel("time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
    plt.show()


def plot_tucker_smooth(basic_params):
    params = 10 ** np.array((1, 2, 3, 4))
    model = MyGRTuckerDecomp(
        rank=10, laplacian_parameters=basic_params, tol=1e-4, threshold=None
    )
    model.laplacian_parameters["lambda_interval"] = 0

    # Create a vertical stack of subplots
    fig, axes = plt.subplots(
        len(params), 1, figsize=(10, 3 * len(params)), sharex=True, sharey=True
    )
    fig.suptitle("Effects of the Smoothing Laplacian", fontsize=16)

    for i, param in enumerate(params):
        print(f"param value: {param} ...")
        model.laplacian_parameters["lambda_smooth"] = param
        X_hat = model.fit_transform(T, L)

        # Plot on the specific subplot axis
        ax = axes[i]
        ax.plot(T[source, dest, :], "--", alpha=0.5, label="original")
        ax.plot(X_hat[source, dest, :], label=f"lambda = {param}", color="tab:orange")

        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.xlabel("time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
    plt.show()


def plot_tucker_interval(basic_params):
    params = 10 ** np.array((0, 1, 2, 3))
    model = MyGRTuckerDecomp(
        rank=10, laplacian_parameters=basic_params, tol=1e-4, local_threshold=None
    )
    model.laplacian_parameters["lambda_smooth"] = 0

    # Create a vertical stack of subplots
    fig, axes = plt.subplots(
        len(params), 1, figsize=(10, 3 * len(params)), sharex=True, sharey=True
    )
    fig.suptitle("Effects of the Diurnal Laplacian", fontsize=16)

    for i, param in enumerate(params):
        print(f"param value: {param} ...")
        model.laplacian_parameters["lambda_interval"] = param
        X_hat = model.fit_transform(T, L)

        # Plot on the specific subplot axis
        ax = axes[i]
        ax.plot(T[source, dest, :], "--", alpha=0.5, label="original")
        ax.plot(
            X_hat[source, dest, :],
            label=f"lambda = {param}",
            color="tab:green",
        )

        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.xlabel("time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def ks_laplacians(mode):
    L = make_mode_laplacian(
        tensor=T[:, :, :300], mode=mode, k=10, measure="angular", sparse=False
    )
    plt.figure(figsize=(8, 6))
    plt.imshow(
        L,
        cmap="RdBu_r",
        # norm=colors.SymLogNorm(linthresh=0.01, linscale=1, vmin=-1, vmax=5),
        norm=colors.PowerNorm(gamma=0.5),
    )
    plt.colorbar(label="Weight")
    plt.title("knn mode 1 laplacian\n(Normalized)")
    plt.show()


def plot_tucker_ks2():
    params = 10.0 ** np.array((0, 1, 2, 3))
    tucker_lap_params = {
        "lambda_1": 0,
        "lambda_2": 0,
        "lambda_3": 0,
        "lambda_smooth": 0,
        "lambda_interval": 0,
        "measure": "euclidean",
        "ks_1": 8,
        "ks_2": 8,
        "ks_3": 10,
    }
    model = MyGRTuckerDecomp(
        rank=10, laplacian_parameters=tucker_lap_params, tol=1e-4, local_threshold=0
    )
    # Create a vertical stack of subplots
    fig, axes = plt.subplots(
        len(params), 1, figsize=(10, 3 * len(params)), sharex=True, sharey=True
    )
    fig.suptitle("Effects of the K-NN (mode 3) laplacian", fontsize=16)

    for i, param in enumerate(params):
        print(f"param value: {param} ...")
        model.laplacian_parameters["lambda_3"] = param
        X_hat = model.fit_transform(T, L)

        # Plot on the specific subplot axis
        ax = axes[i]
        ax.plot(T[source, dest, :], "--", alpha=0.5, label="original")
        ax.plot(
            X_hat[source, dest, :],
            label=f"lambda = {param}",
            color="tab:green",
        )

        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.xlabel("time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# plot_tucker_interval(tucker_lap_params)
plot_tucker_smooth()
# plot_tucker_jr(tucker_lap_params)
# plot_tucker_ks2()
# plot_robust_tucker()
# ks_laplacians(mode=2)
