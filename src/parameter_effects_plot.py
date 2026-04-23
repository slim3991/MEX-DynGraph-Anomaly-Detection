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
from models.RobustCp import MyRCPTenDecomp
from utils.anomaly_injector import (
    inject_DDoS,
    inject_outage,
    inject_random_spikes_normal,
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

lap_parms = {
    "lambda_1": 5,
    "lambda_2": 0.0007,
    "lambda_smooth": 2900,
    "lambda_interval": 8300,
    "measure": "dot",
    "ks_1": 0,
    "ks_2": 5,
}
tucker_lap_params = {
    "lambda_1": 5,
    "lambda_2": 0.0007,
    "lambda_smooth": 29,
    "lambda_interval": 100,
    "measure": "dot",
    "ks_1": 8,
    "ks_2": 8,
}
X_hat_basic = tl.tucker_to_tensor(
    tl.decomposition.tucker(T, rank=(10, 10, 10), tol=1e-4, init="random")
)


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
        ax.plot(
            X_hat[source, dest, :], label=f"lambda smooth = {param}", color="tab:orange"
        )

        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.xlabel("time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for suptitle
    plt.show()


def plot_tucker_interval(basic_params):
    params = 10 ** np.array((0, 1, 2, 3))
    model = MyGRTuckerDecomp(
        rank=10, laplacian_parameters=basic_params, tol=1e-4, threshold=None
    )
    model.laplacian_parameters["lambda_smooth"] = 0

    # Create a vertical stack of subplots
    fig, axes = plt.subplots(
        len(params), 1, figsize=(10, 3 * len(params)), sharex=True, sharey=True
    )
    fig.suptitle("Effects of the Interval Laplacian", fontsize=16)

    for i, param in enumerate(params):
        print(f"param value: {param} ...")
        model.laplacian_parameters["lambda_interval"] = param
        X_hat = model.fit_transform(T, L)

        # Plot on the specific subplot axis
        ax = axes[i]
        ax.plot(T[source, dest, :], "--", alpha=0.5, label="original")
        ax.plot(
            X_hat[source, dest, :],
            label=f"lambda interval = {param}",
            color="tab:green",
        )

        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.xlabel("time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_tucker_ks2(basic_params):
    params = 10.0 ** np.array((-2, -1, 0, 1))
    model = MyGRTuckerDecomp(
        rank=10, laplacian_parameters=basic_params, tol=1e-4, threshold=None
    )
    model.laplacian_parameters["lambda_smooth"] = 0
    model.laplacian_parameters["lambda_interval"] = 0
    model.laplacian_parameters["lambda_2"] = 0
    model.laplacian_parameters["measure"] = "angular"

    # Create a vertical stack of subplots
    fig, axes = plt.subplots(
        len(params), 1, figsize=(10, 3 * len(params)), sharex=True, sharey=True
    )
    fig.suptitle("Effects of the Interval Laplacian", fontsize=16)

    for i, param in enumerate(params):
        print(f"param value: {param} ...")
        model.laplacian_parameters["lambda_1"] = param
        X_hat = model.fit_transform(T, L)

        # Plot on the specific subplot axis
        ax = axes[i]
        ax.plot(T[source, dest, :], "--", alpha=0.5, label="original")
        ax.plot(
            X_hat[source, dest, :],
            label=f"lambda interval = {param}",
            color="tab:green",
        )

        ax.set_ylabel("Value")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.xlabel("time")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def ks_laplacians(basic_params):
    L = basic_params["lambda_1"] * make_mode_laplacian(
        T, mode=0, k=basic_params["lambda_1"], measure=basic_params["measure"]
    )
    plt.figure(figsize=(8, 6))
    plt.imshow(
        L.toarray(),
        cmap="RdBu_r",
        # norm=colors.SymLogNorm(linthresh=0.01, linscale=1, vmin=-1, vmax=5),
        norm=colors.PowerNorm(gamma=0.5),
    )
    plt.colorbar(label="Weight")
    plt.title("knn mode 1 laplacian\n(Normalized)")
    plt.show()


# plot_tucker_interval(tucker_lap_params)
plot_tucker_smooth(tucker_lap_params)
# plot_tucker_ks2(tucker_lap_params)
# ks_laplacians(tucker_lap_params)
