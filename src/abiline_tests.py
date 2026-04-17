from itertools import product
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import normalize
import tensorly as tl
import optuna

from models.GRTenDecomp import MyGRTenDecomp
from models.GRTucker import MyGRTuckerDecomp
from models.RobustCp import MyRCPTenDecomp
from utils.anomaly_injector import inject_DDoS, inject_random_spikes_normal
from utils.datasets import create_event_dataset_train, create_spike_dataset_train
from utils.metrics import compute_metrics_with_optimal_threshold
from utils.tensor_processing import (
    de_anomalize_tensor,
    make_mode_knn,
    make_mode_laplacian,
    normalize_tensor,
    preprocess,
)


T = np.load("data/abiline_ten.npy")
T = T[:, :, :4000]
# T = T[:, :, 10_000:15_000]
for i, j in product(range(12), repeat=2):
    T[i, j, :] = normalize_tensor(T[i, j, :], "minmax")
T = de_anomalize_tensor(T, 20)

# T = normalize_tensor(T, "minmax")
source, dest = np.random.randint(0, 11), np.random.randint(0, 11)
# source, dest = 5, 8

T, L, _, _ = create_event_dataset_train()

# L = np.zeros_like(T)
# for _ in range(100):
#     a = np.random.randint(0, 12)
#     T, Lp = inject_DDoS(T, duration=10, n_senders=5, amplitude_factor=10, target=a)
#     L += Lp
# L = np.where(L > 0, 1, 0)

X_hat = MyGRTenDecomp(
    rank=(5),
    ks=[2, 10, 0],
    lambdas=[0, 0, 1e4],
    local_threshold=0,
    measure="angular",
    tol=1e-4,
).fit_transform(T, L)

X_hat_cp = tl.cp_to_tensor(
    tl.decomposition.parafac(
        T,
        tol=1e-4,
        rank=(5),
        init="random",
    )
)

plt.plot(L[source, dest, :], alpha=0.5)
plt.plot(T[source, dest, :], alpha=0.5)
plt.plot((X_hat_cp[source, dest, :]), label="regular")
plt.plot((X_hat[source, dest, :]), label="graph")
plt.legend()
plt.show()
exit()

# best_score =score: {best_score}, best val {best_val}")
# best (7,9,1,3)
rank = 3

laps = []
# laps.append(make_laplacian(T, mode=0, k=11))
# laps.append(make_laplacian(T, mode=1, k=1))
# laps.append(make_laplacian(T, mode=2, k=300))


def find_laps():
    def objective(trial: optuna.Trial, T):
        k1 = trial.suggest_int("k1", 5, 60)
        k2 = trial.suggest_int("k2", 5, 60)
        k3 = trial.suggest_int("k3", 5, 600)
        rank = trial.suggest_int("rank", 3, 11)
        l1 = trial.suggest_float("l1", -5000, 5000)
        l2 = trial.suggest_float("l2", -5000, 5000)
        l3 = trial.suggest_float("l3", -5000, 5000)

        laps = [
            make_mode_laplacian(T, mode=0, k=k1, measure="euclidean") * l1,
            make_mode_laplacian(T, mode=1, k=k2, measure="euclidean") * l2,
            make_mode_laplacian(T, mode=2, k=k3, measure="euclidean") * l3,
        ]
        print(laps[0].shape, laps[1].shape, laps[2].shape, rank)
        factors = estimate_from_laps(rank=5, laps=laps, mode_shapes=(0, 1, 2))
        x_hat = tl.cp_to_tensor(factors)
        obj = tl.sum((x_hat - T) ** 2)
        return obj

    study = optuna.create_study(direction="minimize", study_name="GRT")
    study.optimize(lambda trial: objective(trial, T), n_trials=20)
    print("Best trial: ", study.best_trial.number, ", with value: ", study.best_value)
    print("Best Params", study.best_params, end="\n\n")


def plot_regualrizaton_tensor():
    laps = []

    laps.append(make_mode_laplacian(T, mode=0, k=8, measure="euclidean") * (46))
    laps.append(make_mode_laplacian(T, mode=1, k=5, measure="euclidean") * (0.001))
    laps.append(make_mode_laplacian(T, mode=2, k=4, measure="euclidean") * (0.04))

    factors = estimate_from_laps(rank=11, laps=laps, mode_shapes=(0, 1, 2))
    x_hat = tl.cp_to_tensor(factors)

    # for i in range(12):
    #     for j in range(12):
    #         x_hat[i, j, :] = normalize_tensor(x_hat[i, j, :], "minmax")

    plt.plot(T[source, dest, :], alpha=0.5, label="Original data")
    plt.plot(x_hat[source, dest, :], label="Pure Regularizer (scaled)")
    plt.title("Regularizers pure effect")
    plt.xlabel("Time")
    plt.legend()
    plt.show()


def decomp_recomp(T: tl.tensor, rank: int):
    init = "random" if rank > 12 else "svd"
    weights, factors = tl.decomposition.parafac2(T, rank=rank, verbose=False, init=init)
    reconst = tl.cp_to_tensor(cp_tensor=(weights, factors))
    return reconst


# find_laps()
plot_regualrizaton_tensor()
exit()
orig_shape = T.shape
# T = np.reshape(T, (12 * 24, 7, -1))
laps = []
laps.append(make_mode_laplacian(T, mode=0, k=3))
laps.append(make_mode_laplacian(T, mode=1, k=3))
laps.append(make_mode_laplacian(T, mode=2, k=30))
factors = estimate_from_laps(laps, 11, mode_shapes=(0, 1, 2))
x_hat = tl.cp_to_tensor(factors)

# x_hat = np.reshape(x_hat, orig_shape)
# T = np.reshape(T, orig_shape)
plt.plot(normalize_tensor(x_hat[3, 5, :], "minmax"))
plt.plot(T[3, 5, :])

plt.show()


print(T.shape)


exit()
sumsum = lambda x: np.sum(np.sum(x, axis=0), axis=0)

weights, factors, _ = tl.decomposition.parafac2(T, rank=6, verbose=False)
print(factors[2].shape)
plt.plot(np.sum(factors[2], axis=1))
plt.plot(sumsum(T))
plt.show()
