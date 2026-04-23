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

T, L, _, _ = create_event_dataset("train", ampf=10)
# T, L, _, _ = create_spike_dataset("train")

# # ddos injection
# L = np.zeros_like(T)
# for _ in range(100):
#     a = np.random.randint(0, 12)
#     T, Lp = inject_DDoS(T, duration=10, n_senders=5, target=a, amplitude_factor=10)
#     L += Lp
# L = np.where(L > 0, 1, 0)

# T, L, _, _ = create_outage_dataset("train")
# T, L, _, _ = create_ddos_dataset("train")

# lap_parms = {
#     "lambda_1": 0,
#     "lambda_2": 0,
#     "lambda_smooth": 10,
#     "lambda_interval": 200,
#     "measure": "dot",
#     "ks_1": 0,
#     "ks_2": 5,
# }
# tucker_lap_params = {
#     "lambda_1": 0,
#     "lambda_2": 0,
#     "lambda_smooth": 1,
#     "lambda_interval": 10,
#     "measure": "euclidean",
#     "ks_1": 0,
#     "ks_2": 0,
# }

ANOMALY_TYPE = "events"
# with open("src/model_config.yaml") as f:
#     m_conf = yaml.safe_load(f)
# model_confs = m_conf[f"{ANOMALY_TYPE}_configs"]
# cp_lap_params = model_confs["GRRCP_no_robust"]["laps_params"]
# tucker_lap_params = model_confs["GRRTucker_no_robust"]["laps_params"]

rank = (15, 15, 15)

# X_hat = MyGRTuckerDecomp(
#     rank=rank, laplacian_parameters=tucker_lap_params, tol=1e-4, threshold=0
# ).fit_transform(T, L)

# X_hat_basic = tl.tucker_to_tensor(
#     tl.decomposition.tucker(T, rank=rank, tol=1e-4, init="random")
# )
X_hat_rob_t = MyRHOOITenDecomp(rank=rank, tol=1e-4).fit_transform(T, L)

# X_hat = MyGRTenDecomp(
#     rank=rank,
#     laplacian_parameters=cp_lap_params,
#     threshold=0,
#     tol=1e-4,
# ).fit_transform(T, L)

rank = 15
# X_hat_basic = tl.cp_to_tensor(
#     tl.decomposition.parafac(
#         T,
#         tol=1e-4,
#         rank=rank,
#         init="random",
#     )
# )
X_hat_rob = MyRCPTenDecomp(rank=rank, tol=1e-4).fit_transform(T, L)

# res = np.abs(X_hat - T)
# precision, recall, thresholds = precision_recall_curve(L.ravel(), res.ravel())
# pr_auc = auc(recall, precision)
# print("pr-auc, graph: ", pr_auc)

res = np.abs(X_hat_rob - T)
precision, recall, thresholds = precision_recall_curve(L.flatten(), res.flatten())
pr_auc = auc(recall, precision)
print("pr-auc, robust: ", pr_auc)

precision, recall, thresholds = precision_recall_curve(L.flatten(), res.flatten())
pr_auc = auc(recall, precision)
print("pr-auc, basic: ", pr_auc)

# print("graph: ", tl.norm(X_hat - Tp) / tl.norm(Tp))
print("robust: ", tl.norm(X_hat_rob - Tp) / tl.norm(Tp))
# print("basic: ", tl.norm(X_hat_basic - Tp) / tl.norm(Tp))


plt.plot(L[source, dest, :], alpha=0.5)
plt.plot(T[source, dest, :], alpha=0.5)
# plt.plot((X_hat_basic[source, dest, :]), label="Basic")
plt.plot((X_hat_rob[source, dest, :]), label="robust")
plt.plot((X_hat_rob_t[source, dest, :]), label="robust tucker")

# plt.plot((X_hat[source, dest, :]), label="Graph-Regularized")
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
        param0 = trial.suggest_float(name="param0", low=0, high=10000)
        # param1 = trial.suggest_float(name="param1", low=0, high=10000)
        # ks0 = trial.suggest_int(name="ks0", low=0, high=10)
        # ks1 = trial.suggest_int(name="ks1", low=0, high=5)
        # measure = trial.suggest_categorical(
        #     name="measure", choices=["euclidean", "angular", "dot"]
        # )

        X_hat = MyGRTenDecomp(
            rank=10,
            ks=[None, None, None],
            lambdas=[0, 0, 0],
            local_threshold=None,
            measure="dot",
            tol=1e-4,
        ).fit_transform(T, L)
        res = np.abs(X_hat - T)
        precision, recall, _ = precision_recall_curve(L.flatten(), res.flatten())
        pr_auc = auc(recall, precision)
        return pr_auc

    study = optuna.create_study(direction="maximize", study_name="GRT")
    study.optimize(lambda trial: objective(trial, T), n_trials=20)
    print("Best trial: ", study.best_trial.number, ", with value: ", study.best_value)
    print("Best Params", study.best_params, end="\n\n")


find_laps()
exit()


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
