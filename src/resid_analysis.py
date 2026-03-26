import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt


from hyperparam_tune import find_best_cp_rank
from models.RHOOI import r_hooi
from models.lap_reg_cp import graph_regularized_als
from utils.anomaly_injector import *
from models.robust_cp import robust_cp
from utils.model_eval import compute_tensor_model_metrics, print_metrics
from utils.tensor_processing import make_mode_laplacian, preprocess


# def decomp_recomp(T: tl.tensor, rank: int):
#     init = "random" if rank > 12 else "svd"
#     weights, factors, _ = robust_cp(T, rank=rank, n_iter=30, verbose=False, init=init)
#     reconst = tl.cp_to_tensor(cp_tensor=(weights, factors))
#
#     return reconst
#


def decomp_recomp(T: tl.tensor, rank: int):
    # rank = 12
    # lambdas = (3.2, 0.01, 0.2)
    # k = (9, 3, 113)
    rank = 20
    lambdas = (1.5, 0.8, 0.01)
    k = (1, 126, 347)
    laps = (
        make_mode_laplacian(T, mode=0, k=k[0], normalize=True),
        make_mode_laplacian(T, mode=1, k=k[1], normalize=True),
        make_mode_laplacian(T, mode=2, k=k[2], normalize=True),
    )
    cp_decomp, _ = graph_regularized_als(
        T,
        rank=rank,
        lmbda=lambdas,
        laps=laps,
        verbose=True,
        n_E=1000,
        n_iter=20,
    )
    reconst = tl.cp_to_tensor(cp_tensor=cp_decomp)
    return reconst


# def decomp_recomp(T: tl.tensor, rank: int):
#     init = "random" if rank > 12 else "svd"
#     weights, factors, _ = tl.decomposition.parafac2(
#         T, rank=rank, verbose=False, init=init
#     )
#     reconst = tl.cp_to_tensor(cp_tensor=(weights, factors))
#     return reconst


# def decomp_recomp(T: tl.tensor, rank):
#     # init = "random" if rank >, detect_anomalies_soft_tucker 12 else "svd"
#     weights, factors, _ = r_hooi(T, ranks=rank, verbose=False)
#     reconst = tl.tucker_to_tensor((weights, factors))
#     return reconst


def detect_anomalies_soft(res):
    abs_res = np.abs(res)
    # sigma = np.median(abs_res[abs_res < np.percentile(abs_res, 50)]) / 0.6745

    sigma = np.median(np.abs(res)) / 0.6745
    # lam = 2.5 * sigma
    lam = sigma * np.sqrt(2 * np.log(res.size))

    E = np.sign(res) * np.maximum(np.abs(res) - lam, 0)
    return E, lam


def main():
    T = np.load("data/abiline_ten.npy")
    T_train = T[:, :, : 12 * 24 * 7 * 3]
    print(T_train.shape)
    del T
    source, dest = np.random.randint(0, 12, 2)
    T_train = preprocess(T_train)
    T_train, L = inject_random_spikes(T_train)
    # T_train = np.reshape(T_train, (12 * 12, -1))
    # T_train = np.reshape(T_train, (12 * 24, 7, -1))
    # L = np.reshape(L, (12 * 12, -1))
    # L = np.reshape(L, (12 * 24, 7, -1))
    # T_train = np.reshape(T_train, (12 * 24, 2, -1))
    # L = np.reshape(L, (12 * 24, 2, -1))

    resids = T_train - decomp_recomp(T_train, 7)
    E, lam = detect_anomalies_soft(resids)
    # metrics = compute_tensor_model_metrics(E, L)
    # print_metrics(metrics=metrics)
    # exit()
    # E = E[E != 0]
    # print(np.size(E))
    # plt.plot(L[source, dest, :])
    # plt.plot(E[source, dest, :])
    # plt.hist(E.flatten(), bins=200)
    bins = np.linspace(np.min(resids), np.max(resids), 200)
    plt.hist(resids[L > 0].flatten(), bins=bins, density=False)
    plt.hist(resids[L <= 0].flatten(), alpha=0.5, bins=bins, density=False)
    plt.ylim((0, 70))
    # plt.hist(E[L > 0].flatten(), bins=bins, density=False)
    # plt.hist(E[L <= 0].flatten(), alpha=0.5, bins=bins, density=False)

    # plt.vlines(lam, 0, 7999)

    plt.show()


if __name__ == "__main__":
    main()
