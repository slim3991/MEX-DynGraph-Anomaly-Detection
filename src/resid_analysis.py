import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt


from models import MyGRTenDecomp
from utils.anomaly_injector import *
from utils.model_eval import print_metrics
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
    T_train = T[:, :, :1000]
    print(T_train.shape)
    del T
    source, dest = np.random.randint(0, 12, 2)
    T_train = preprocess(T_train, 20, alpha=0.4, keep_percentile=95)
    T_train, L = inject_random_spikes_normal(T_train)

    x_hat = MyGRTenDecomp(
        10, (-4558, -3787, 3476), (5, 7, 10), threshold=0.001, local_threshold=0.22
    ).fit_transform(T_train)
    plt.plot(T_train[source, dest, :])
    plt.plot(x_hat[source, dest, :])
    plt.plot(L[source, dest, :])
    plt.show()
    exit()

    resids = x_hat.residuals(T_train)
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
