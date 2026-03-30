from itertools import product
from typing import Optional
import numpy as np
import tensorly as tl
import matplotlib.pyplot as plt

from utils.tensor_processing import de_anomalize_tensor, normalize_tensor
from utils.utils import detect_anomalies_soft


# def detect_anomalies(S, factors, epsilon):
#     M_hat = factors
#     residuals = np.abs(S - M_hat)
#     flat_residuals = residuals.flatten()
#     if epsilon >= len(flat_residuals):
#         threshold = 0
#     else:
#         threshold = np.partition(flat_residuals, -epsilon)[-epsilon]
#     E = np.where(residuals >= threshold, S, 0)
#     # E = np.where(residuals >= threshold, S + M_hat, 0)  # TODO: Verify this
#     anomaly_indices = np.argwhere(residuals >= threshold)
#
#     return E, anomaly_indices


def r_hooi(
    X, ranks, n_iter=50, tol=1e-6, threshold: Optional[float] = None, verbose=False
):
    """
    HOOI decomposition

    Parameters
    ----------
    X : ndarray
        Input tensor
    ranks : list
        Desired Tucker ranks
    n_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance

    Returns
    -------
    core : ndarray
    factors : list of factor matrices
    """

    # Initialize factors via HOSVD
    x_hat = tl.decomposition.tucker(X, ranks, tol=1e-3, init="random")
    factors = x_hat.factors
    core = x_hat.core

    normX = np.linalg.norm(X)
    M = X.copy()
    old_error = 1e20
    S = np.zeros_like(M)

    for iteration in range(n_iter):

        for mode in range(X.ndim):

            # project tensor onto all modes except current
            projection_factors = [
                factors[i].T if i != mode else None for i in range(X.ndim)
            ]

            Y = tl.tenalg.multi_mode_dot(M, projection_factors, skip=mode)

            Yn = tl.base.unfold(Y, mode)

            U, _, _ = np.linalg.svd(Yn, full_matrices=False)

            factors[mode] = U[:, : ranks[mode]]

        # compute core tensor
        core = tl.tenalg.multi_mode_dot(M, [f.T for f in factors])

        # reconstruction
        X_hat = tl.tenalg.multi_mode_dot(core, factors)
        residual = X - X_hat

        error = np.linalg.norm(residual) / tl.norm(M)
        diff = abs(old_error - error)

        S = detect_anomalies_soft(residual)

        M = X - S

        if verbose and iteration % 10 == 0:
            print("diff: ", diff)
            print("iteration: ", iteration)
        if diff < tol:
            if verbose:
                print(f"reached tol")
            break
        old_error = error

    return (core, factors), S


##################################


def main():
    # T, idx = create_anomaly_tensor(1, 8 * 8, 3)
    # print(T.shape)
    # core, factors, S = r_hooi(T, (27, 27, 3), tol=1e-4, n_iter=51)
    # reconst = tl.tucker_to_tensor((core, factors))
    # visualize_tensor_grid(T, random_select=False)
    # visualize_tensor_grid(reconst, random_select=False)
    # visualize_tensor_grid(S, random_select=False)
    # plt.show()
    # exit()

    source, dest = np.random.randint(0, 11), np.random.randint(0, 11)

    T = np.load("data/abiline_ten.npy")
    T = T[:, :, 5_000:10_000]

    for i in range(12):
        for j in range(12):
            T[i, j, :] = normalize_tensor(T[i, j, :], "minmax")

    T = de_anomalize_tensor(T, 20)

    # T, mask, idx_list = inject_shapes(
    #     T,
    #     start_min=10,
    #     start_max=4000,
    #     min_durantion=10,
    #     max_duration=1000,
    #     shape="ramp",
    #     retrun_idx_list=True,
    #     amplitude_range=(0.3, 0.9),
    # )

    # exit()
    # plt.plot(T[source, dest, :], "--", alpha=0.5, label="original")
    # plt.plot(mask[source, dest, :], "--", label="injected anomaly")
    # plt.show()
    # exit()
    # mask = np.zeros_like(T)

    # ############# Find best tucker rank ################
    best_rank = (0, 0, 0)
    record_f1 = 0
    for i, j, k in product(range(7, 13), range(7, 13), range(19, 26)):
        print((i, j, k))

        core, factors, S = r_hooi(T, ranks=(i, j, k), tol=1e-8, n_anomalies=1000)
        reconst = tl.tucker_to_tensor((core, factors))
        err = (reconst - T) ** 2
        err = err.flatten()
        mask = mask.flatten()
        precision_curve, recall_curve, thresholds = precision_recall_curve(mask, err)
        pr_auc = average_precision_score(mask, err)
        f1_scores = (
            2
            * (precision_curve * recall_curve)
            / (precision_curve + recall_curve + 1e-10)
        )
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        if best_f1 > record_f1:
            record_f1 = best_f1
            best_rank = (i, j, k)
            print(record_f1, best_rank)
    print(record_f1, best_rank)

    exit()
    ######################################################

    ranks = (12, 12, 20)
    core, factors, S = r_hooi(T, ranks=ranks, tol=1e-8, n_anomalies=1000)
    reconst = tl.tucker_to_tensor((core, factors))
    err = (reconst - T) ** 2
    basic = tl.decomposition.tucker(T, rank=(11, 10, 8), tol=1e-8)
    basic = tl.tucker_to_tensor(basic)

    metrics = compute_tensor_model_metrics(err, mask)
    # metrics = compute_tensor_model_binary_metrics(S > 0, mask)
    print(metrics_to_latex(metrics=metrics, name="Robust Tucker"))
    exit()

    print(f"basic reconst error: {tl.norm(basic-T)}")
    print(f"regularized reconst error: {tl.norm(reconst-T)}")

    # random shapes
    S = S > 0
    correct_anomalies = 0
    # for source, dest, start, end in idx_list:
    #     correct_anomalies += (
    #         np.sum(S[source, dest, start:end]) > 0
    #     )  # nr of identified anomalies

    # eval_tensor_model_binary(S, mask)
    eval_tensor_model((reconst - T) ** 2, mask)

    tucker_reconst = tl.tucker_to_tensor(tl.decomposition.tucker(T, rank=ranks))
    tucker_resid = (T - tucker_reconst) ** 2
    eval_tensor_model(tucker_resid, mask)

    exit()

    print(f"number of anomalous points: {np.sum(mask)}")
    N = np.size(T) - np.sum(mask)
    P = np.sum(mask)
    print(
        "False positive rate: ", np.sum((S > 0) & (mask == 0)) / N
    )  # False positives rate
    print("True positive rate: ", np.sum((S > 0) & (mask > 0)) / P)
    # print(f"correct_anomalies: {correct_anomalies} of {len(idx_list)}")

    # # Radnom spikes
    # print(f"number of random spikes: {np.sum(mask)}")
    # print(f"fratction identified: {np.sum((E > 0) & (mask > 0)) / np.sum(mask):.2f}")

    plt.plot(T[source, dest, :], "--", alpha=0.5, label="original")
    plt.plot(basic[source, dest, :], "--", alpha=0.8, label="basic cp")
    plt.plot(reconst[source, dest, :], label="regularized")
    # plt.plot(S[source, dest, :], label="identified anomaly")
    # plt.plot(mask[source, dest, :], "--", label="injected anomaly")
    plt.legend()
    plt.title(f"{(source,dest)}")
    plt.show()
    exit()
