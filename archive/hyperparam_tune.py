import gc
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, average_precision_score

from models.lap_reg_cp import graph_regularized_als
from utils.tensor_processing import de_anomalize_tensor, normalize_tensor
from utils.anomaly_injector import *
from utils.model_eval import *


# -------------------- utilities --------------------


def cp_decompose(T, rank, tol=5e-5, init="random"):
    """Perform CP decomposition and reconstruct tensor."""
    factors = tl.decomposition.CP(
        rank=rank, tol=tol, init=init, verbose=0
    ).fit_transform(T)

    T_reconstructed = tl.cp_to_tensor(factors)
    return T_reconstructed


def reconstruction_error(T, T_reconstructed):
    """Relative Frobenius norm error."""
    return tl.norm(T_reconstructed - T) / tl.norm(T)


def squared_error_tensor(T, T_reconstructed):
    """Element-wise squared error."""
    return (T_reconstructed - T) ** 2


def compute_pr_auc(L, err):
    """Compute PR-AUC from labels and error."""
    L_flat = L.flatten()
    err_flat = err.flatten()

    precision, recall, _ = precision_recall_curve(L_flat, err_flat)
    pr_auc = average_precision_score(L_flat, err_flat)

    return pr_auc, precision, recall


# -------------------- Rank search --------------------
# ------------------- Basic methods -------------------


def find_best_cp_rank(T, L, rank_range=range(15, 30)):
    """Find CP rank maximizing PR-AUC."""
    best_rank = None
    best_pr_auc = -np.inf

    for rank in tqdm(range(*rank_range)):
        tqdm.write(f"rank: {rank}")

        T_rec = cp_decompose(T, rank)
        err = squared_error_tensor(T, T_rec)

        pr_auc, _, _ = compute_pr_auc(L, err)

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_rank = rank
            tqdm.write(f"New best: {(best_pr_auc, best_rank)}")

        gc.collect()

    print(f"Best PR-AUC: {best_pr_auc}, Rank: {best_rank}")
    return best_rank, best_pr_auc


def find_cp_rank_by_error(T, target_err=0.2, max_rank=15):
    """Find smallest rank achieving target reconstruction error."""
    for rank in tqdm(range(1, max_rank)):
        print(rank)

        T_rec = cp_decompose(T, rank)
        err = reconstruction_error(T, T_rec)

        if err < target_err:
            print(f"rank: {rank} | error: {err}")
            return rank, err

        gc.collect()

    print(f"Error never below {target_err}")
    return rank, err


def find_best_tucker_rank(T, L, rank_ranges):
    """Grid search Tucker ranks using best F1."""
    best_rank = None
    best_auc = -np.inf

    for ranks in tqdm(list(product(*rank_ranges))):
        factors = tl.decomposition.tucker(T, rank=ranks, tol=1e-3)
        T_rec = tl.tucker_to_tensor(factors)

        err = squared_error_tensor(T, T_rec)
        pr_auc, _, _ = compute_pr_auc(L, err)

        if pr_auc > best_auc:
            best_auc = pr_auc
            best_rank = ranks
            tqdm.write(f"New best: {(best_auc, best_rank)}")

        gc.collect()

    return best_rank, best_auc


# -------------------- Custom Methods --------------------


def find_best_graph_reg_rank(
    T,
    L,
    rank_range,
    laps,
    lambdas=(10, 0.1, 100),
    n_iter=20,
    n_E=1000,
):
    """Find best rank for graph-regularized CP using PR-AUC."""

    best_rank = None
    best_pr_auc = -np.inf

    L_flat = L.flatten()

    for rank in tqdm(range(*rank_range)):
        tqdm.write(f"calculating rank: {rank}")

        factors, _ = graph_regularized_als(
            T,
            rank=rank,
            laps=laps,
            lmbda=lambdas,
            n_iter=n_iter,
            n_E=n_E,
            verbose=False,
        )

        T_rec = tl.cp_to_tensor(factors)
        err = squared_error_tensor(T, T_rec)

        pr_auc, _, _ = compute_pr_auc(L, err)

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_rank = rank
            tqdm.write(f"New best: {(best_pr_auc, best_rank)}")

        gc.collect()

    print(f"Best PR-AUC: {best_pr_auc}, Rank: {best_rank}")
    return best_rank, best_pr_auc


def find_best_graph_reg_lambda(
    T,
    L,
    rank,
    laps,
    lambda_range=None,
    n_iter=10,
    n_E=1000,
):
    """Grid search best graph regularization parameters (lambdas)."""

    if lambda_range is None:
        lambda_range = 10.0 ** np.arange(-3, 3)

    best_lambda = None
    best_pr_auc = -np.inf

    L_flat = L.flatten()

    for lambdas in tqdm(list(product(lambda_range, repeat=3))):
        tqdm.write(f"testing lambda: {tuple(map(float, lambdas))}")

        factors, _ = graph_regularized_als(
            T,
            rank=rank,
            laps=laps,
            lmbda=lambdas,
            n_iter=n_iter,
            n_E=n_E,
            verbose=False,
        )

        T_rec = tl.cp_to_tensor(factors)
        err = squared_error_tensor(T, T_rec).flatten()

        pr_auc = average_precision_score(L_flat, err)

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_lambda = tuple(map(float, lambdas))
            tqdm.write(f"New best: {(best_pr_auc, best_lambda)}")

        gc.collect()

    print(f"Best PR-AUC: {best_pr_auc}, Lambda: {best_lambda}")
    return best_lambda, best_pr_auc
