from typing import Literal
from annoy import AnnoyIndex
import numpy as np
import tensorly as tl
from scipy.sparse import lil_matrix

type T_tensor = np.typing.NDArray | tl.tensor


def de_anomalize_tensor(
    T,
    low_rank,
    keep_pecentile: int = 98,
    alpha: float = 0.4,
    cp_kwargs: dict | None = None,
):
    if cp_kwargs is None:
        cp = tl.decomposition.CP(
            tol=5e-4, rank=low_rank, init="random", normalize_factors=True
        )
    else:
        cp = tl.decomposition.CP(rank=low_rank, **cp_kwargs)

    factors = cp.fit_transform(T)
    T_reconstructed = tl.cp_to_tensor(factors)
    resid = T - T_reconstructed
    threshold = np.percentile(np.abs(resid), keep_pecentile)
    resid_cleaned = np.where(np.abs(resid) < threshold, resid, 0)
    noise = alpha * resid_cleaned
    T_prepared = T_reconstructed + noise
    return T_prepared


def normalize_tensor(tensor, method="zscore", eps=1e-10):
    """
    Normalize tensor using specified method.
    Args:
        tensor (np.ndarray): Input tensor
        method (str): 'zscore' or 'minmax'
        eps (float): Small constant to avoid division by zero

    Returns:
        np.ndarray: Normalized tensor
    """
    if method == "zscore":
        mean = np.mean(tensor)
        std = np.std(tensor)
        return (tensor - mean) / (std + eps)

    elif method == "minmax":
        min_val = np.min(tensor)
        max_val = np.max(tensor)
        return (tensor - min_val) / (max_val - min_val + eps)

    else:
        raise ValueError("Unsupported normalization method")


def make_mode_similarity(
    tensor: T_tensor,
    mode: int,
    sim_type: Literal["covariance", "gaussian", "sim_hash"],
    sigma: float = 1,
) -> T_tensor:
    T = tensor.copy()
    T = tl.base.unfold(T, mode=mode)
    if sim_type == "covariance":
        W = T @ T.T
        W = np.fill_diagonal(W, 0)
        return W
    elif sim_type == "gaussian":
        _, nc = T.shape
        W = np.zeros((nc, nc))
        for i in range(nc):
            for j in range(nc):
                if i == j:
                    return
                W[i, j] = np.exp(-tl.norm(T[i, :] - T[j, :]) / sigma)
        return W
    else:
        print(f"{sim_type}:not supported...")


def make_mode_knn(
    tensor: T_tensor,
    mode: int,
    k_neighbors: int = 10,
    n_trees: int = 10,
) -> T_tensor:

    T_unfolded = tl.base.unfold(tensor, mode=mode)
    nr, nc = T_unfolded.shape
    print(f"mode:{mode}, shapes:{(nr,nc)}")
    t = AnnoyIndex(nc, "euclidean")

    for i in range(nr):
        t.add_item(i, T_unfolded[i, :])

    t.build(n_trees=n_trees)  # 10 trees
    knn_graph = lil_matrix((nr, nr))
    for i in range(nr):
        indices = t.get_nns_by_item(i, k_neighbors + 1)
        neighbors = [idx for idx in indices if idx != i][:k_neighbors]
        knn_graph[i, neighbors] = 1
    knn_graph = knn_graph.tocsr()
    return knn_graph
