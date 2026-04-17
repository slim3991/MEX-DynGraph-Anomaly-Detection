from itertools import product
import matplotlib.pyplot as plt
from typing import Literal
from annoy import AnnoyIndex
import numpy as np
import numpy.typing as npt
from scipy import sparse
import tensorly as tl
from scipy.sparse import lil_matrix, diags, eye

type T_tensor = np.typing.NDArray | tl.tensor


def preprocess(T, rank, keep_percentile, alpha):
    for i in range(12):
        for j in range(12):
            T[i, j, :] = normalize_tensor(T[i, j, :], "minmax")
    T = de_anomalize_tensor(
        T, low_rank=rank, keep_pecentile=keep_percentile, alpha=alpha
    )
    return T


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


def make_mode_knn_annoy(
    tensor,
    mode: int,
    k: int = 10,
    n_trees: int = 10,
    sparse: bool = True,
    distance: str = "euclidean",
):
    X = tl.base.unfold(tensor, mode=mode)

    n_samples, n_features = X.shape
    # print("n_samples: ", n_samples, "n_features: ", n_features)

    # print(n_samples, n_features)

    index = AnnoyIndex(n_features, metric=distance)
    for i in range(n_samples):
        index.add_item(i, X[i])
    index.build(50)

    W = lil_matrix((n_samples, n_samples))

    for i in range(n_samples):
        neighbors = index.get_nns_by_item(i, k + 1, search_k=1200)
        neighbors = [j for j in neighbors if j != i][:k]
        W[i, neighbors] = 1

    W = W.tocsr()

    if not sparse:
        return W.toarray()
    return W


def angle_between(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)

    return np.arccos(np.clip(dot_product, -1.0, 1.0))


def make_mode_knn(
    tensor,
    mode: int,
    k: int = 10,
    distance: str = "euclidean",
    sparse: bool = True,
):

    # Select distance function
    if distance == "euclidean":
        distance_func = lambda x, y: np.linalg.norm(x - y)
        reverse = False  # smaller = closer
    elif distance == "dot":
        distance_func = lambda x, y: np.dot(x, y)
        reverse = True  # larger = closer
    elif distance == "angular":
        distance_func = angle_between
        reverse = False
    else:
        raise ValueError("distance not provided")

    # Unfold tensor
    X = tl.base.unfold(tensor, mode=mode)
    n_samples, n_features = X.shape

    # Compute full distance/similarity matrix
    A = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            A[i, j] = distance_func(X[i], X[j])

    # Build k-NN graph
    W = lil_matrix((n_samples, n_samples))

    for i in range(n_samples):
        if reverse:
            neighbors = np.argsort(-A[i])  # largest first
        else:
            neighbors = np.argsort(A[i])  # smallest first

        neighbors = neighbors[1 : k + 1]  # skip self (index 0)

        for j in neighbors:
            W[i, j] = A[i, j]
            W[j, i] = A[i, j]  # make symmetric

    W = W.tocsr()

    if not sparse:
        return W.toarray()
    return W


def make_mode_laplacian(
    tensor,
    mode: int,
    k: int = 10,
    n_trees: int = 10,
    normalize: bool = True,
    sparse: bool = True,
    measure: str = "euclidian",
):
    W = make_mode_knn_annoy(tensor, mode=mode, k=k, sparse=True, distance=measure)
    W = 0.5 * (W + W.T)

    deg = np.array(W.sum(axis=1)).flatten()

    if normalize:
        with np.errstate(divide="ignore"):
            d_inv_sqrt = deg**-0.5
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

        D_inv_sqrt = diags(d_inv_sqrt)
        I = eye(W.shape[0], format="csr")

        L = I - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        D = diags(deg)
        L = D - W

    if not sparse:
        return L.toarray()
    return L


def make_interval_lap(size: int, interval: int):
    rows = []
    cols = []

    for i in range(size):
        for j in range(i + interval, size, interval):
            rows.append(i)
            cols.append(j)
            rows.append(j)
            cols.append(i)

    data = np.ones(len(rows))
    A = sparse.csr_matrix((data, (rows, cols)), shape=(size, size))

    d = np.array(A.sum(axis=1)).flatten()

    d_inv_sqrt = np.power(d, -0.5, where=d != 0)
    D_inv_sqrt = sparse.diags(d_inv_sqrt)

    I = sparse.eye(size)
    L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt

    return L_norm


def make_gaussian_proximity_laplacian(size: int, sigma: float):
    # 1. Define the window of influence (truncate at 4*sigma for precision)
    window = int(np.ceil(4 * sigma))

    rows = []
    cols = []
    data = []

    for i in range(size):
        # Only look at neighbors within the window
        start = max(0, i - window)
        end = min(size, i + window + 1)

        for j in range(start, end):
            if i == j:
                continue  # Skip self-loops for Adjacency

            # Gaussian formula
            dist_sq = (i - j) ** 2
            sim = np.exp(-dist_sq / (2 * sigma**2))

            rows.append(i)
            cols.append(j)
            data.append(sim)

    A = sparse.csr_matrix((data, (rows, cols)), shape=(size, size))

    # 2. Normalize to get the Laplacian
    d = np.array(A.sum(axis=1)).flatten()
    # Handle isolated nodes if sigma is too small
    d_inv_sqrt = np.power(d, -0.5, where=d != 0)
    D_inv_sqrt = sparse.diags(d_inv_sqrt)

    I = sparse.eye(size)
    L_norm = I - D_inv_sqrt @ A @ D_inv_sqrt

    return L_norm


def test():
    L = make_interval_lap(100, 10)
    plt.spy(L)
    plt.show()


if __name__ == "__main__":
    test()
