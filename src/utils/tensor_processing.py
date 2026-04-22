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
    # n_trees: int = 10,
    normalize: bool = False,
    sparse: bool = True,
    measure: str = "euclidian",
):
    W = make_mode_knn(tensor, mode=mode, k=k, sparse=True, distance=measure)
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


import numpy as np
from scipy import sparse


def make_interval_lap(size: int, interval: int, weights: list = [0.25, 0.5, 0.25]):
    """
    Creates a Laplacian where connections are weighted by a specific distribution.

    Args:
        size: Dimension of the matrix.
        interval: The step size for future connections.
        weights: A list of weights (must be odd length to have a clear center).
                 The center of the list aligns with the 'step' (i + interval).
    """

    rows = []
    cols = []
    data = []

    # Calculate window offset based on weights length
    # e.g., if weights is [0.25, 0.5, 0.25], window_radius is 1
    window_radius = len(weights) // 2

    for i in range(size):
        # Look at future intervals
        for step in range(i + interval, size, interval):

            # Iterate through the weights and apply them to neighbors
            for idx, weight in enumerate(weights):
                # Calculate the neighbor's position relative to the center 'step'
                neighbor = step - window_radius + idx

                if 0 <= neighbor < size:
                    # Connection from i to neighbor
                    rows.append(i)
                    cols.append(neighbor)
                    data.append(weight)

                    # Symmetric connection (neighbor back to i)
                    rows.append(neighbor)
                    cols.append(i)
                    data.append(weight)

    # 1. Create Weighted Adjacency Matrix
    # Using sum_duplicates=True (default) handles cases where edges might overlap
    A = sparse.csr_matrix((data, (rows, cols)), shape=(size, size))

    # 2. Laplacian Calculation: L = D - A
    # The degree d_i is the sum of weights connected to node i
    d = np.array(A.sum(axis=1)).flatten()
    D = sparse.diags(d)

    L = D - A

    return L


def make_gaussian_proximity_laplacian(size: int, sigma: float):
    window = int(np.ceil(4 * sigma))
    rows, cols, data = [], [], []

    for i in range(size):
        start = max(0, i - window)
        end = min(size, i + window + 1)
        for j in range(start, end):
            if i == j:
                continue
            dist_sq = (i - j) ** 2
            sim = np.exp(-dist_sq / (2 * sigma**2))

            # Clip very small values to keep the matrix sparse but meaningful
            if sim > 1e-4:
                rows.append(i)
                cols.append(j)
                data.append(sim)

    A = sparse.csr_matrix((data, (rows, cols)), shape=(size, size))

    # Use Combinatorial Laplacian: L = D - A
    d = np.array(A.sum(axis=1)).flatten()
    D = sparse.diags(d)
    L = D - A
    return L


def make_ar_similarity_laplacian(size: int, lookback: int = 3, decay: float = 0.5):
    """
    Creates a Laplacian where each node is connected to its previous 'p' neighbors,
    with weights decaying according to an AR-like influence.

    Args:
        size: Dimension of the matrix (number of time steps).
        lookback: The order 'p' (how many steps back to connect).
        decay: The rate at which similarity decreases for older steps.
               Weight = decay^distance
    """
    rows = []
    cols = []
    data = []

    for i in range(size):
        # Look back up to 'lookback' steps
        for p in range(1, lookback + 1):
            prev_node = i - p

            if prev_node >= 0:
                # AR-style weighting: influence drops as distance increases
                weight = decay**p

                # Connection from i to past neighbor
                rows.append(i)
                cols.append(prev_node)
                data.append(weight)

                # Symmetric connection (past neighbor to i)
                rows.append(prev_node)
                cols.append(i)
                data.append(weight)

    # 1. Create Weighted Adjacency Matrix
    # sum_duplicates=True is important here if edges are defined twice
    A = sparse.csr_matrix((data, (rows, cols)), shape=(size, size))

    # 2. Laplacian Calculation: L = D - A
    # The degree matrix D contains the sum of weights for each time step
    d = np.array(A.sum(axis=1)).flatten()
    D = sparse.diags(d)

    L = D - A

    return L


import matplotlib.colors as colors


def test():
    size = 70
    L_ar = make_ar_similarity_laplacian(size, lookback=8, decay=0.8)
    L_int = make_interval_lap(size=size, interval=10)

    # 1. Separate Plot for Gaussian
    plt.figure(figsize=(8, 6))
    # We use SymLogNorm to handle the scale difference between
    # the diagonal (degree) and the small weights.
    plt.imshow(
        L_int.toarray(),
        cmap="RdBu_r",
        # norm=colors.SymLogNorm(linthresh=0.01, linscale=1, vmin=-1, vmax=5),
        norm=colors.PowerNorm(gamma=0.5),
    )
    plt.colorbar(label="Weight")
    plt.title("Interval Proximity Laplacian\n(Normalized)")
    plt.show()

    # 2. Separate Plot for AR Similarity
    plt.figure(figsize=(8, 6))
    # Alternatively, use a power-law norm to "dim" the diagonal
    plt.imshow(
        L_ar.toarray(),
        cmap="PRGn",
        norm=colors.PowerNorm(gamma=0.5),
        # norm=colors.SymLogNorm(linthresh=0.01, linscale=1, vmin=-1, vmax=5),
    )
    plt.colorbar(label="Weight")
    plt.title("AR Similarity Laplacian\n(Normalized)")
    plt.show()


if __name__ == "__main__":
    test()
