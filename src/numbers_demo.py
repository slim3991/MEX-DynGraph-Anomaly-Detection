from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, precision_recall_curve
from tqdm import tqdm
import tensorly as tl

from tensorly.tenalg import unfolding_dot_khatri_rao
from scipy.sparse import diags, eye, csc_matrix

from dataset_parsers.read_mnist import create_anomaly_tensor, visualize_tensor_grid
from utils.anomaly_injector import (
    generate_shape,
    inject_alpha_anomaly,
    inject_random_spikes,
    inject_random_spikes_normal,
)
from utils.model_eval import (
    compute_tensor_model_binary_metrics,
    compute_tensor_model_metrics,
    metrics_to_latex,
    print_metrics,
)
from utils.tensor_processing import de_anomalize_tensor, make_mode_knn, normalize_tensor


def make_laplacian(T, mode, k, sparse_output=True):
    """
    Build the normalized graph Laplacian for the k-NN graph of a tensor mode.

    Parameters
    ----------
    T : tensorly tensor or ndarray
        Input tensor.
    mode : int
        Mode along which to compute the Laplacian.
    k : int
        Number of neighbors for k-NN graph.
    sparse_output : bool
        If True, return a sparse Laplacian; otherwise, return dense.

    Returns
    -------
    L : ndarray or csr_matrix
        Normalized Laplacian matrix.
    """
    # Build k-NN adjacency matrix (symmetric)
    W = make_mode_knn(T, mode=mode, k_neighbors=k, sparse=True)
    W = 0.5 * (W + W.T)  # make symmetric

    # Degree vector
    deg = np.array(W.sum(axis=1)).flatten()

    # D^(-1/2)
    with np.errstate(divide="ignore"):
        d_inv_sqrt = np.power(deg, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    if sparse_output:
        D_inv_sqrt = diags(d_inv_sqrt)
        I = eye(W.shape[0], format="csr")
        L = I - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(W.shape[0]) - D_inv_sqrt @ W.toarray() @ D_inv_sqrt

    return L


T, _ = create_anomaly_tensor(8, 8 * 8, 5)
print(T.shape)
noise = np.random.randn(*T.shape)
T = T + noise

# temp = tl.base.unfold(T, 1)
# U, S, _ = np.linalg.svd(temp, full_matrices=False)
# plt.semilogy(S)
# plt.show()
# exit()

rank = 10
laps = []
laps.append(make_laplacian(T, mode=0, k=20))
laps.append(make_laplacian(T, mode=1, k=20))
laps.append(make_laplacian(T, mode=2, k=20))


custom_factors = tl.decomposition.tucker(
    T,
    rank=(23, 23, 50),
)

# print(custom_factors.factors[0].shape)
# print(custom_factors.factors[1].shape)
# print(custom_factors.factors[2].shape)
reconst = tl.tucker_to_tensor(custom_factors)
visualize_tensor_grid(T, random_select=False)
visualize_tensor_grid(reconst, random_select=False)
plt.show()
# E = E > 0
# resid = reconst - T
exit()
