import numpy as np
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from tensorly.tenalg import khatri_rao
from tensorly.base import unfold
import matplotlib.pyplot as plt

from scipy.linalg import solve_sylvester
from scipy.sparse import diags, eye, csr_matrix
from utils import model_eval
from utils.anomaly_injector import (
    generate_shape,
    inject_alpha_anomaly,
    inject_random_spikes,
    inject_shapes,
)
from utils.tensor_processing import de_anomalize_tensor, make_mode_knn, normalize_tensor


def normalize_cp_with_weights(A, B, C):
    R = A.shape[1]
    weights = np.zeros(R)
    for r in range(R):
        # Calculate the norm for each factor column
        norm_a = np.linalg.norm(A[:, r])
        norm_b = np.linalg.norm(B[:, r])
        norm_c = np.linalg.norm(C[:, r])

        # The total magnitude of this component
        weights[r] = norm_a * norm_b * norm_c

        # Normalize the factors to unit length
        if weights[r] > 0:
            A[:, r] /= norm_a + 1e-12
            B[:, r] /= norm_b + 1e-12
            C[:, r] /= norm_c + 1e-12

    return A, B, C, weights


def calculate_loss(M, A, B, C, Fa, Fb, Fc, Da, Db, Dc, gx, gy, gz):
    # 1. Norm of original tensor (pre-calculate this once outside the loop!)
    norm_M = np.linalg.norm(M) ** 2

    # 2. Norm of the reconstructed tensor (using Gram matrices)
    # ||[[A,B,C]]||^2 = sum((A.T@A) * (B.T@B) * (C.T@C))
    G_A, G_B, G_C = A.T @ A, B.T @ B, C.T @ C
    norm_hat = np.sum(G_A * G_B * G_C)

    # 3. Inner product <M, [[A,B,C]]>
    # This is equivalent to sum(unfold(M, 0) * (A @ khatri_rao(C, B).T))
    # But it's faster to reuse the numerator logic from your update:
    inner_prod = np.sum(A * (tl.unfold(M, 0) @ tl.tenalg.khatri_rao([C, B])))

    reconstruction_loss = 0.5 * (norm_M + norm_hat - 2 * inner_prod)

    # 4. Graph Regularization terms: Tr(A.T * L * A)
    # L = D - F
    reg_a = 0.5 * gx * np.trace(A.T @ (Da - Fa) @ A)
    reg_b = 0.5 * gy * np.trace(B.T @ (Db - Fb) @ B)
    reg_c = 0.5 * gz * np.trace(C.T @ (Dc - Fc) @ C)

    return reconstruction_loss + reg_a + reg_b + reg_c


def detect_anomalies(S, factors, epsilon):
    """
    Implements Section 4.2: Solution to Anomaly Detection Sub-Problem.

    S: The original input tensor (the observation)
    factors: The tuple (weights, [A, B, C]) from your factorization
    epsilon: The sparsity budget (total number of anomaly entries to detect)
    """
    # 1. Reconstruct the normal tensor M from the factors
    # This represents what the network *should* look like if there were no anomalies
    M_hat = tl.cp_to_tensor(factors)

    # 2. Calculate the absolute residual (Error)
    # The paper refers to these errors as the basis for anomaly selection
    residuals = np.abs(S - M_hat)

    # 3. Find the threshold beta(epsilon)
    # beta(epsilon) is the epsilon-th largest value in the residual tensor
    flat_residuals = residuals.flatten()

    # Use partition to efficiently find the top-k threshold
    # It puts the k-th largest element at its sorted position
    if epsilon >= len(flat_residuals):
        # Edge case: budget larger than tensor size
        threshold = 0
    else:
        threshold = np.partition(flat_residuals, -epsilon)[-epsilon]

    # 4. Create the Sparse Anomaly Tensor E (Equation 27)
    # If the error is above the threshold, it is an anomaly.
    # Otherwise, it's considered noise or part of the normal traffic.
    E = np.where(residuals >= threshold, S, 0)  # TODO: Verify this
    # E = np.where(residuals >= threshold, S - M_hat, 0)  # TODO: Verify this

    # Optional: Get indices of top anomalies for debugging
    anomaly_indices = np.argwhere(residuals >= threshold)

    return E, anomaly_indices


def graph_tensor_cp(
    X,
    rank,
    Da,
    Fa,
    Db,
    Fb,
    Dc,
    Fc,
    gamma_a=0.1,
    gamma_b=0.1,
    gamma_c=0.1,
    max_iter=200,
    tol=1e-6,
    verbose=False,
):
    M = X.copy()
    old_loss = 1e20

    I, J, K = M.shape
    R = rank
    eps = 1e-7

    A = np.random.rand(I, R)
    B = np.random.rand(J, R)
    C = np.random.rand(K, R)
    old_norm = 1e20

    for it in range(max_iter):
        print(f"starting iteration: {it}")
        X1 = unfold(M, 0)
        X2 = unfold(M, 1)
        X3 = unfold(M, 2)

        # ===== Update A =====
        KR_CB = khatri_rao([C, B])
        G_A = (B.T @ B) * (C.T @ C)

        num_A = X1 @ KR_CB + gamma_a * (Fa @ A)
        den_A = A @ G_A + gamma_a * (Da @ A)

        A *= num_A / (den_A + eps)
        # A = np.maximum(A, eps)

        # ===== Update B =====
        KR_CA = khatri_rao([C, A])
        G_B = (A.T @ A) * (C.T @ C)

        num_B = X2 @ KR_CA + gamma_b * (Fb @ B)
        den_B = B @ G_B + gamma_b * (Db @ B)

        B *= num_B / (den_B + eps)
        # B = np.maximum(B, eps)

        # ===== Update C =====
        KR_BA = khatri_rao([B, A])
        G_C = (A.T @ A) * (B.T @ B)

        num_C = X3 @ KR_BA + gamma_c * (Fc @ C)
        den_C = C @ G_C + gamma_c * (Dc @ C)

        C *= num_C / (den_C + eps)
        # C = np.maximum(C, eps)

        weights = np.ones(rank)
        X_prime = (weights, [A, B, C])

        E, idx = detect_anomalies(X, X_prime, 10000)
        M = X - E

        A, B, C, weights = normalize_cp_with_weights(A, B, C)

        new_norm = tl.norm(weights)
        diff = old_norm - new_norm
        if verbose:
            print(f"Iter {it}, diff={diff:.6e}")

        old_norm = new_norm
        # if diff < tol:
        #     break
        # old_loss = loss

    return CPTensor((weights, [A, B, C])), E


# def make_laplacian(T, mode, k):
#     W = make_mode_knn(T, mode=mode, k_neighbors=k, sparse=True)
#     W = 0.5 * (W + W.T)
#
#     W = W.toarray()
#
#     # Degree
#     deg = W.sum(axis=1)
#     D = np.diag(deg)
#
#     return D, W


def make_laplacian(T, mode, k):
    unf = tl.unfold(T, mode=mode)

    W = unf @ unf.T
    np.fill_diagonal(W, 0)
    W = normalize_tensor(W, "minmax")
    # W = np.where(W > 0.5, 1, 0)

    # Degree
    deg = W.sum(axis=1)
    D = np.diag(deg)

    return D, W


##################################


source, dest = np.random.randint(0, 11), np.random.randint(0, 11)

T = np.load("data/abiline_ten.npy")
T = T[:, :, 5_000:10_000]

# for i in range(12):
#     for j in range(12):
#         T[i, j, :] = normalize_tensor(T[i, j, :], "minmax")

T = normalize_tensor(T, "minmax")
T = de_anomalize_tensor(T, 5)


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
T, mask = inject_random_spikes(T)


Da, Fa = make_laplacian(T, mode=0, k=5)
Db, Fb = make_laplacian(T, mode=1, k=5)
Dc, Fc = make_laplacian(T, mode=2, k=200)

# np.random.seed(42)


reconst, E = graph_tensor_cp(
    T,
    12,
    Da=Da,
    Fa=Fa,
    Db=Db,
    Fb=Fb,
    Dc=Dc,
    Fc=Fc,
    gamma_a=10,
    gamma_b=10,
    gamma_c=0.1,
    max_iter=20,
    tol=2,
    verbose=True,
)

reconst = tl.cp_to_tensor(reconst)
# basic = tl.decomposition.CP(rank=10, n_iter_max=20, init="random").fit_transform(T)
# basic = tl.cp_to_tensor(basic)

# random shapes
E = E > 0
# correct_anomalies = 0
# for source, dest, start, end in idx_list:
#     correct_anomalies += (
#         np.sum(E[source, dest, start:end]) > 0
#     )  # nr of identified anomalies


# model_eval.eval_tensor_model_binary(E, mask)
model_eval.eval_tensor_model((reconst - T) ** 2, mask)
exit()
print(f"{np.sum(mask)}")
print(
    "False positive rate", np.sum((E > 0) & (mask == 0)) / np.sum(E)
)  # False positives rate
# print(f"correct_anomalies: {correct_anomalies} of {len(idx_list)}")

# # Radnom spikes
# print(f"number of random spikes: {np.sum(mask)}")
# print(f"fratction identified: {np.sum((E > 0) & (mask > 0)) / np.sum(mask):.2f}")


plt.plot(T[source, dest, :], "--", alpha=0.5, label="original")
plt.plot(basic[source, dest, :], "--", alpha=0.8, label="basic cp")
plt.plot(reconst[source, dest, :], label="regularized")
plt.plot(E[source, dest, :], label="identified anomaly")
plt.plot(mask[source, dest, :], "--", label="injected anomaly")
plt.legend()
plt.show()
exit()
