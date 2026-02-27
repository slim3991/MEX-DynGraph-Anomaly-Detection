import gc
import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from models.incremental_svd import IncrementalSVD
from utils.anomaly_injector import *
from utils.model_eval import *


def tucker_largest_comp(T, source, dest, R):
    core, factors = tl.decomposition.tucker(T, rank=[R, R, R], init="svd")
    U, V, W = factors

    T_reconstructed = tl.tucker_to_tensor((core, [U, V, W]))

    # ---------------------------------------------------------
    # 1. Find largest entries in the core tensor (by magnitude)
    # ---------------------------------------------------------
    num_components = 4  # how many largest core elements to inspect

    flat_indices = np.argsort(np.abs(core.ravel()))[::-1]
    top_indices = np.unravel_index(flat_indices[:num_components], core.shape)
    top_indices = list(zip(*top_indices))

    print("Top core entries (index, value):")
    for idx in top_indices:
        print(idx, str(core[idx]))

    # ---------------------------------------------------------
    # 2. Build individual Tucker components
    # ---------------------------------------------------------
    def build_component(i, j, k):
        return core[i, j, k] * np.einsum("a,b,c->abc", U[:, i], V[:, j], W[:, k])

    components = [build_component(i, j, k) for (i, j, k) in top_indices]

    # ---------------------------------------------------------
    # 3. Plot selected components
    # ---------------------------------------------------------
    fig, axes = plt.subplots(
        num_components, 1, figsize=(8, 2.5 * num_components), sharex=True
    )

    if num_components == 1:
        axes = [axes]

    for idx, comp in enumerate(components):
        axes[idx].plot(comp[source, dest, :], label=f"Component {top_indices[idx]}")
        axes[idx].plot(T[source, dest, :], "--", alpha=0.5, label="Original")
        axes[idx].plot(
            T_reconstructed[source, dest, :], "--", alpha=0.5, label="Full Recon"
        )

        axes[idx].set_title(
            f"Core index {top_indices[idx]}  |  value={core[top_indices[idx]]:.4f}"
        )
        axes[idx].legend()
        axes[idx].grid(True)

    axes[-1].set_xlabel("time")
    plt.tight_layout()
    plt.show()


def cp_largest_comp(T, source, dest, rank):
    # Fit CP
    cp = tl.decomposition.CP(
        tol=5e-6, rank=rank, init="svd", verbose=10, normalize_factors=True
    )
    factors = cp.fit_transform(T)

    weights, (A, B, C) = factors  # A: mode-0, B: mode-1, C: mode-2 (time)

    # Number of time points
    T_len = C.shape[0]
    order = np.argsort(-weights)  # negative for descending

    # print(weights)
    # exit()

    weights = weights[order]
    A = A[:, order]
    B = B[:, order]
    C = C[:, order]
    fig, axes = plt.subplots(rank, 1, figsize=(8, 2.5 * rank), sharex=True)

    T_reconstructed = tl.cp_to_tensor(factors)
    # If rank == 1, axes is not iterable
    if rank == 1:
        axes = [axes]

    for r in range(rank):
        component_signal = weights[r] * A[source, r] * B[dest, r] * C[:, r]
        axes[r].plot(component_signal, label="component")
        axes[r].plot(T[source, dest, :], "--", alpha=0.5, label="original")
        axes[r].plot(
            T_reconstructed[source, dest, :], "--", alpha=0.5, label="reconstructed"
        )

        axes[r].set_title(f"Component {r+1}, weight: {weights[r]/np.sum(weights):.2f}")
        axes[r].grid(True)
        axes[r].legend()

    plt.show()


def cp_banded_ranks(T, source, dest, rank):
    cp = tl.decomposition.CP(
        tol=5e-6, rank=rank, init="random", verbose=1, normalize_factors=True
    )
    factors = cp.fit_transform(T)

    weights, (A, B, C) = factors  # A: mode-0, B: mode-1, C: mode-2 (time)

    order = np.argsort(-np.abs(weights))  # print(weights)
    # exit()

    weights = weights[order]
    A = A[:, order]
    B = B[:, order]
    C = C[:, order]

    T_reconstructed_1 = tl.cp_to_tensor(factors)
    err_1 = tl.norm(T - T_reconstructed_1) / tl.norm(T)
    low_end = 5
    mid_end = 15

    plt.plot(T[source, dest, :], "--", alpha=0.5, label="original")
    plt.plot(T_reconstructed_1[source, dest, :], "--", alpha=0.5, label="reconstructed")
    idx_low = slice(0, low_end)

    component_low = np.sum(
        weights[idx_low] * A[source, idx_low] * B[dest, idx_low] * C[:, idx_low], axis=1
    )
    plt.plot(component_low, label="component-low")
    idx_mid = slice(low_end, mid_end)

    component_mid = np.sum(
        weights[idx_mid] * A[source, idx_mid] * B[dest, idx_mid] * C[:, idx_mid], axis=1
    )

    plt.plot(component_mid, label="component-mid")
    idx_high = slice(mid_end, rank_1)

    component_high = np.sum(
        weights[idx_high] * A[source, idx_high] * B[dest, idx_high] * C[:, idx_high],
        axis=1,
    )
    plt.plot(component_high, label="component-high")
    plt.title(f"Relative error: {err_1:.2f}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    T = np.load("data/abiline_ten.npy")
    T = T[:, :, :10000]
    for i in range(12):
        for j in range(12):
            T[i, j, :] = normalize_tensor(T[i, j, :], "minmax")

    np.random.seed(4)
    source, dest = np.random.randint(0, 11), np.random.randint(0, 11)

    # cp_largest_comp(T, source, dest, 4)
    # tucker_largest_comp(T, source, dest, 4)
    cp_banded_ranks(T, source, dest, 4)
