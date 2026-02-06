import numpy as np
import matplotlib.pyplot as plt
import numpy.typing as npt
import tensorly as tl
import gc
import sparse
from joblib import Memory
from scipy.linalg import svd, expm

memory = Memory(location="./cache", verbose=0)


def perform_decomp(path: str, rank: int) -> npt.NDArray:

    raw_data = np.load(path)
    nn, _, nt = raw_data.shape
    sparse_data = sparse.COO.from_numpy(raw_data.astype(np.float32))
    del raw_data
    gc.collect()

    print(f"Starting Sparse CP on tensor of shape {sparse_data.shape}...")

    weights, factors = tl.decomposition.parafac(
        sparse_data, rank=rank, init="random", tol=10e-6
    )

    time_err = np.zeros(nt)

    for t in range(nt):
        actual_slice = sparse_data[:, :, t].todense()

        A, B, C = factors
        C_t = C[t, :] * weights
        reconstructed_slice = A @ np.diag(C_t) @ B.T

        num = np.linalg.norm(actual_slice - reconstructed_slice)
        den = np.linalg.norm(actual_slice)
        time_err[t] = num / (den + 1e-10)

    return time_err


@memory.cache
def load_tensor_and_compute_singular_vectors(
    npy_path, backend="numpy", svd_full_matrices=False, zero_tol=0.0
):
    tl.set_backend(backend)

    # Load NumPy tensor
    np_tensor = np.load(npy_path)

    # Convert to TensorLy tensor
    tensor = tl.tensor(np_tensor)

    results = {}

    for mode in range(tensor.ndim):
        # Mode-n unfolding
        unfolding = tl.unfold(tensor, mode)
        unfolding_np = tl.to_numpy(unfolding)
        print(unfolding_np.shape)

        # Identify nonzero columns
        col_norms = np.sum(unfolding_np, axis=0)
        kept_columns = np.where(col_norms > 0)[0]

        # Remove zero columns
        cleaned_unfolding = unfolding_np[:, kept_columns]

        if cleaned_unfolding.size == 0:
            # Handle degenerate case
            results[mode] = {
                "unfolding": cleaned_unfolding,
                "U": None,
                "S": None,
                "Vh": None,
                "kept_columns": kept_columns,
            }
            continue

        # SVD
        U, S, Vh = svd(cleaned_unfolding, full_matrices=svd_full_matrices)

        results[mode] = {
            "unfolding": cleaned_unfolding,
            "U": U,
            "S": S,
            "Vh": Vh,
            "kept_columns": kept_columns,
        }

    return results


def graph_density_plot(path):
    data = np.load(path)
    n_el = data.size

    data = np.squeeze(np.sum(data, 2)) / n_el
    plt.pcolormesh(expm(data))
    plt.show()


def main():
    #     graph_density_plot("./data/EU_email.npy")
    #     return
    #     results = load_tensor_and_compute_singular_vectors("./data/EU_email.npy")
    #     plt.semilogy(results[0]["S"], label="mode-1")
    #     plt.semilogy(results[1]["S"], label="mode-2")
    #     plt.semilogy(results[2]["S"], label="mode-3")
    #     plt.legend()
    #     plt.show()
    #     return
    time_err = perform_decomp("./data/EU_email.npy", rank=40)

    # Example post-processing
    print("Per-time-step reconstruction error:")
    print(time_err)
    plt.plot(time_err)
    plt.show()

    print(f"Mean error: {time_err.mean():.4f}")
    print(f"Max error:  {time_err.max():.4f}")


if __name__ == "__main__":
    main()
