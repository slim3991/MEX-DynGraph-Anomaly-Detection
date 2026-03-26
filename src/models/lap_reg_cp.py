import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh
import tensorly as tl
from tensorly.tenalg import unfolding_dot_khatri_rao
from utils.tensor_processing import make_mode_knn
from utils.utils import global_cg_sylvester


def detect_anomalies(S, factors, epsilon):
    M_hat = tl.cp_to_tensor(factors)

    residuals = S - M_hat

    flat_residuals = residuals.flatten() ** 2

    if epsilon >= len(flat_residuals):
        threshold = 0
    else:
        threshold = np.partition(flat_residuals, -epsilon)[-epsilon]

    E = np.where(residuals**2 >= threshold, S, 0)  # TODO: Verify this

    anomaly_indices = np.argwhere(residuals >= threshold)

    return E, anomaly_indices


def detect_anomalies_soft(T, T_hat):
    res = T - tl.cp_to_tensor(T_hat)
    abs_res = np.abs(res)
    # sigma = np.median(abs_res[abs_res < np.percentile(abs_res, 50)]) / 0.6745

    sigma = np.median(np.abs(res)) / 0.6745
    # lam = 2.5 * sigma
    lam = sigma * np.sqrt(2 * np.log(res.size))

    E = np.sign(res) * np.maximum(np.abs(res) - lam, 0)
    return E


def graph_regularized_als(
    tensor,
    rank,
    laps,
    lmbda=(0.1, 0.1, 0.1),
    n_iter=20,
    n_E=1000,
    verbose=False,
):
    weights, factors = tl.decomposition.parafac(
        tensor, rank=rank, n_iter_max=10, init="random"
    )

    M = tensor.copy()
    E = np.zeros_like(M)

    old_err = 1e10
    tol = 1e-4  # Convergence threshold

    for i in range(n_iter):
        if verbose:
            print(f"Iteration {i}...")

        # Update factors A (0), B (1), C (2)
        for mode in range(3):
            idx = [m for m in range(3) if m != mode]
            G1 = tl.dot(factors[idx[0]].T, factors[idx[0]])
            G2 = tl.dot(factors[idx[1]].T, factors[idx[1]])

            S = G1 * G2
            S = S * (weights[:, None] * weights[None, :])

            mttkrp = unfolding_dot_khatri_rao(M, (weights, factors), mode=mode)

            # 3. Solve Regularized System: (λL)X + X(S) = MTTKRP
            if laps is None or laps[mode] is None:
                eps = 1e-8
                factors[mode] = np.linalg.solve(S + eps * np.eye(rank), mttkrp.T).T
            else:
                factors[mode] = global_cg_sylvester(
                    lmbda[mode] * laps[mode],
                    S,
                    mttkrp,
                    max_iter=1000,
                    verbose=verbose,
                    tol=1e-4,
                )
        for r in range(rank):
            norm = 1.0
            for mode in range(3):
                col_norm = np.linalg.norm(factors[mode][:, r]) + 1e-12
                factors[mode][:, r] /= col_norm
                norm *= col_norm
            weights[r] *= norm

        # Update Anomaly component
        if n_E > 0:
            # We create a temporary CPTensor for the anomaly detection function
            current_cp = (weights, factors)
            E = detect_anomalies_soft(tensor, current_cp)
            M = tensor - E

        err = np.linalg.norm(M - tl.cp_to_tensor((weights, factors)))
        delta = np.abs(err - old_err) / (old_err + 1e-12)

        if verbose:
            print(f"Iteration {i}, Factor Change: {delta:.6f}")

        if delta < tol:
            if verbose:
                print("Converged.")
            break

        old_err = err

    return (weights, factors), E


def estimate_from_laps(laps, rank, mode_shapes):
    """
    Creates an estimate decomposition based only on the Graph Laplacians.
    """
    factors = []

    for i, L in enumerate(laps):
        if L is not None:
            # Get the smallest eigenvectors (bottom of the spectrum)
            # These are the smoothest signals on the graph
            vals, vecs = eigsh(L, k=rank, which="SM")
            factors.append(vecs)
        else:
            # If no graph exists for this mode, we default to random
            # or identity if you want a 'blank' slate
            factors.append(np.random.standard_normal((mode_shapes[i], rank)))

    # Return in the same CP format as your function (weights, factors)
    return (np.ones(rank), factors)
