import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh
import tensorly as tl
from tensorly.tenalg import unfolding_dot_khatri_rao
from utils.tensor_processing import make_mode_knn
from utils.utils import detect_anomalies_soft, global_cg_sylvester


def global_cg_sylvester(A, B, C, max_iter=1000, tol=1e-6, verbose=False):
    A = sparse.csr_matrix(A)
    B = sparse.csc_matrix(B)

    # Precompute diagonal preconditioner
    dA = A.diagonal()
    dB = B.diagonal()

    # Avoid division by zero
    denom = dA[:, None] + dB[None, :]
    denom[denom == 0] = 1e-12

    def apply_preconditioner(R):
        return R / denom

    X = np.zeros_like(C)

    R = C.copy()
    Z = apply_preconditioner(R)
    P = Z.copy()

    rz_inner = np.vdot(R, Z).real
    rz0 = rz_inner  # for relative stopping

    for k in range(max_iter):
        W = A @ P + P @ B

        denom_cg = np.vdot(P, W).real
        if denom_cg <= 1e-16:
            # if verbose:
            #     print(f"Breakdown at iter {k}")
            break

        alpha = rz_inner / denom_cg

        X += alpha * P
        R -= alpha * W

        # Check convergence (relative to initial residual)
        if np.vdot(R, R).real <= (tol**2) * np.vdot(C, C).real:
            if verbose:
                print(f"Converged in {k+1} iterations")
            return X

        Z = apply_preconditioner(R)

        rz_new = np.vdot(R, Z).real
        beta = rz_new / rz_inner

        P = Z + beta * P
        rz_inner = rz_new

        # if verbose and k % 10 == 0:
        #     res = np.sqrt(np.vdot(R, R).real)
        #     print(f"iter {k}, residual {res:.2e}")

    return X


def graph_regularized_als(
    tensor,
    rank,
    laps,
    lmbda=(0.1, 0.1, 0.1),
    n_iter=20,
    verbose=False,
    threshold=None,
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

        res = M - tl.cp_to_tensor((weights, factors))
        E = detect_anomalies_soft(res, threshold=threshold)
        M = tensor - E

        err = np.linalg.norm(res)
        delta = np.abs(err - old_err) / (old_err + 1e-12)

        if verbose:
            print(f"Iteration {i}, Factor Change: {delta:.6f}")

        if delta < tol:
            if verbose:
                print("Converged.")hosvd
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
            print(L.shape, rank)
            vals, vecs = eigsh(L, k=rank, which="SM")
            factors.append(vecs)
        else:
            # If no graph exists for this mode, we default to random
            # or identity if you want a 'blank' slate
            factors.append(np.random.standard_normal((mode_shapes[i], rank)))

    # Return in the same CP format as your function (weights, factors)
    return (np.ones(rank), factors)
