import numpy as np
from scipy import sparse


def detect_anomalies_soft(res, threshold: float | None = None):
    if threshold is None:
        # abs_res = np.abs(res)
        # sigma = np.median(abs_res[abs_res < np.percentile(abs_res, 50)]) / 0.6745
        sigma = np.median(np.abs(res)) / 0.6745
        # lam = 2.5 * sigma
        lam = sigma * np.sqrt(2 * np.log(res.size))
    else:
        lam = threshold
    E = np.sign(res) * np.maximum(np.abs(res) - lam, 0)
    return E


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
