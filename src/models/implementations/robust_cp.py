from copy import deepcopy
import numpy as np
import tensorly as tl
import scipy.linalg as sla

from utils.utils import detect_anomalies_soft


def cp_als_robust(tensor, rank, n_iter=50, tol=1e-4, threshold=None):
    shape = tensor.shape
    n_modes = len(shape)
    # factors = [np.random.rand(shape[i], rank) for i in range(n_modes)]
    (_, factors) = tl.decomposition.CP(
        rank=rank, tol=tol, n_iter_max=n_iter, init="random"
    ).fit_transform(tensor)
    prev_error = 0
    M = deepcopy(tensor)
    S = np.zeros_like(M)

    for iteration in range(n_iter):
        for n in range(n_modes):
            relevant_factors = [factors[i] for i in range(n_modes) if i != n]
            kr_product = tl.tenalg.khatri_rao(relevant_factors)

            gram_matrices = [np.dot(f.T, f) for f in factors]

            hadamard_product = np.ones((rank, rank))
            for i, g in enumerate(gram_matrices):
                if i != n:
                    hadamard_product *= g

            unfolded = tl.unfold(M, mode=n)
            rhs = np.dot(unfolded, kr_product).T
            X = factors[n].T
            R = rhs - np.matmul(hadamard_product, X)
            P = R.copy()
            rs_old = np.sum(R**2, axis=0)
            for _ in range(n_iter):
                Ap = hadamard_product @ P
                denom = np.sum(P * Ap, axis=0) + 1e-12
                alpha = rs_old / denom
                X += alpha * P
                R -= alpha * Ap
                rs_new = np.sum(R**2, axis=0)
                if np.all(rs_new < tol**2):
                    break
                P = R + (rs_new / rs_old) * P
                rs_old = rs_new
            factors[n] = X.T

        if iteration == 0 or iteration % 4 == 0:
            X_tensor = tl.cp_to_tensor((None, factors))
            residuals = tensor - X_tensor
            S = (
                detect_anomalies_soft(residuals, threshold=threshold)
                if threshold
                else 0
            )
            M = tensor - S

            error = np.linalg.norm(residuals) / tl.norm(M)
            diff = abs(prev_error - error)
            if iteration > 0 and abs(prev_error - error) < tol:
                print(f"Converged at iteration {iteration}")
                break

            prev_error = error

    return (None, factors), S


def cp_als_robust_solve(tensor, rank, n_iter=50, tol=1e-4, threshold=None):
    shape = tensor.shape
    n_modes = len(shape)

    # Initialize factors
    (_, factors) = tl.decomposition.CP(
        rank=rank, tol=tol, n_iter_max=n_iter, init="random"
    ).fit_transform(tensor)

    prev_error = 0
    M = deepcopy(tensor)
    S = np.zeros_like(M)

    for iteration in range(n_iter):
        for n in range(n_modes):
            # Calculate the Khatri-Rao product and Gram matrices
            gram_matrices = [np.dot(f.T, f) for f in factors]

            # Form the Hadamard product of Gram matrices (The LHS matrix)
            hadamard_product = np.ones((rank, rank))
            for i, g in enumerate(gram_matrices):
                if i != n:
                    hadamard_product *= g

            # Calculate the RHS: unfolded_M * Khatri-Rao product
            relevant_factors = [factors[i] for i in range(n_modes) if i != n]
            kr_product = tl.tenalg.khatri_rao(relevant_factors)
            unfolded = tl.unfold(M, mode=n)
            rhs = np.dot(unfolded, kr_product)  # Shape: (shape[n], rank)

            # Solve the system: factors[n] * hadamard_product = rhs
            # We solve (hadamard_product) * (factors[n].T) = rhs.T
            # factors[n] = np.linalg.solve(hadamard_product, rhs.T).T
            factors[n] = np.linalg.solve(hadamard_product, rhs.T).T

        # Robust update logic
        if iteration == 0 or iteration % 4 == 0:
            X_tensor = tl.cp_to_tensor((None, factors))
            residuals = tensor - X_tensor
            if threshold != 0:
                S = detect_anomalies_soft(residuals, threshold=threshold)
                M = tensor - S

            error = np.linalg.norm(residuals)
            if iteration > 0 and abs(prev_error - error) < tol:
                break
            prev_error = error

    return (None, factors), S


def cp_als_robust_cholesky(tensor, rank, n_iter=50, tol=1e-4, threshold=None):
    shape = tensor.shape
    n_modes = len(shape)

    (_, factors) = tl.decomposition.CP(
        rank=rank, tol=tol, n_iter_max=n_iter, init="random"
    ).fit_transform(tensor)

    prev_error = 0
    M = deepcopy(tensor)
    S = np.zeros_like(M)

    for iteration in range(n_iter):
        for n in range(n_modes):
            # Compute Gram matrices and Hadamard product
            gram_matrices = [np.dot(f.T, f) for f in factors]
            hadamard_product = np.ones((rank, rank))
            for i, g in enumerate(gram_matrices):
                if i != n:
                    hadamard_product *= g

            # Add small ridge to ensure positive definiteness (regularization)
            hadamard_product += np.eye(rank) * 1e-10

            # Calculate RHS
            relevant_factors = [factors[i] for i in range(n_modes) if i != n]
            kr_product = tl.tenalg.khatri_rao(relevant_factors)
            rhs = np.dot(tl.unfold(M, mode=n), kr_product)

            # Cholesky Decomposition: A = L * L.T
            L = np.linalg.cholesky(hadamard_product)

            # Solve L*y = rhs.T and then L.T * x = y
            # We use scipy.linalg.cho_solve for efficiency
            factors[n] = sla.cho_solve((L, True), rhs.T).T

        # Robust update logic
        if iteration == 0 or iteration % 4 == 0:
            X_tensor = tl.cp_to_tensor((None, factors))
            residuals = tensor - X_tensor
            S = (
                detect_anomalies_soft(residuals, threshold=threshold)
                if threshold
                else 0
            )
            M = tensor - S

            error = np.linalg.norm(residuals)
            if iteration > 0 and abs(prev_error - error) < tol:
                break
            prev_error = error

    return (None, factors), S


def robust_cp(X, rank, n_iter=50, tol=1e-6, verbose=False, init="svd", threshold=None):
    """
    Robust CP decomposition (CP + anomaly separation)

    Parameters
    ----------
    X : ndarray
        Input tensor
    rank : int
        CP rank
    n_iter : int
        Max iterations
    tol : float
        Convergence tolerance
    n_anomalies : int
        Number of anomalies to detect

    Returns
    -------
    weights : ndarray
    factors : list of factor matrices
    S : sparse anomaly tensor
    """

    # Initial CP decomposition
    cp_tensor = tl.decomposition.parafac(X, rank=rank, n_iter_max=10, init=init)
    weights, factors = cp_tensor

    M = X.copy()
    old_error = 1e20
    S = np.zeros_like(M)

    for iteration in range(n_iter):

        cp_tensor = tl.decomposition.parafac(
            M, rank=rank, n_iter_max=10, init=(weights, factors)
        )
        weights, factors = cp_tensor

        # Reconstruction
        X_hat = tl.cp_to_tensor((weights, factors))
        residuals = X - X_hat

        error = np.linalg.norm(residuals) / tl.norm(M)
        diff = abs(old_error - error)

        S = detect_anomalies_soft(residuals, threshold=threshold)

        # Update clean tensor
        M = X - S

        if verbose and iteration % 10 == 0:
            print("diff:", diff)
            print("iteration:", iteration)

        if diff < tol:
            if verbose:
                print("reached tol")
            break

        old_error = error

    return (weights, factors), S
