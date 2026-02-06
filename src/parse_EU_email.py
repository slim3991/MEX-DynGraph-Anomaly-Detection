import numpy as np

# Load data (equivalent to readmatrix)
A = np.loadtxt("./raw_data/email-Eu-temporal.txt")

# MATLAB: min(A(:,2))
print(np.min(A[:, 1]))

# Convert seconds → hours and round
A[:, 2] = np.round(A[:, 2] / (60**2))

# MATLAB uses 1-based indexing; Python uses 0-based
# This shift preserves MATLAB-style node numbering
A[:, 0] += 1
A[:, 1] += 1

# Keep first 16000 rows
A = A[:16000, :]

# Start time
start = np.min(A[:, 2])

# Unique times
times = np.unique(A[:, 2])
times_minus_start = times - start
nt = len(times)

# Number of nodes
nNodes = int(max(A[:, 0].max(), A[:, 1].max()))

# Estimated size (float32 = 4 bytes)
est_size_gb = nt * nNodes * nNodes * 4 / 1e9
print(f"est size: {est_size_gb:.2f}Gb")

# Initialize tensor
T = np.zeros((nNodes, nNodes, nt), dtype=np.float32)

# Fill tensor
for i, t in enumerate(times):
    Ap = A[A[:, 2] == t, :2].astype(int)
    # Convert to 0-based indexing for Python arrays
    src = Ap[:, 0] - 1
    dst = Ap[:, 1] - 1
    T[src, dst, i] = 1


np.save("./data/EU_email.npy", T)
