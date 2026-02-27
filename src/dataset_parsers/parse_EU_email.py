import numpy as np

# Load data (equivalent to readmatrix)
A = np.loadtxt("./raw_data/email-Eu-temporal.txt")
print(np.unique(A[:, 0]).__len__())

n = 200


# Convert seconds → hours and round
A[:, 2] = np.round(A[:, 2] / (60**2))

A[:, 0] += 1
A[:, 1] += 1

values, counts = np.unique(A[:, 0], return_counts=True)
top_n_val = values[np.argsort(counts)[-n:]]
A = A[np.isin(A[:, 0], top_n_val)]

values, counts = np.unique(A[:, 1], return_counts=True)
top_n_val = values[np.argsort(counts)[-n:]]
A = A[np.isin(A[:, 1], top_n_val)]
print(A.shape)

unique_vals, inverse = np.unique(A[:, 0], return_inverse=True)
A[:, 0] = inverse + 1
unique_vals, inverse = np.unique(A[:, 1], return_inverse=True)
A[:, 1] = inverse + 1
# exit()

# Keep first 16000 rows
# A = A[:16000, :]


# Start time
start = np.min(A[:, 2])

# Unique times
times = np.unique(A[:, 2])
# A[:, 2] = times - start
nt = len(times)

# Number of nodes
nNodes = int(max(A[:, 0].max(), A[:, 1].max()))
print(f"nt: {nt}, nn: {nNodes}")

# Estimated size (float32 = 4 bytes)
est_size_gb = nt * nNodes * nNodes * 4 / 1e9
print(f"est size: {est_size_gb:.2f}Gb")
# exit()

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
