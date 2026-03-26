import gc
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly import tenalg


# from src.utils.anomaly_injector import inject_fiber_anomaly


T = np.load("data/EU_email.npy")
T = T[:, :, :2000]
nt = len(T[1, 1, :])
nn = len(T[:, 1, 1])


rand = np.sqrt(1 / 50) * np.random.randn(50, 200)

P = np.zeros((50, 50, 2000))
for i in range(2000):
    P[:, :, i] = (rand @ T[:, :, i]) @ rand.T

T = P


# # Create similarity matrix (temporal)
# print("creating temporal similarity matrix...")
# flat_T = T.reshape(-1, nt)
# norms = np.linalg.norm(flat_T, axis=0, keepdims=True)
# flat_T = flat_T / (norms + 1e-8)
# S = flat_T.T @ flat_T
# S /= np.sqrt(np.sum(S**2))
# np.fill_diagonal(S, 0)
# L = np.diag(np.sum(S, axis=1)) - S
#
#
# T = tl.tensor(T)
#
#
# ## Create truncated Laplacian
#
# print("creating tuncated Laplacian...")
# print(f"dims: {T.shape}")
# k = 500
# D, F = np.linalg.eigh(L)
# idx = np.flip(np.argsort(np.abs(D)))
# # idx = np.argsort(np.abs(D))
# D_sorted = D[idx]
# F_sorted = F[:, idx]
# Fk = F_sorted[:, :k]
# Fk = tl.tensor(Fk)
#
# print(np.size(T) * 8 / 1e9)
#
# # Create projected tensor
# I = np.eye(Fk.shape[0])
# T_prime = tenalg.mode_dot(T, Fk @ Fk.T, 2)
#
# # temp = tl.base.unfold(T, 2)
# del T
# gc.collect()
# # non_zero_column_indices = tl.where(tl.sum(tl.abs(temp), axis=0) != 0)[0]
# # temp = temp[:, non_zero_column_indices]
# # temp = temp[:, :1000]
#
# plt.imshow(np.sum(T_prime, axis=2))
# plt.show()
# exit()
#
# plt.plot(np.sum(np.sum(T_prime, axis=0), axis=0))
# plt.show()
# exit()
#
#
# _, S, _ = np.linalg.svd(temp, full_matrices=False)
# plt.semilogy(S)
# plt.show()
#
#
# exit()
## Decompose and reconsturct
factors = tl.decomposition.Tucker(
    (50, 50, 50), 2, verbose=10, init="random"
).fit_transform(T)
T_reconstructed = tl.tucker_to_tensor(factors)
err = np.abs(T_reconstructed - T)
print(np.linalg.norm(err) / np.linalg.norm(T))


exit()

del T_reconstructed
del T
gc.collect()


L = np.load("data/DDos_labels.npy")
# Labels = Labels[:100, :100, :]
L = L == 1
err = (err - err.min()) / (err.max() - err.min() + 1e-8)
err = err > 0.2  # A more standard threshold


TP = np.sum(L & err)
TN = np.sum(~L & ~err)
FP = np.sum(~L & err)
FN = np.sum(L & ~err)

total = L.size

print(f"Accuracy: {(TP + TN) / total:.2%}")
print(f"Precision: {TP / (TP + FP) if (TP + FP) > 0 else 0:.2%}")
print(f"Recall: {TP / (TP + FN) if (TP + FN) > 0 else 0:.2%}")
print("-" * 20)
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"True Negatives:  {TN}")
print(f"True Positives:  {TP}")
exit()


## Plot errors
norms = np.zeros(nt)
for i in range(nt):
    norms[i] = np.linalg.norm(err[:, :, i], "fro")
plt.title(f"time step norms, rank={k}")
plt.plot(norms)
plt.xlabel("time")
plt.ylabel("norm")
plt.legend()
plt.show()
exit()


A_h = flat_T.T - Uk @ (Uk.T @ flat_T.T)
A_h = A_h.reshape((nn, nn, nt))
A_l = Uk @ (Uk.T @ flat_T.T)
A_l = A_l.reshape((nn, nn, nt))


norms_h = np.zeros(nt)
norms_l = np.zeros(nt)
for i in range(nt):
    norms_h[i] = np.linalg.norm(A_h[:, :, i], "fro")
    norms_l[i] = np.linalg.norm(A_l[:, :, i], "fro")

plt.title(f"time step norms, rank={k}")
plt.plot(norms_l, label="projected")

plt.plot(norms_h, label="orthogonal compliment")
plt.xlabel("time")
plt.ylabel("norm")
plt.legend()
plt.show()
