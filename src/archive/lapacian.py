import gc
import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly import tenalg

from models.incremental_svd import IncrementalSVD


T = np.load("data/DDos_data.npy")
# T = T[:100, :100, :]
print(T.shape)

# T = T[:, :, :500]
nt = len(T[1, 1, :])
nn = len(T[:, 1, 1])


rank_1 = 10
rank_2 = 10
rank_time = 20
k = rank_time


# flat_T = T.reshape(-1, nt)

# S = flat_T.T @ flat_T  # Covariance matrix
# np.fill_diagonal(S, 0)

# L = np.sum(S, axis=1) - S

# D, F = np.linalg.eigh(L)
# idx = np.argsort(np.abs(D))
# D_sorted = D[idx]
# F_sorted = F[:, idx]
# Fk = F_sorted[:, :k]

# plt.pcolor(F_sorted)
# plt.plot(D_sorted)
# plt.show()
# exit()

T = tl.tensor(T)

# temp = tl.base.unfold(T, 2)
# non_zero_column_indices = tl.where(tl.sum(tl.abs(temp), axis=0) != 0)[0]
# temp = temp[:, non_zero_column_indices]

# U, S, V = np.linalg.svd(Fk.T @ temp, full_matrices=True)
# Sigma = np.zeros((U.shape[1], V.shape[0]))
# np.fill_diagonal(Sigma, S)

# reconst = (Fk @ U) @ Sigma @ V
# plt.imshow(np.log(temp + 1))
# plt.colorbar()
# # plt.semilogy(S)
# # plt.grid()
# plt.show()
# exit()


iSVD1 = IncrementalSVD(rank=rank_1, forgetting_factor=1)
iSVD2 = IncrementalSVD(rank=rank_2, forgetting_factor=1)
iSVD3 = IncrementalSVD(rank=rank_time, forgetting_factor=1)

print("fitting iSVD1")
T_unf = tl.base.unfold(T, 0)
iSVD1.fit(T_unf[:, :100])
for i in range(100, T_unf.shape[1], 100):
    print(i)
    iSVD1.increment(T_unf[:, i : i + 100])
U = iSVD1.U
del iSVD1

print("fitting iSVD2")
T_unf = tl.base.unfold(T, 1)
iSVD2.fit(T_unf[:, :100])
for i in range(100, T_unf.shape[1], 100):
    print(i)
    iSVD2.increment(T_unf[:, i : i + 100])
V = iSVD2.U
del iSVD2


print("fitting iSVD3")
T_unf = tl.base.unfold(T, 2)
iSVD3.fit(T_unf[:, :100])
for i in range(100, T_unf.shape[1], 100):
    print(i)
    iSVD3.increment(T_unf[:, i : i + 100])
W = iSVD3.U
del iSVD3
# W = Fk @ W_hat

core = tl.tenalg.multi_mode_dot(T, [U.T, V.T, W.T], modes=[0, 1, 2])

T_hat = tl.tenalg.multi_mode_dot(core, [U, V, W], modes=[0, 1, 2])


err = np.linalg.norm(T_hat - T)
del T
del T_hat


gc.collect()

Labels = np.load("data/DDos_labels.npy")
# Labels = Labels[:100, :100, :]
Labels = Labels == 1
err = (err - err.min()) / (err.max() - err.min() + 1e-8)
err = err > 0.5  # A more standard threshold


TP = np.sum(Labels & err)
TN = np.sum(~Labels & ~err)
FP = np.sum(~Labels & err)
FN = np.sum(Labels & ~err)

total = Labels.size

print(f"Accuracy: {(TP + TN) / total:.2%}")
print(f"Precision: {TP / (TP + FP) if (TP + FP) > 0 else 0:.2%}")
print(f"Recall: {TP / (TP + FN) if (TP + FN) > 0 else 0:.2%}")
print("-" * 20)
print(f"False Positives: {FP}")
print(f"False Negatives: {FN}")
print(f"True Negatives:  {TN}")
print(f"True Negatives:  {TP}")

# plt.hist(np.abs(np.reshape(err, -1)), bins=220)
# plt.show()

exit()
norms = np.zeros((nt, 1))
for i in range(nt):
    print(i)
    norms[i] = tl.norm(err[:, :, i])

plt.plot(norms)
plt.show()
