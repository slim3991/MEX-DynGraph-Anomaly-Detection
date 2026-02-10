import numpy as np
import matplotlib.pyplot as plt


T = np.load("data/EU_email.npy")
nt = len(T[1, 1, :])
nn = len(T[:, 1, 1])
flat_T = T.reshape(-1, nt)

S = flat_T.T @ flat_T
np.fill_diagonal(np.log(1 + S), 0)
L = np.diag(np.sum(S, axis=1)) - S


k = 12

D, U = np.linalg.eigh(L)
idx = np.argsort(np.abs(D))
D_sorted = D[idx]
U_sorted = U[:, idx]
Uk = U[:, :k]


A_h = flat_T.T - Uk @ (Uk.T @ flat_T.T)
A_h = A_h.reshape((nn, nn, nt))
A_l = Uk @ (Uk.T @ flat_T.T)
A_l = A_l.reshape((nn, nn, nt))


norms_h = np.zeros(nt)
norms_l = np.zeros(nt)
for i in range(nt):
    norms_h[i] = np.linalg.norm(A_h[:, :, i], "fro")
    norms_l[i] = np.linalg.norm(A_l[:, :, i], "fro")

plt.plot(norms_l, label="l")
plt.plot(norms_h, label="h")
plt.legend()
plt.show()
