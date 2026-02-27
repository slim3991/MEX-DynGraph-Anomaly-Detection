import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl

# factors = tl.decomposition.CP(
#     tol=5e-4, rank=30, init="random", verbose=10
# ).fit_transform(T)

factors = tl.decomposition.non_negative_parafac(T, rank=11, verbose=10)

T_reconstructed = tl.cp_to_tensor(factors)


source, dest = 1, 9
od_rec = T_reconstructed[source, dest, :]
od_orig = T[source, dest, :]

plt.plot(od_rec, label="reconst")
plt.plot(od_orig, label="original")
plt.plot(np.abs(od_rec - od_orig), label="error")
plt.legend()
plt.show()


exit()


for i in range(12):
    for j in range(12):
        od = T_reconstructed[i, j, :]
        if np.max(od) > 100 or np.max(od) < 6:
            continue
        plt.plot(od, label=f"({i},{j})")
plt.show()
