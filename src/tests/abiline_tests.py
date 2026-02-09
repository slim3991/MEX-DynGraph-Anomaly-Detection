import matplotlib.pyplot as plt
import numpy as np
import tensorly as tl
from incremental_svd import IncrementalSVD

rank_space = 12
rank_time = 100

T = np.load("./data/abiline_ten.npy")
T = tl.tensor(T)


test_size = 1000
train_size = 200

test_end_idx = train_size + test_size


## initialize the svd

testTen = T[:, :, :train_size]

TTen1 = tl.unfold(testTen, 0)
TTen2 = tl.unfold(testTen, 1)
TTen3 = tl.unfold(testTen, 2).T
del testTen

iSVD1 = IncrementalSVD(rank=rank_space, forgetting_factor=0.9)
iSVD2 = IncrementalSVD(rank=rank_space, forgetting_factor=0.9)
iSVD3 = IncrementalSVD(rank=rank_time, forgetting_factor=0.9)

iSVD1.fit(TTen1)
iSVD2.fit(TTen2)
iSVD3.fit(TTen3)
print(iSVD3)

errors = np.zeros((train_size + test_size, 1))
for i in range(train_size, train_size + test_size):
    new = T[:, :, i]

    projection1 = iSVD1.U @ (iSVD1.U.T @ new)
    err1 = new - projection1
    iSVD1.increment(new)

    projection2 = iSVD2.U @ (iSVD2.U.T @ new.T)
    err2 = new - projection2
    iSVD2.increment(new.T)

    new = tl.reshape(new, (*new.shape, 1))
    new = tl.unfold(new, 2)
    new = tl.reshape(new, new.shape[1])
    projection3 = iSVD3.U @ (iSVD3.U.T @ new)
    err3 = new - projection3
    iSVD3.increment(new)

    errors[i] = tl.norm(err1) ** 2 + tl.norm(err2) ** 2 + tl.norm(err3) ** 2

    if i % 50 == 0:
        print(i)

plt.plot(np.sqrt(errors))
plt.show()
