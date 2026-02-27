import numpy as np

A = np.loadtxt("raw_data/BEED_Data.csv", delimiter=",", skiprows=1)

print(A.shape)
