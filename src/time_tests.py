from itertools import product
import re
import numpy as np
import tensorly as tl

from utils.anomaly_injector import inject_random_spikes_normal
from utils.metrics import compute_metrics_with_optimal_threshold
from utils.tensor_processing import normalize_tensor, preprocess
from models.implementations.robust_cp import (
    cp_als_robust_solve,
    cp_als_robust_cholesky,
    cp_als_robust,
)

T = np.load("data/abiline_ten.npy")

T = T[:, :, :10000]
# T = T[:, :, 10_000:15_000]
for i, j in product(range(12), repeat=2):
    T[i, j, :] = normalize_tensor(T[i, j, :], "minmax")

# T = normalize_tensor(T, "minmax")
source, dest = np.random.randint(0, 11), np.random.randint(0, 11)
# source, dest = 5, 8

T = preprocess(T, 20, 96, 0.5)
T, L = inject_random_spikes_normal(T, 10, 1000)


import time
import numpy as np
import matplotlib.pyplot as plt


def plot_rank_scaling(T, ranks=[5, 10, 15, 20, 25, 30], repetitions=3):
    methods = {
        "Original (CG)": cp_als_robust,
        "NP Solve": cp_als_robust_solve,
        "Cholesky": cp_als_robust_cholesky,
    }

    # Dictionary to store results: { method_name: { rank: [times] } }
    perf_data = {name: {rank: [] for rank in ranks} for name in methods}

    for rank in ranks:
        print(f"Benchmarking Rank: {rank}")
        for name, func in methods.items():
            for _ in range(repetitions):
                start = time.perf_counter()
                try:
                    func(T, rank=rank)
                except Exception as e:
                    print(f"Error in {name} at rank {rank}: {e}")
                end = time.perf_counter()
                perf_data[name][rank].append(end - start)

    # --- Plotting ---
    plt.figure(figsize=(10, 6))

    for name in methods:
        means = [np.mean(perf_data[name][r]) for r in ranks]
        stds = [np.std(perf_data[name][r]) for r in ranks]

        plt.errorbar(
            ranks, means, yerr=stds, label=name, capsize=5, marker="o", linestyle="-"
        )

    plt.xlabel("Rank")
    plt.ylabel("Time (seconds)")
    plt.title("CP-ALS Robust Performance by Rank")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()


plot_rank_scaling(T, ranks=[5, 10, 15])
