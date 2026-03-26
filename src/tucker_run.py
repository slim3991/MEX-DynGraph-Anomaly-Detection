import numpy as np
import tensorly as tl
from decomp_results import evaluate, tucker_find_best_rank_grid_search

from utils.tensor_processing import de_anomalize_tensor, normalize_tensor
from utils.anomaly_injector import *
from utils.model_eval import *


def main():
    T = np.load("data/abiline_ten.npy")
    T_train = T[:, :, :5000]
    T_test = T[:, :, 10_000:15_000]

    def decomp_recomp(T, rank):
        tucker_T = tl.decomposition.tucker(T, rank=rank)
        recomp = tl.tucker_to_tensor(tucker_T)
        return recomp

    # find_best_rank_grid_search(
    #     T_train, events_anomalies=True, recomp_func=decomp_recomp
    # )
    # find_best_rank_grid_search(
    #     T_train, events_anomalies=False, recomp_func=decomp_recomp
    # )
    # exit()
    event_rank = (11, 9, 17)
    spikes_rank = (11, 10, 30)

    evaluate(
        T_train,
        event_rank,
        recomp_func=decomp_recomp,
        events_based=True,
        test="Tucker train events",
    )
    evaluate(
        T_test,
        event_rank,
        recomp_func=decomp_recomp,
        events_based=True,
        test="Tucker test events",
    )
    evaluate(
        T_train,
        spikes_rank,
        recomp_func=decomp_recomp,
        events_based=False,
        test="Tucker train spikes",
    )
    evaluate(
        T_test,
        spikes_rank,
        recomp_func=decomp_recomp,
        events_based=False,
        test="Tucker test spikes",
    )
    exit()


if __name__ == "__main__":
    main()
