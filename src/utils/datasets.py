from typing import Dict
import numpy as np
from utils.anomaly_injector import inject_random_shapes, inject_random_spikes_normal
from utils.tensor_processing import preprocess


import numpy as np


def get_dataset_std():
    T = np.load("data/abiline_ten.npy")
    start = 0
    end = 5000
    T = T[:, :, start:end]
    preprocess_rank = 20
    keep_percentile = 95
    alpha = 0.4
    T = preprocess(
        T, rank=preprocess_rank, keep_percentile=keep_percentile, alpha=alpha
    )
    params = {
        "start": start,
        "end": end,
        "preprocess_rank": preprocess_rank,
        "keep_percentile": keep_percentile,
        "alpha": alpha,
    }

    return T, params


def create_spike_dataset_std():
    T, data_param = get_dataset_std()
    n_spikes = 1000
    amplitude_factor = 10

    T, L = inject_random_spikes_normal(
        T, amplitude_factor=amplitude_factor, n_spikes=n_spikes
    )
    params = {"amplitude_factor": amplitude_factor, "n_spikes": n_spikes}

    return T, L, params | data_param


def create_event_dataset_std():
    T, data_param = get_dataset_std()

    params = {
        "start_min": 20,
        "start_max": 4000,
        "min_duration": 10,
        "max_duration": 100,
        "n_shapes": 20,
        "amplitude_factor": 10,
    }
    T, L, events = inject_random_shapes(T, **params)

    return T, L, events, params | data_param
