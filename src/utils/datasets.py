from typing import Dict, Optional, Tuple
import numpy as np
import numpy.typing as npt
from utils.anomaly_injector import inject_random_shapes, inject_random_spikes_normal
from utils.tensor_processing import preprocess


import numpy as np


def get_test_dataset() -> Tuple[npt.NDArray, dict]:
    T = np.load("data/abiline_ten.npy").astpye("float64")
    start = 15000
    end = start + 4500
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


def get_train_dataset() -> Tuple[npt.NDArray, dict]:
    T = np.load("data/abiline_ten.npy").astype("float64")
    start = 0
    end = 4500
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


def create_spike_dataset_train() -> Tuple[npt.NDArray, npt.NDArray, None, dict]:
    T, data_param = get_train_dataset()
    n_spikes = 1000
    amplitude_factor = 6

    T, L = inject_random_spikes_normal(
        T, amplitude_factor=amplitude_factor, n_spikes=n_spikes
    )
    params = {"amplitude_factor": amplitude_factor, "n_spikes": n_spikes}

    return T, L, None, params | data_param


def create_event_dataset_train() -> Tuple[npt.NDArray, npt.NDArray, list, dict]:
    T, data_param = get_train_dataset()

    params = {
        "start_min": 20,
        "start_max": 4000,
        "min_duration": 10,
        "max_duration": 50,
        "n_shapes": 100,
        "amplitude_factor": 6,
    }
    T, L, events = inject_random_shapes(T, **params)

    return T, L, events, params | data_param


def create_spike_dataset_test() -> Tuple[npt.NDArray, npt.NDArray, None, dict]:
    T, data_param = get_test_dataset()
    n_spikes = 1000
    amplitude_factor = 6

    T, L = inject_random_spikes_normal(
        T, amplitude_factor=amplitude_factor, n_spikes=n_spikes
    )
    params = {"amplitude_factor": amplitude_factor, "n_spikes": n_spikes}

    return T, L, None, params | data_param


def create_event_dataset_test() -> Tuple[npt.NDArray, npt.NDArray, list, dict]:
    T, data_param = get_test_dataset()

    params = {
        "start_min": 20,
        "start_max": 4000,
        "min_duration": 10,
        "max_duration": 50,
        "n_shapes": 10,
        "amplitude_factor": 6,
    }
    T, L, events = inject_random_shapes(T, **params)

    return T, L, events, params | data_param
