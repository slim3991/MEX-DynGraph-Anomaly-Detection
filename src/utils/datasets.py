from typing import Callable, Dict, Literal, Optional, Tuple
import numpy as np
import numpy.typing as npt
from utils.anomaly_injector import (
    inject_DDoS,
    inject_outage,
    inject_random_shapes,
    inject_random_spikes_normal,
)
from utils.tensor_processing import preprocess

type dataset_fetcher = Callable[
    [Literal["train", "test"], Optional[float]], Tuple[npt.NDArray, dict]
]


def get_test_dataset() -> Tuple[npt.NDArray, dict]:
    T = np.load("data/abiline_ten.npy").astype("float64")
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


def _get_dataset(train_test: Literal["train", "test"]) -> Tuple[npt.NDArray, dict]:
    if train_test == "train":
        return get_train_dataset()
    elif train_test == "test":
        return get_test_dataset()
    else:
        raise ValueError("train_test must be 'train' or 'test'")


def create_spike_dataset(
    train_test: Literal["train", "test"], ampf: float = 6
) -> Tuple[npt.NDArray, npt.NDArray, None, dict]:

    T, data_param = _get_dataset(train_test)

    n_spikes = 1000
    amplitude_factor = ampf

    T, L = inject_random_spikes_normal(
        T, amplitude_factor=amplitude_factor, n_spikes=n_spikes
    )
    params = {"amplitude_factor": amplitude_factor, "n_spikes": n_spikes}

    return T, L, None, params | data_param


def create_outage_dataset(
    train_test: Literal["train", "test"],
) -> Tuple[npt.NDArray, npt.NDArray, None, dict]:

    T, data_param = _get_dataset(train_test)
    duration = 40
    n_nodes = 5
    n_events = 10

    L = np.zeros_like(T)
    for _ in range(n_events):
        T, Lp = inject_outage(T, duration=duration, n_nodes=n_nodes)
        L += Lp

    L = np.where(L > 0, 1, 0)

    params = {
        "n_events": n_events,
        "duration": duration,
        "n_nodes": n_nodes,
    }

    return T, L, None, params | data_param


def create_ddos_dataset(
    train_test: Literal["train", "test"], ampf: float = 10
) -> Tuple[npt.NDArray, npt.NDArray, None, dict]:

    T, data_param = _get_dataset(train_test)
    duration = 10
    n_senders = 7
    amplitude_factor = ampf
    n_events = 100

    L = np.zeros_like(T)
    for _ in range(n_events):
        target = np.random.randint(0, T.shape[0])
        T, Lp = inject_DDoS(
            T,
            duration=duration,
            n_senders=n_senders,
            amplitude_factor=amplitude_factor,
            target=target,
        )
        L += Lp

    L = np.where(L > 0, 1, 0)

    params = {
        "n_events": n_events,
        "duration": duration,
        "n_senders": n_senders,
        "amplitude_factor": amplitude_factor,
    }

    return T, L, None, params | data_param


def create_event_dataset(
    train_test: Literal["train", "test"],
    ampf: float = 6,
) -> Tuple[npt.NDArray, npt.NDArray, list, dict]:

    T, data_param = _get_dataset(train_test)
    n_shapes = 100
    params = {
        "start_min": 20,
        "start_max": 4000,
        "min_duration": 10,
        "max_duration": 50,
        "n_shapes": n_shapes,
        "amplitude_factor": ampf,
    }

    T, L, events = inject_random_shapes(T, **params)

    return T, L, events, params | data_param
