from typing import Callable, Literal, Tuple
import numpy as np
import numpy.typing as npt

from utils.wavelet_denoise import get_synth_signal


def generate_shape(
    duration: int,
    begin_shape: Literal["ramp", "gaussian", "step"],
    end_shape: Literal["ramp", "gaussian", "step"],
    ratios: Tuple[float | int, float | int, float | int],
    amplitude: float | int,
) -> npt.NDArray:
    ratio_sum = sum(ratios)
    ratios = (ratios[0] / ratio_sum, ratios[1] / ratio_sum, ratios[2] / ratio_sum)

    start_len = int(round(duration * ratios[0]))
    middle_len = int(round(duration * ratios[1]))
    end_len = duration - start_len - middle_len  # ensures total matches

    arr = np.zeros(duration)

    def gen_part(shape: str, length: int) -> npt.NDArray:
        if length <= 0:
            return np.array([])
        x = np.linspace(0, 1, length)
        if shape == "ramp":
            return x
        elif shape == "gaussian":
            x = np.linspace(-2, 2, length)
            return np.exp(-(x**2))
        elif shape == "step":
            return np.ones(length)
        else:
            raise ValueError("Not a supported shape")

    arr[:start_len] = gen_part(begin_shape, start_len)
    arr[start_len : start_len + middle_len] = gen_part("step", middle_len)
    arr[start_len + middle_len :] = np.flip(gen_part(end_shape, end_len))
    arr *= amplitude
    return arr


def inject_random_spikes(): ...


def inject_DDoS(
    tensor: npt.NDArray,
    duration: int,
    n_senders: int,
    amplitude_factor: float,
    target: int,
) -> Tuple[npt.NDArray, npt.NDArray]:
    T = tensor.copy()
    mask = np.zeros_like(T)
    nx, _, nt = T.shape

    # Ensure injection fits within time bounds
    start = np.random.randint(0, nt - duration + 1)

    attackers = np.random.choice(nx, size=n_senders, replace=False)

    for sender in attackers:
        baseline_std = np.std(T[target, sender, :])

        ddos_shape = generate_shape(
            duration,
            "ramp",
            "ramp",
            (0.1, 0.1, 0.8),
            amplitude_factor * float(baseline_std),
        )

        T[target, sender, start : start + duration] += ddos_shape
        mask[target, sender, start : start + duration] = 1

    return T, mask


def inject_alpha_anomaly(
    tensor: npt.NDArray, duration: int, amplitude_factor, return_info: bool = False
) -> Tuple[npt.NDArray, npt.NDArray] | Tuple[npt.NDArray, npt.NDArray, dict]:

    T = tensor.copy()
    nx, _, nt = T.shape
    mask = np.zeros_like(T)

    start = np.random.randint(0, nt - duration + 1)
    od = np.random.choice(np.arange(nx), size=2, replace=False)
    amplitude = np.std(T[od[0], od[1], :]) * amplitude_factor

    T[od[0], od[1], start : start + duration] += amplitude
    mask[od[0], od[1], start : start + duration] = 1
    if return_info:
        return T, mask, {"od": od}
    else:
        return T, mask
