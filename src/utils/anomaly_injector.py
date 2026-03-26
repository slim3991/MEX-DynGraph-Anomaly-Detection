from typing import Callable, Literal, NamedTuple, Tuple
import numpy as np
import numpy.typing as npt


class AnomalyEvent(NamedTuple):
    source: int
    dest: int
    start: int
    duration: int


def generate_shape(
    duration: int,
    begin_shape: Literal["ramp", "gaussian", "step"],
    end_shape: Literal["ramp", "gaussian", "step"],
    ratios: Tuple[float, float, float],
    amplitude: float,
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


def inject_shape(T, source, dest, start, duration, amplitude, up_shape, down_shape):

    shape = generate_shape(
        duration=duration,
        begin_shape=up_shape,
        end_shape=down_shape,
        ratios=(1, 0, 5),
        amplitude=amplitude,
    )

    T[source, dest, start : start + duration] += shape

    return T


def inject_random_shapes(
    T,
    start_min,
    start_max,
    min_durantion,
    max_duration,
    amplitude_factor=5.0,
    n_shapes=1,
):

    nx, ny, nt = T.shape
    L = np.zeros_like(T, dtype=np.bool)
    events_list = []
    shapes = ("ramp", "gaussian", "step")

    for _ in range(n_shapes):
        source, dest = np.random.randint(0, 11, size=2)

        duration = np.random.randint(low=min_durantion, high=max_duration)
        start = np.random.randint(low=start_min, high=start_max)

        std = np.std(T[source, dest, :])
        amplitude = std * amplitude_factor

        # up_shape, down_shape = np.random.choice(shapes), np.random.choice(shapes)
        up_shape, down_shape = np.random.choice(shapes, size=2)
        T = inject_shape(
            T,
            source=source,
            dest=dest,
            start=start,
            duration=duration,
            amplitude=amplitude,
            up_shape=up_shape,
            down_shape=down_shape,
        )

        L[source, dest, start : start + duration] = True
        events_list.append(
            AnomalyEvent(
                source=source,
                dest=dest,
                start=start,
                duration=duration,
            )
        )

    return T, L, events_list


# def inject_random_spikes(T, n_spikes=1000):
#     nx, ny, nt = T.shape
#
#     L = np.zeros_like(T)
#     anomalies = np.zeros_like(T)
#
#     count = 0
#     while count < n_spikes:
#         source = np.random.randint(0, nx)
#         dest = np.random.randint(0, ny)
#         t = np.random.randint(0, nt)
#
#         if L[source, dest, t] == 0:
#             tmax = np.max(T[source, dest, :])
#             anomalies[source, dest, t] = np.random.uniform(low=tmax, high=tmax * 2)
#             L[source, dest, t] = 1
#             count += 1
#
#     T_spiked = np.maximum(T, anomalies)
#     return T_spiked, L


def inject_random_spikes_normal(T, amplitude_factor=5, n_spikes=1000):
    nx, ny, nt = T.shape

    L = np.zeros_like(T)
    anomalies = np.zeros_like(T)

    count = 0
    while count < n_spikes:
        source = np.random.randint(0, nx)
        dest = np.random.randint(0, ny)
        t = np.random.randint(0, nt)
        if L[source, dest, t] == 0:
            std = np.std(T[source, dest, :])

            anomalies[source, dest, t] = std * amplitude_factor
            L[source, dest, t] = 1
            count += 1

    T_spiked = np.maximum(T, anomalies)
    return T_spiked, L


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
    tensor: npt.NDArray,
    duration: int,
    amplitude_factor,
    return_info: bool = False,
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
