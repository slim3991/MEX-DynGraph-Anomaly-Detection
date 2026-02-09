import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


class PoissonTraffic:
    def __init__(self, days: int, drift: float) -> None:
        self.days = days
        self.total_hours = 24 * days
        self._window = self._generate_window()
        self.drift = drift

    def _generate_window(self) -> npt.NDArray:
        window_size = 8
        window = np.hamming(window_size) + 1
        return window / window.sum()

    def generate(self, scale: float = 1000) -> npt.NDArray:
        signal = np.full(self.total_hours, 0.1)

        signal += np.random.normal(0, 0.4, self.days * 24)

        phase_shift = np.random.randint(-1, 2)

        for h in range(self.total_hours):
            shifted_h = (h + phase_shift) % self.total_hours
            day_of_week = (h // 24) % 7
            hour_of_day = h % 24

            if day_of_week < 5:
                if 8 <= hour_of_day <= 17:
                    signal[shifted_h] = 1
                elif 18 <= hour_of_day <= 21:
                    signal[shifted_h] = 0.6
            else:
                if 10 <= hour_of_day <= 18:
                    signal[shifted_h] = 0.5

        drift = np.cumsum(np.random.normal(0, self.drift, self.total_hours))
        drift = np.clip(drift, 0, None)
        signal = np.clip(signal + drift, 0, None)

        mean_curve = np.convolve(signal, self._window, mode="same")
        return np.random.poisson(scale * mean_curve)


if __name__ == "__main__":
    data = np.load("data/abiline_ten.npy")

    pt = PoissonTraffic(14, 0.01)

    plt.plot(pt.generate(200))
    plt.show()
