import numpy as np
import pywt

T = np.load("data/abiline_ten.npy")


def get_synth_signal(signal, level: int = 5):

    wavelet = "db5"
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    approx_only = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]

    baseline_signal = pywt.waverec(approx_only, wavelet)
    details_only = [np.zeros_like(coeffs[0])] + coeffs[1:]
    detail_signal = pywt.waverec(details_only, wavelet)

    detail_variance = np.var(detail_signal)
    noise = np.random.normal(
        loc=0, scale=np.sqrt(detail_variance) / 4, size=len(baseline_signal)
    )
    synthetic_signal = baseline_signal + noise
    return synthetic_signal
