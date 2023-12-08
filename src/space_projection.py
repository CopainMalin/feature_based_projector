from sklearn.neighbors import KernelDensity
from pandas import DataFrame
from numpy import linspace, exp, ndarray, diff, arange
from typing import Tuple
from scipy.signal import welch
from scipy.signal import cwt, ricker

from src.utils import compute_differenciated_serie


def compute_gaussian_kde(serie: DataFrame) -> Tuple[ndarray, ndarray]:
    data = serie.iloc[:, -1].values.reshape((-1, 1))
    kde = KernelDensity(kernel="gaussian", bandwidth=10)
    kde.fit(data)
    x_points = linspace(min(data), max(data), 1000).reshape(-1, 1)
    densities = exp(kde.score_samples(x_points))
    return x_points, densities


def compute_freq_and_psd(
    serie: DataFrame, frequency: int = 24
) -> Tuple[ndarray, ndarray]:
    time_series = compute_differenciated_serie(serie)
    return welch(time_series, fs=1, nperseg=3 * frequency)


def compute_wavelets(serie: DataFrame, frequency: int = 24) -> ndarray:
    time_series = compute_differenciated_serie(serie)
    widths = arange(1, frequency + 10)
    wavelet = ricker
    return cwt(time_series, wavelet, widths)
