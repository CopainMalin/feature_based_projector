from sklearn.neighbors import KernelDensity
from pandas import DataFrame
from numpy import linspace, exp, ndarray, diff, arange
from numpy.fft import fftfreq
from scipy.fft import fft
from scipy.signal import welch, cwt, ricker
from tsfeatures import tsfeatures
from typing import Tuple

from src.utils import compute_differenciated_serie


def compute_gaussian_kde(serie: DataFrame) -> Tuple[ndarray, ndarray]:
    """
    Given a serie, computes its gaussian kernel density estimation.

    Args:
        serie (DataFrame): The time serie to transform.

    Returns:
        Tuple[ndarray, ndarray]: The tuple [x_points, densities] resulting from the KDE.
    """
    data = serie.iloc[:, -1].values.reshape((-1, 1))
    kde = KernelDensity(kernel="gaussian", bandwidth=10)
    kde.fit(data)
    x_points = linspace(min(data), max(data), 1000).reshape(-1, 1)
    densities = exp(kde.score_samples(x_points))
    return x_points, densities


def compute_freq_and_psd(
    serie: DataFrame, frequency: int = 24
) -> Tuple[ndarray, ndarray]:
    """
    Given a serie, compute its frequency and its psd computation using the Welch method.

    Args:
        serie (DataFrame): The time serie to transform.
        frequency (int, optional): The seasonal frequency of the serie. Defaults to 24.

    Returns:
        Tuple[ndarray, ndarray]: The [frequency, power spectral distribution] returned by the welch method.
    """
    time_series = compute_differenciated_serie(serie)
    return welch(time_series, fs=1, nperseg=3 * frequency)


def compute_fft(dataset: DataFrame) -> Tuple[ndarray, ndarray]:
    """
    Given a serie, compute its frequency and its fourier transform using the FFT algorithm.

    Args:
        dataset (DataFrame): The time serie to transform.

    Returns:
        Tuple[ndarray, ndarray]: The [frequency, fourier transformed serie] tuple.
    """
    fft_result = fft(dataset.values.ravel())
    frequencies = fftfreq(len(fft_result))
    return (frequencies, fft_result)


def compute_wavelets(
    serie: DataFrame, frequency: int = 24
) -> Tuple[ndarray, ndarray, ndarray]:
    """
    Given a serie and its seasonal frequency, returns its widths, the ricker wavelet
    and the result of the continuous wavelet transform using the computed widths and the
    ricker wavelet.

    Args:
        serie (DataFrame): The time serie to transform.
        frequency (int, optional): The seasonality period. Defaults to 24.

    Returns:
        Tuple[ndarray, ndarray, ndarray]: [The computed widths, the ricker wavelet, the cwt result].
    """
    time_series = compute_differenciated_serie(serie)
    widths = arange(1, frequency + 10)
    wavelet = ricker
    return widths, wavelet, cwt(time_series, wavelet, widths)


def compute_tsfeatures(
    df: DataFrame, freq: int = None, fill_value: int = 0
) -> DataFrame:
    """
    Given a dataset of time series and their seasonal frequency computes the Hyndman's tsfeatures of each serie.

    Args:
        df (DataFrame): The dataset containing the time series to project in the feature space.
        freq (int, optional): The seasonal frequency of the series. Defaults to None.
        fill_value (int, optional): The value to fill the features that cannot be computed. Defaults to 0.

    Returns:
        DataFrame: The dataframe of the series projected in the features space.
    """
    features = tsfeatures(df, freq=freq)
    return features.fillna(value=fill_value)
