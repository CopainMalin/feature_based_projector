# File with all the subtransformations neeeded to compute the features
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.seasonal import DecomposeResult
from statsmodels.tsa.stattools import acf
from typing import Iterable, Tuple
from scipy.optimize import curve_fit

from pandas import Series
from numpy import (
    arange,
    pi,
    random,
    zeros_like,
    array,
    floor,
    ones_like,
    zeros,
    cos,
    sin,
    sum as nsum,
    array_split,
    mean,
    var,
)


### Helper functions ###
# STL decomposition
def compute_STL_decompose(arr: Iterable, period: int = 21) -> DecomposeResult:
    stl = STL(arr, period=period)
    return stl.fit()


# Fitting an orthogonal quadratic regression to the time serie
def quadratic_regression(t, b0, b1, b2):
    return b0 + b1 * t + b2 * t**2


def fit_orthogonal_regression(arr: Iterable) -> Tuple[float, float]:
    params, _ = curve_fit(quadratic_regression, arange(len(arr)), arr)

    return (params[1], params[2])


# Computing acf
def compute_acf(arr: Iterable) -> list:
    return acf(arr, nlags=10)


# Tiled-window computation to compute lumpiness and stability
def tiled_windows_computations(
    arr: Iterable, window_size: int = 21
) -> Tuple[float, float]:
    num_windows = len(arr) // window_size

    windows = array_split(arr, num_windows)

    means = mean(windows, axis=1)
    variances = var(windows, axis=1)

    return (var(means), var(variances))


# generating seasonal time serie
def generate_seasonal_ts():
    # https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_seasonal.html#Synthetic-data-creation
    duration = 100 * 3
    periodicities = [10]
    num_harmonics = [3]
    std = array([2])
    random.seed(8678309)

    terms = []
    for ix, _ in enumerate(periodicities):
        s = __simulate_seasonal_term(
            periodicities[ix],
            duration / periodicities[ix],
            harmonics=num_harmonics[ix],
            noise_std=std[ix],
        )
        terms.append(s)

    terms.append(ones_like(terms[0]) * 10.0)
    return Series(nsum(terms, axis=0))


def __simulate_seasonal_term(periodicity, total_cycles, noise_std=1.0, harmonics=None):
    # https://www.statsmodels.org/dev/examples/notebooks/generated/statespace_seasonal.html#Synthetic-data-creation
    duration = periodicity * total_cycles
    assert duration == int(duration)
    duration = int(duration)
    harmonics = harmonics if harmonics else int(floor(periodicity / 2))

    lambda_p = 2 * pi / float(periodicity)

    gamma_jt = noise_std * random.randn((harmonics))
    gamma_star_jt = noise_std * random.randn((harmonics))

    total_timesteps = 100 * duration  # Pad for burn in
    series = zeros(total_timesteps)
    for t in range(total_timesteps):
        gamma_jtp1 = zeros_like(gamma_jt)
        gamma_star_jtp1 = zeros_like(gamma_star_jt)
        for j in range(1, harmonics + 1):
            cos_j = cos(lambda_p * j)
            sin_j = sin(lambda_p * j)
            gamma_jtp1[j - 1] = (
                gamma_jt[j - 1] * cos_j
                + gamma_star_jt[j - 1] * sin_j
                + noise_std * random.randn()
            )
            gamma_star_jtp1[j - 1] = (
                -gamma_jt[j - 1] * sin_j
                + gamma_star_jt[j - 1] * cos_j
                + noise_std * random.randn()
            )
        series[t] = nsum(gamma_jtp1)
        gamma_jt = gamma_jtp1
        gamma_star_jt = gamma_star_jtp1
    wanted_series = series[-duration:]  # Discard burn in

    return wanted_series
