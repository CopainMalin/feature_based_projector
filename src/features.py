from typing import Iterable
from numpy import var, delete
from statsmodels.tsa.seasonal import DecomposeResult

from src.tools import fit_orthogonal_regression, compute_acf, tiled_windows_computations

# Based on this paper
# https://robjhyndman.com/papers/fforma.pdf


# 1) T
def length_ts(arr: Iterable) -> int:
    return len(arr)


# 2) trend
def trend_strength(decompose: DecomposeResult) -> float:
    # https://otexts.com/fpp2/seasonal-strength.html
    return max(
        0.0, 1.0 - var(decompose.resid) / (var(decompose.trend + decompose.resid))
    )


# 3) seasonality
def seasonal_strength(decompose: DecomposeResult) -> float:
    # https://otexts.com/fpp2/seasonal-strength.html
    return max(
        0.0, 1.0 - var(decompose.resid) / (var(decompose.seasonal + decompose.resid))
    )


# 4) linearity
def linearity(arr: Iterable) -> float:
    return fit_orthogonal_regression(arr)[0]


# 5) curvature
def curvature(arr: Iterable) -> float:
    return fit_orthogonal_regression(arr)[1]


# 6) spikiness
def spikiness(decomposition: DecomposeResult) -> float:
    # https://pkg.robjhyndman.com/tsfeatures/articles/tsfeatures.html
    return var(
        [
            var(delete(decomposition.resid, i))
            for i in range(len(decomposition.observed))
        ]
    )


# 7) e_acf1
def e_acf1(decomposition: DecomposeResult) -> float:
    return compute_acf(decomposition.resid)[1]


# 8) e_acf10
def e_acf10(decomposition: DecomposeResult) -> float:
    return compute_acf(decomposition.resid)[10]


# 9) stability
def stability(arr: Iterable) -> float:
    return tiled_windows_computations(arr)[0]


# 10) lumpiness
def lumpiness(arr: Iterable) -> float:
    return tiled_windows_computations(arr)[1]
