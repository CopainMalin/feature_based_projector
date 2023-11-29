from typing import Iterable
from numpy import var
from statsmodels.tsa.seasonal import DecomposeResult

# Based on this paper
# https://robjhyndman.com/papers/fforma.pdf


def length_ts(arr: Iterable) -> int:
    return len(arr)


def trend_strength(decompose: DecomposeResult) -> float:
    # https://otexts.com/fpp2/seasonal-strength.html
    return max(
        0.0, 1.0 - var(decompose.resid) / (var(decompose.trend + decompose.resid))
    )


def seasonal_strength(decompose: DecomposeResult) -> float:
    # https://otexts.com/fpp2/seasonal-strength.html
    return max(
        0.0, 1.0 - var(decompose.resid) / (var(decompose.seasonal + decompose.resid))
    )
