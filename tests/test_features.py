import pytest
from numpy import ones, arange, isclose, sin, linspace
from src.features import length_ts, trend_strength, seasonal_strength

from src.tools import compute_STL_decompose, generate_seasonal_ts


# 1) serie length
def test_length_ts():
    serie = ones(10)
    assert length_ts(serie) == 10


# 2) trend strength
def test_trend_strength_trended_case():
    trended_serie = arange(100)
    assert isclose(trend_strength(compute_STL_decompose(trended_serie)), 1)


def test_trend_strength_untrended_case():
    untrended_serie = sin(arange(100))
    assert isclose(trend_strength(compute_STL_decompose(untrended_serie)), 0)


# 3) seasonal strength
def test_seasonal_strength_seasonal_case():
    seasonal_serie = generate_seasonal_ts()
    assert seasonal_strength(compute_STL_decompose(seasonal_serie, 10)) > 0.95


def test_trend_strength_unseasonal_case():
    unseasonal_serie = sin(arange(100))
    assert isclose(trend_strength(compute_STL_decompose(unseasonal_serie)), 0)
