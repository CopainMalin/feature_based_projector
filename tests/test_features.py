import pytest
from numpy import ones, arange, isclose, sin, square
from src.features import (
    length_ts,
    trend_strength,
    seasonal_strength,
    linearity,
    curvature,
    spikiness,
)

from src.tools import compute_STL_decompose, generate_seasonal_ts


# 1) serie length
def test_length_ts():
    serie = ones(10)
    assert length_ts(serie) == 10


# 2) trend strength
def test_trend_strength_trended_case():
    trended_serie = arange(100)
    assert trend_strength(compute_STL_decompose(trended_serie)) > 0.95


def test_trend_strength_untrended_case():
    untrended_serie = sin(arange(100))
    assert trend_strength(compute_STL_decompose(untrended_serie)) < 0.05


# 3) seasonal strength
def test_seasonal_strength_seasonal_case():
    seasonal_serie = generate_seasonal_ts()
    assert seasonal_strength(compute_STL_decompose(seasonal_serie, 10)) > 0.95


def test_trend_strength_unseasonal_case():
    unseasonal_serie = sin(arange(100))
    assert trend_strength(compute_STL_decompose(unseasonal_serie)) < 0.05


# 4) linearity
def test_linear_case():
    assert linearity(arange(100)) > 0


def test_flat_linear_case():
    assert isclose(linearity(ones(100)), 0)


# 5) curvature
def test_curved_case():
    assert curvature(-square(arange(100))) < 0


def test_neg_curved_case():
    assert curvature(square(arange(100))) > 0


def test_flat_case():
    assert isclose(curvature(ones(100)), 0)


# 6) spikiness
def test_spikiness():
    assert spikiness(arange(100)) > 0
