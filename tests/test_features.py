import pytest
from numpy import ones, arange, isclose, sin, square, cumsum
from numpy.random import randn
from src.features import (
    length_ts,
    trend_strength,
    seasonal_strength,
    linearity,
    curvature,
    spikiness,
    e_acf1,
    e_acf10,
    stability,
    lumpiness,
    entropy,
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
    assert spikiness(compute_STL_decompose(arange(100))) > 0


# 7) e_acf1
def test_e_acf1():
    assert e_acf1(compute_STL_decompose(square(arange(100)))) > 0.5


# 8) e_acf10
def test_e_acf10():
    assert e_acf10(compute_STL_decompose(ones(100))) < 0.1


# 9) stability
def test_stability_flat_case():
    assert isclose(stability(ones(100)), 0)


def test_stability_non_flat_case():
    assert stability(square(arange(100))) > 0


# 10) lumpiness
def test_lumpiness_flat_case():
    assert isclose(lumpiness(ones(100)), 0)


def test_lumpiness_non_flat_case():
    assert stability(square(arange(100))) > 0


# 11) entropy
def test_entropy_flat_case():
    assert isclose(entropy(ones(100)), 0)


def test_entropy_rw_case():
    assert entropy(cumsum(randn(100))) > 0.05
