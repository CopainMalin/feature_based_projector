import pytest
from numpy.random import seed
from pandas import DataFrame

from precomputed_ressources.loader import (
    load_kde_h1_seed_0,
    load_hourly_m4_dataset,
    load_welch_freq_and_psd,
    load_wavelet_transform,
    load_fft,
)
from src.space_projection import (
    compute_gaussian_kde,
    compute_freq_and_psd,
    compute_wavelets,
    compute_fft,
)
from src.utils import transform_nixtla_format


@pytest.fixture
def dataset():
    return load_hourly_m4_dataset()


def test_gaussian_kde_computation(dataset: DataFrame):
    seed(0)
    _, densities = compute_gaussian_kde(transform_nixtla_format(dataset, "H1"))
    precomputed_densities = load_kde_h1_seed_0()
    assert (densities == precomputed_densities).all()


def test_psd_computation(dataset: DataFrame):
    seed(0)
    freq, psd = compute_freq_and_psd(
        transform_nixtla_format(dataset, "H1"), frequency=24
    )
    precomputed_freq, precomputed_psd = load_welch_freq_and_psd()
    assert (freq == precomputed_freq).all()
    assert (psd == precomputed_psd).all()


def test_wavelets_computation(dataset: DataFrame):
    seed(0)
    _, _, continuous_wavelet_transform = compute_wavelets(
        transform_nixtla_format(dataset, "H1"), 24
    )
    precomputed_cwt = load_wavelet_transform()
    assert (continuous_wavelet_transform == precomputed_cwt).all()


# def test_fft_computation(dataset: DataFrame):
#     stochastic result
#     freq, fft = compute_fft(transform_nixtla_format(dataset, "H1").loc[:, "H1"])
#     precomputed_freq, precomputed_fft = load_fft()
#     assert (freq == precomputed_freq).all()
#     assert (fft == precomputed_fft).all()
