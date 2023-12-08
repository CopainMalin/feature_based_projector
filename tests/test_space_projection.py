import pytest
from numpy.random import seed
from numpy.testing import assert_array_almost_equal
from pandas import DataFrame

from precomputed_ressources.loader import load_kde_h1_seed_0, load_hourly_m4_dataset
from src.space_projection import compute_gaussian_kde
from src.utils import transform_nixtla_format


@pytest.fixture
def dataset():
    return load_hourly_m4_dataset()


def test_gaussian_kde_computation(dataset: DataFrame):
    seed(0)
    _, densities = compute_gaussian_kde(transform_nixtla_format(dataset, "H1"))
    precomputed_densities = load_kde_h1_seed_0()
    print(densities)
    print(precomputed_densities)
    assert (densities == precomputed_densities).all()
