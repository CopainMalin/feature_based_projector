import pytest
from pandas import DataFrame
from precomputed_ressources.loader import load_hourly_m4_dataset, load_features_list

from src.features_computations import compute_tsfeatures


class TestFeaturesComputation:
    @pytest.fixture
    def features(self) -> DataFrame:
        return compute_tsfeatures(load_hourly_m4_dataset(), freq=24, fill_value=0)

    def test_no_nan(self, features: DataFrame):
        assert features.isna().sum().sum() == 0

    def test_all_features_are_computed(self, features: DataFrame):
        assert set(load_features_list()).issubset(features.columns)
