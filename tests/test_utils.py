from pandas import Series, DataFrame
from numpy.random import rand, randn
from numpy.testing import assert_array_almost_equal

from precomputed_ressources.loader import load_transformed_h1, load_hourly_m4_dataset
from src.utils import (
    transform_nixtla_format,
    build_reduc_dim_df,
    advanced_describe,
    encoder,
    preprocess_features,
)


def test_transform_nixtla_format():
    targeted_col = load_transformed_h1()
    original_df = load_hourly_m4_dataset()
    assert targeted_col.equals(transform_nixtla_format(original_df, "H1"))


def test_build_reduc_df():
    original_data = rand(10, 3)
    names = [*["H1"] * 3, *["H2"] * 3, *["H3"] * 4]
    reducted_df = build_reduc_dim_df(original_data, names)
    assert (reducted_df.iloc[:, :3].values == original_data).all()
    assert (names == reducted_df["Name"]).all()
    assert (reducted_df["Style"] == [*[0] * 3, *[1] * 3, *[2] * 4]).all()


def test_advanced_describe():
    describe = advanced_describe(Series(randn(100)))
    assert set(
        [
            "count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
            "skew",
            "kurt",
        ]
    ).issubset(describe.index)


def test_encoder():
    assert encoder("H1", ["H1"]) == "Selected"
    assert encoder("H1", []) == "Base"
    assert encoder("Test", ["H1"]) == "Added"


def test_preprocess_features():
    df = DataFrame()
    df["unique_id"] = ["Test"] * 100
    df["1"] = [1] * 100
    df["2"] = [2] * 100

    names, features, features_values = preprocess_features(df)
    assert (names == ["Test"] * 100).all()
    assert features.equals(df.drop("unique_id", axis=1))
    assert_array_almost_equal(features_values, df.drop("unique_id", axis=1).values)
