from precomputed_ressources.loader import load_transformed_h1, load_hourly_m4_dataset
from src.utils import transform_nixtla_format, build_reduc_dim_df
from numpy.random import rand


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
