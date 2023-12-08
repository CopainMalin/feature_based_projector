from precomputed_ressources.loader import load_transformed_h1, load_hourly_m4_dataset
from src.utils import transform_nixtla_format


def test_transform_nixtla_format():
    targeted_col = load_transformed_h1()
    original_df = load_hourly_m4_dataset()
    assert targeted_col.equals(transform_nixtla_format(original_df, "H1"))
