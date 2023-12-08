from pandas import DataFrame


def transform_nixtla_format(nixtla_df: DataFrame, column: str = "H1") -> DataFrame:
    sub_df = nixtla_df[nixtla_df["unique_id"] == column]
    return DataFrame({"ds": sub_df["ds"], column: sub_df["y"]})
