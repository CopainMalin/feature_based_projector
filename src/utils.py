from pandas import DataFrame
from sklearn.preprocessing import LabelEncoder
from numpy import diff, ndarray
from typing import Iterable


def transform_nixtla_format(nixtla_df: DataFrame, column: str = "H1") -> DataFrame:
    sub_df = nixtla_df[nixtla_df["unique_id"] == column]
    return DataFrame({"ds": sub_df["ds"], column: sub_df["y"]})


def build_reduc_dim_df(reduc: ndarray, serie_names: Iterable) -> DataFrame:
    return DataFrame(
        {
            "fst_dim": reduc[:, 0],
            "snd_dim": reduc[:, 1],
            "trd_dim": reduc[:, 2],
            "Name": serie_names,
            "Style": LabelEncoder().fit_transform(serie_names),
        }
    )


def compute_differenciated_serie(serie: DataFrame) -> ndarray:
    return diff(serie.iloc[:, -1].values)
