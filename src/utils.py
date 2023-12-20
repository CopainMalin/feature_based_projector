from pandas import DataFrame, Series, concat
from streamlit import dataframe
from sklearn.preprocessing import LabelEncoder
from numpy import diff, ndarray
from typing import Iterable, Tuple


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


def get_top_five_correlations(reducted_dims: DataFrame, features: DataFrame) -> dict:
    merged_df = concat([reducted_dims, features], axis=1)
    correlations = merged_df.corr(method="kendall").iloc[:3, 3:].T
    top_five = {}
    for reduc_axis in correlations.columns:
        top_five[reduc_axis] = (
            correlations.loc[:, reduc_axis].sort_values(ascending=False).iloc[:5]
        )
    return top_five


def advanced_describe(serie: Series) -> DataFrame:
    stats = serie.describe()
    stats.loc["skew"] = serie.skew()
    stats.loc["kurt"] = serie.kurtosis()
    return stats


def print_ts_features(features: DataFrame, serie_name: str) -> None:
    values = features[features.loc[:, "unique_id"] == serie_name].drop(
        "unique_id", axis=1
    )
    values.index = [serie_name]

    for iteration in range(0, values.shape[1], 6):
        dataframe(values.iloc[:, iteration : iteration + 6], width=1000)


def encoder(x: str, selected_datasets: list) -> str:
    if x in selected_datasets:
        return "Selected"
    elif "H" in x:
        return "Base"
    else:
        return "Added"


def preprocess_features(features: DataFrame) -> Tuple[Series, DataFrame, ndarray]:
    names = features.loc[:, "unique_id"]
    features = features.drop("unique_id", axis=1)
    features_values = features.fillna(0).values
    return names, features, features_values
