from pandas import DataFrame, Series, concat, read_csv, read_excel
from streamlit import dataframe, file_uploader, write
from sklearn.preprocessing import LabelEncoder
from numpy import diff, ndarray, zeros, arange, pi, sin, array
from numpy.random import randn
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


def load_data() -> DataFrame:
    file = file_uploader(
        label="Load your dataset here !",
        type=["xlsx", "csv"],
        accept_multiple_files=False,
        label_visibility="hidden",
    )
    if file is not None:
        if ".xlsx" in file.name:
            dataframe = read_excel(file)
        elif ".csv" in file.name:
            dataframe = read_csv(file)
        else:
            raise TypeError("File must be a .csv or a .xlsx.")

        cols = [x for x in dataframe.columns if "Unnamed" in x]
        if cols:
            dataframe.drop(cols, axis=1, inplace=True)
        return dataframe


def transform_dataset(dataset: DataFrame) -> DataFrame:
    df_columns = dataset.columns
    if ("unique_id" in df_columns) & ("ds" in df_columns) & ("y" in df_columns):
        return dataset

    elif "date" in df_columns:
        ds = dataset.loc[:, "date"].to_list() * (len(df_columns) - 1)

    else:
        ds = dataset.index.tolist() * len(df_columns)

    y = list()
    unique_id = list()
    for colonne in [x for x in df_columns if x != "date"]:
        unique_id.append([colonne] * dataset.shape[0])
        y.append(dataset.loc[:, colonne])

    # flatten list of lists
    unique_id = [x for xs in unique_id for x in xs]
    y = [x for xs in y for x in xs]

    return DataFrame(
        data={
            "unique_id": unique_id,
            "ds": ds,
            "y": y,
        }
    )


def __generate_seasonal_trend_series(n, freq, trend_slope=0.02, seasonal_amplitude=10):
    trend = arange(n) * trend_slope

    seasonal = seasonal_amplitude * sin(2 * pi * arange(n) / freq)

    noise = zeros(shape=n)

    series = trend + seasonal + noise

    return series


def __generate_autocorrelated_data(n, rho):
    noise = randn(n)

    data = [noise[0]]
    for i in range(1, n):
        data.append(rho * data[i - 1] + noise[i])

    return array(data)


def inject_toy_series(dataframe: DataFrame, freq: int = 24) -> DataFrame:
    size = 1000

    autocorr = DataFrame(
        {
            "ds": arange(size),
            "unique_id": ["Autoregression (φ=0.9)"] * size,
            "y": __generate_autocorrelated_data(size, 0.9),
        }
    )

    noise = DataFrame(
        {
            "ds": arange(size),
            "unique_id": ["White noise"] * size,
            "y": randn(size),
        }
    )

    seasonal = DataFrame(
        {
            "ds": arange(size),
            "unique_id": ["Seasonality"] * size,
            "y": __generate_seasonal_trend_series(
                n=1000,
                trend_slope=0,
                seasonal_amplitude=10,
                freq=freq,
            ),
        }
    )

    trended = DataFrame(
        {
            "ds": arange(size),
            "unique_id": ["Trend"] * size,
            "y": __generate_seasonal_trend_series(
                n=1000, trend_slope=0.1, seasonal_amplitude=0, freq=freq
            ),
        }
    )

    s_and_t = DataFrame(
        {
            "unique_id": ["Seasonal/trend"] * size,
            "ds": arange(size),
            "y": __generate_seasonal_trend_series(
                n=size, trend_slope=0.2, seasonal_amplitude=10, freq=freq
            ),
        }
    )

    new_df = concat([dataframe, autocorr])
    new_df = concat([new_df, noise])
    new_df = concat([new_df, seasonal])
    new_df = concat([new_df, s_and_t])
    new_df = concat([new_df, trended])

    return new_df
