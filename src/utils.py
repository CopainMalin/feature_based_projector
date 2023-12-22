from pandas import DataFrame, Series, concat, read_csv, read_excel
from streamlit import dataframe, file_uploader, write
from sklearn.preprocessing import LabelEncoder
from numpy import diff, ndarray, zeros, arange, pi, sin, array
from numpy.random import randn
from typing import Iterable, Tuple


def transform_nixtla_format(nixtla_df: DataFrame, column: str = "H1") -> DataFrame:
    """
    Given a dataset in the nixtla format and a serie name, return the series values
    and its associated time index.

    Args:
        nixtla_df (DataFrame): The nixtla dataset to extract the serie on
        column (str, optional): The name of the serie. Defaults to "H1".

    Returns:
        DataFrame: The targeted serie and its time index.
    """
    sub_df = nixtla_df[nixtla_df["unique_id"] == column]
    return DataFrame({"ds": sub_df["ds"], column: sub_df["y"]})


def build_reduc_dim_df(reduc: ndarray, serie_names: Iterable) -> DataFrame:
    """
    Given the 3d reducted projection of the datasets and the names of the series, construct a dataframe to be plotted by plotly.

    Args:
        reduc (ndarray): The 3d reducted projection of the datasets.
        serie_names (Iterable): The time series names.

    Returns:
        DataFrame: The plotable dataset.
    """
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
    """
    Given a dataframe where the serie is in the last column (typical nixtla format),
    compute its first order differencing.

    Args:
        serie (DataFrame): The dataframe containing the serie.

    Returns:
        ndarray: The differentiated time serie vector.
    """
    return diff(serie.iloc[:, -1].values)


def get_top_five_correlations(reducted_dims: DataFrame, features: DataFrame) -> dict:
    """
    Given the 3d reducted projection of the datasets and the original projection of the datasets
    in the features space, compute the correlations between each 3d axis and the original features,
    and extract the top 5 correlated features (using kendall's Tau) to each axis.

    Args:
        reducted_dims (DataFrame): 3d reducted projection of the datasets in the feature space.
        features (DataFrame): The original feature space.

    Returns:
        dict: top 5 correlated features (using kendall's Tau) per axis,
        where the axis names are the keys of the dict.
    """
    merged_df = concat([reducted_dims, features], axis=1)
    correlations = merged_df.corr(method="kendall").iloc[:3, 3:].T
    top_five = {}
    for reduc_axis in correlations.columns:
        top_five[reduc_axis] = (
            correlations.loc[:, reduc_axis].sort_values(ascending=False).iloc[:5]
        )
    return top_five


def advanced_describe(serie: Series) -> DataFrame:
    """
    Given a series, provide an advanced describe adding skewness and kurtosis to original pandas
    describe method.

    Args:
        serie (Series): The serie to study.

    Returns:
        DataFrame: The described serie.
    """
    stats = serie.describe()
    stats.loc["skew"] = serie.skew()
    stats.loc["kurt"] = serie.kurtosis()
    return stats


def print_ts_features(features: DataFrame, serie_name: str) -> None:
    """
    Given the projection of the series in the features space and a serie name,
    prints the features on the streamlit app.

    Args:
        features (DataFrame): The features space dataframe.
        serie_name (str): The name of the serie to analyze.
    """
    values = features[features.loc[:, "unique_id"] == serie_name].drop(
        "unique_id", axis=1
    )
    values.index = [serie_name]

    for iteration in range(0, values.shape[1], 6):
        dataframe(values.iloc[:, iteration : iteration + 6], width=1000)


def encoder(x: str, selected_datasets: list) -> str:
    """
    Given a value and a list of selected datasets, encode the value.

    Args:
        x (str): The value to encode.
        selected_datasets (list): The selected datasets name list.

    Returns:
        str: The encoded value.
    """
    if x in selected_datasets:
        return "Selected"
    elif x in [
        "Autoregression (φ=0.9)",
        "White noise",
        "Seasonality",
        "Trend",
        "Seasonal/trend",
    ]:
        return "Added"
    else:
        return "Base"


def preprocess_features(features: DataFrame) -> Tuple[Series, DataFrame, ndarray]:
    """
    Preprocess the features space projection dataset by removing the "unique_id"
    and filling the NaNs if needed.

    Args:
        features (DataFrame): The features space projection dataset

    Returns:
        Tuple[Series, DataFrame, ndarray]: [The names of the series, the features dataset, the features matrix].
    """
    names = features.loc[:, "unique_id"]
    features = features.drop("unique_id", axis=1)
    features_values = features.fillna(0).values
    return names, features, features_values


def load_data() -> DataFrame:
    """
    Function to load the datasets.

    Raises:
        TypeError: If the format is not known.

    Returns:
        DataFrame: The loaded dataframe.
    """
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
    """
    Transform a dataset of series to the nixtla format.

    Args:
        dataset (DataFrame): The dataset to transform.

    Returns:
        DataFrame: The transformed dataset.
    """
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


def __generate_seasonal_trend_series(
    n: int, freq: int, trend_slope: float = 0.02, seasonal_amplitude: float = 10
) -> ndarray:
    """
    Generate toy trended + seasonal serie.

    Args:
        n (int): The number of points to generate.
        freq (float): The seasonal frequency.
        trend_slope (float, optional): The trend slope. Defaults to 0.02.
        seasonal_amplitude (int, optional): The seasonal amplitude. Defaults to 10.

    Returns:
        ndarray: The generated serie.
    """
    trend = arange(n) * trend_slope

    seasonal = seasonal_amplitude * sin(2 * pi * arange(n) / freq)

    noise = zeros(shape=n)

    series = trend + seasonal + noise

    return series


def __generate_autocorrelated_data(n: int, rho: float) -> ndarray:
    """
    Generate toy autoregressive dataset. y_t = rho * y_(t-1) + white noise.

    Args:
        n (int): The number of points to generate.
        rho (float): The AR coefficient.

    Returns:
        ndarray: The generated serie.
    """
    noise = randn(n)

    data = [noise[0]]
    for i in range(1, n):
        data.append(rho * data[i - 1] + noise[i])

    return array(data)


def inject_toy_series(dataframe: DataFrame, freq: int = 24) -> DataFrame:
    """
    Given a dataset of time series, inject to it toys series.

    Args:
        dataframe (DataFrame): The dataset containing the time series.
        freq (int, optional): The toys injected series seasonal frequency. Defaults to 24.

    Returns:
        DataFrame: The modified dataset.
    """
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
