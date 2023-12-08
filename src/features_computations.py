from tsfeatures import tsfeatures
from pandas import DataFrame


def compute_tsfeatures(df: DataFrame, freq: int = None, fill_value: int = 0):
    features = tsfeatures(df, freq=freq)
    return features.fillna(value=fill_value)
