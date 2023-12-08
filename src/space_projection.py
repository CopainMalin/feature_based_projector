from sklearn.neighbors import KernelDensity
from pandas import DataFrame
from numpy import linspace, exp


def compute_gaussian_kde(serie: DataFrame):
    data = serie.iloc[:, -1].values.reshape((-1, 1))
    kde = KernelDensity(kernel="gaussian", bandwidth=10)
    kde.fit(data)
    x_points = linspace(min(data), max(data), 1000).reshape(-1, 1)
    densities = exp(kde.score_samples(x_points))
    return x_points, densities
