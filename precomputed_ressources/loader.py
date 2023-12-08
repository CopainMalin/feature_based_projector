import pickle
from pandas import DataFrame
from os import getcwd


def load_features_list() -> list:
    with open("precomputed_ressources/features_list.pickle", "rb") as handle:
        ressource = pickle.load(handle)
    return ressource


def load_computed_features() -> DataFrame:
    with open("precomputed_ressources/features_m4.pickle", "rb") as handle:
        ressource = pickle.load(handle)
    return ressource


def load_hourly_m4_dataset() -> DataFrame:
    with open("precomputed_ressources/hm4_dataset.pickle", "rb") as handle:
        ressource = pickle.load(handle)
    return ressource


def load_transformed_h1() -> DataFrame:
    with open("precomputed_ressources/transformed_h1.pickle", "rb") as handle:
        ressource = pickle.load(handle)
    return ressource
