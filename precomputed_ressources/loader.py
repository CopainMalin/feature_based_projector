import pickle
from pandas import DataFrame
from numpy import ndarray


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


def load_kde_h1_seed_0() -> list:
    with open("precomputed_ressources/kde_h1_seed_0.pickle", "rb") as handle:
        ressource = pickle.load(handle)
    return ressource


def load_welch_freq_and_psd() -> list:
    with open("precomputed_ressources/welch_freq_and_psd.pickle", "rb") as handle:
        ressource = pickle.load(handle)
    return ressource


def load_wavelet_transform() -> ndarray:
    with open("precomputed_ressources/wavelet_transform.pickle", "rb") as handle:
        ressource = pickle.load(handle)
    return ressource


def load_fft() -> list:
    with open("precomputed_ressources/fft_and_freq.pickle", "rb") as handle:
        ressource = pickle.load(handle)
    return ressource


def load_modified_features() -> list:
    with open("precomputed_ressources/features_m4_modified.pickle", "rb") as handle:
        ressource = pickle.load(handle)
    return ressource
