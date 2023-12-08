from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.manifold import TSNE
from numpy.typing import ArrayLike
from numpy import number
from abc import ABC, abstractmethod

# Data class comprenant toutes les représentations


class Reductor(BaseEstimator, ABC):
    def __init__(self) -> None:
        self.reducted_dataset_ = None

    @abstractmethod
    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        ...

    def test_numeric(self, X: ArrayLike) -> bool:
        if X.dtype != number:
            raise RuntimeError("Input containing non-numeric values")


class PCAReductor(Reductor):
    def __init__(self) -> None:
        super().__init__()

    def __repr__(self):
        return f"PCAReductor\nReducted dataset available : {self.reducted_dataset_ is not None}"

    def fit_transform(self, X: ArrayLike) -> "PCAReductor":
        super().test_numeric(X)
        self.reducted_dataset_ = PCA(n_components=3).fit_transform(X)
        return self.reducted_dataset_


class TSNEReductor(Reductor):
    def __init__(self, perplexity: float = 30) -> None:
        super().__init__()
        self.perplexity = perplexity

    def __repr__(self):
        return f"TSNEReductor\nReducted dataset available : {self.reducted_dataset_ is not None}"

    def fit_transform(self, X: ArrayLike) -> "TSNEReductor":
        super().test_numeric(X)
        self.reducted_dataset_ = TSNE(
            n_components=3, perplexity=self.perplexity
        ).fit_transform(X)
        return self.reducted_dataset_


class UMAPReductor(Reductor):
    def __init__(self, n_neighbors: float = 15, random_state: int = 0) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def __repr__(self):
        return f"UMAPReductor\nReducted dataset available : {self.reducted_dataset_ is not None}"

    def fit_transform(self, X: ArrayLike) -> "UMAPReductor":
        super().test_numeric(X)
        self.reducted_dataset_ = UMAP(
            n_components=3, n_neighbors=self.n_neighbors, random_state=self.random_state
        ).fit_transform(X)
        return self.reducted_dataset_