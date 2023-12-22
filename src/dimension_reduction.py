from abc import ABC, abstractmethod
from numpy.typing import ArrayLike
from numpy import number
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP


class Reductor(BaseEstimator, ABC):
    def __init__(self) -> None:
        self.reducted_dataset_ = None

    @abstractmethod
    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        ...

    def standard_scale(self, X: ArrayLike) -> ArrayLike:
        return StandardScaler().fit_transform(X)

    def test_numeric(self, X: ArrayLike) -> bool:
        if X.dtype != number:
            raise RuntimeError("Input containing non-numeric values")


class PCAReductor(Reductor):
    """
    A dimension reductor using Principal Component Analysis algorithm.
    """

    def __init__(self) -> None:
        super().__init__()

    def __repr__(self):
        return f"PCAReductor\nReducted dataset available : {self.reducted_dataset_ is not None}"

    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        """
        Fit the PCA object and transform the dataset.

        Args:
            X (ArrayLike): The dataset to perform dimension reduction on.

        Returns:
            ndarray: the transformed dataset.
        """
        super().test_numeric(X)
        self.reducted_dataset_ = PCA(n_components=3).fit_transform(
            self.standard_scale(X)
        )
        return self.reducted_dataset_


class TSNEReductor(Reductor):
    """
    A dimension reductor using the T-distributed Stochastic Neighbor Embedding method.
    """

    def __init__(self, perplexity: float = 30) -> None:
        super().__init__()
        self.perplexity = perplexity

    def __repr__(self):
        return f"TSNEReductor\nReducted dataset available : {self.reducted_dataset_ is not None}"

    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        """
        Fit the TSNE object and transform the dataset.

        Args:
            X (ArrayLike): The dataset to perform dimension reduction on.

        Returns:
            ndarray: the transformed dataset.
        """
        super().test_numeric(X)
        self.reducted_dataset_ = TSNE(
            n_components=3, perplexity=self.perplexity
        ).fit_transform(self.standard_scale(X))
        return self.reducted_dataset_


class UMAPReductor(Reductor):
    """
    A dimension reductor using the Uniform Manifold Approximation and Projection algorithm.
    """

    def __init__(self, n_neighbors: float = 15, random_state: int = 0) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def __repr__(self):
        return f"UMAPReductor\nReducted dataset available : {self.reducted_dataset_ is not None}"

    def fit_transform(self, X: ArrayLike) -> ArrayLike:
        """
        Fit the UMAP object and transform the dataset.

        Args:
            X (ArrayLike): The dataset to perform dimension reduction on.

        Returns:
            ndarray: the transformed dataset.
        """
        super().test_numeric(X)
        self.reducted_dataset_ = UMAP(
            n_components=3, n_neighbors=self.n_neighbors, random_state=self.random_state
        ).fit_transform(self.standard_scale(X))
        return self.reducted_dataset_
