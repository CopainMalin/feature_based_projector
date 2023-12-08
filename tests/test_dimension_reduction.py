import pytest
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from src.dimension_reduction import PCAReductor, TSNEReductor, UMAPReductor
from numpy.random import rand, seed
from numpy import isclose, ndarray, chararray


@pytest.fixture
def fake_data() -> ndarray:
    return StandardScaler().fit_transform(rand(100, 10))


class TestPCA:
    def test_non_numeric_error(self):
        with pytest.raises(RuntimeError, match="Input containing non-numeric values"):
            PCAReductor().fit_transform(chararray(shape=(10, 10)))

    def test_pca_results(self, fake_data: ndarray):
        transformed_reductor = PCAReductor().fit_transform(fake_data)
        transformed_sklearn = PCA(n_components=3).fit_transform(fake_data)
        assert isclose(transformed_reductor, transformed_sklearn).all()


class TestTSNE:
    def test_non_numeric_error(self):
        with pytest.raises(RuntimeError, match="Input containing non-numeric values"):
            TSNEReductor().fit_transform(chararray(shape=(10, 10)))

    def test_tsne_results(self, fake_data: ndarray):
        transformed_reductor = TSNEReductor(perplexity=50).fit_transform(fake_data)
        transformed_sklearn = TSNE(n_components=3, perplexity=50).fit_transform(
            StandardScaler().fit_transform(fake_data)
        )
        assert isclose(transformed_reductor, transformed_sklearn).all()


class TestUMAP:
    def test_non_numeric_error(self):
        with pytest.raises(RuntimeError, match="Input containing non-numeric values"):
            TSNEReductor().fit_transform(chararray(shape=(10, 10)))

    def test_umap_results(self, fake_data: ndarray):
        seed(0)
        transformed_reductor = UMAPReductor(n_neighbors=50).fit_transform(fake_data)
        transformed_UMAP = UMAP(
            n_components=3, n_neighbors=50, random_state=0
        ).fit_transform(fake_data)
        assert isclose(transformed_reductor, transformed_UMAP).all()
