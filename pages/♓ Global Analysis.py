from streamlit import columns, selectbox, plotly_chart, multiselect, write, dataframe

from numpy import corrcoef, argsort, abs as nabs, flip
from pandas import concat, DataFrame

from src.utils import (
    build_reduc_dim_df,
    get_top_five_correlations,
    encoder,
    preprocess_features,
)
from src.plotting_tools import plot_reducted_dim, plot_correlation_heatmap
from src.dimension_reduction import PCAReductor, UMAPReductor, TSNEReductor
from precomputed_ressources.loader import load_modified_features

# Data loading - to be removed
features = load_modified_features()

names, features, features_values = preprocess_features(features=features)

# layout
c1, c2 = columns([0.3, 0.7])
with c1:
    reduc_dim_algo = selectbox(
        label="Dimension reduction algorithm :", options=["PCA", "T-SNE", "UMAP"]
    )
with c2:
    selected_datasets = multiselect(label="Dataset(s) to focus on:", options=names)

# Reducted dim scatterplot
if reduc_dim_algo == "PCA":
    reducted_features = PCAReductor().fit_transform(features_values)
elif reduc_dim_algo == "T-SNE":
    reducted_features = TSNEReductor().fit_transform(features_values)
else:
    reducted_features = UMAPReductor().fit_transform(features_values)

reducted_df = build_reduc_dim_df(reducted_features, serie_names=names)
reducted_df["Style"] = names.apply(encoder, selected_datasets=selected_datasets)

fig = plot_reducted_dim(reducted_df, reduc_dim_algo)
plotly_chart(figure_or_data=fig, use_container_width=True)

# Correlation part
top_five = get_top_five_correlations(reducted_df.iloc[:, :3], features)
fig = plot_correlation_heatmap(top_five)
plotly_chart(figure_or_data=fig, use_container_width=True)
