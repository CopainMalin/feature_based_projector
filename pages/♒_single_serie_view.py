import streamlit as st

from precomputed_ressources.loader import load_hourly_m4_dataset
from src.space_projection import compute_fft, compute_wavelets, compute_freq_and_psd
from src.utils import transform_nixtla_format
from src.plotting_tools import (
    plot_time_view,
    plot_fft_view,
    plot_wavelet_view,
    plot_density_view,
    plot_psd_view,
)

# dataset loading & selection - TO BE REMOVED
dataset = load_hourly_m4_dataset()


c1, c2 = st.columns(2)
with c1:
    serie_name = st.selectbox(
        label="Choose the serie to plot:",
        options=dataset.unique_id.sort_values().unique(),
    )
with c2:
    plot_name = st.selectbox(
        label="CHoose the type of plot:",
        options=[
            "Time view",
            "Density view",
            "Frequentist (FFT) view",
            "Frequentist (PSD) view",
            "Wavelets (ricker) view",
        ],
    )

data = transform_nixtla_format(dataset, serie_name)

match plot_name:
    case "Time view":
        fig = plot_time_view(data, serie_name)
    case "Density view":
        fig = plot_density_view(data, serie_name)
    case "Frequentist (FFT) view":
        fig = plot_fft_view(*compute_fft(data.loc[:, serie_name]), serie_name)
    case "Frequentist (PSD) view":
        frequencies, psd = compute_freq_and_psd(data)
        fig = plot_psd_view(frequencies, psd, serie_name)
    case "Wavelets (ricker) view":
        widths, _, cwt_result = compute_wavelets(data)
        fig = plot_wavelet_view(widths, len(data), cwt_result, serie_name)
    case _:
        raise ValueError("Unknown type of plot")
st.plotly_chart(fig, use_container_width=True)

# # plotting
# group_labels = [serie_name]

# fig = ff.create_distplot(
#     [data.loc[:, serie_name]],
#     group_labels,
#     colors=["dodgerblue"],
#     # bin_size=[0.1],
#     show_curve=True,
# )

# fig.update(layout_title_text="Displot")
# st.plotly_chart(fig, use_container_width=True)
