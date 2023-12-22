from streamlit import (
    columns,
    selectbox,
    plotly_chart,
    dataframe,
    write,
    title,
    session_state,
    set_page_config,
)
from src.space_projection import compute_fft, compute_wavelets, compute_freq_and_psd
from src.utils import transform_nixtla_format, advanced_describe, print_ts_features
from src.plotting_tools import (
    plot_time_view,
    plot_fft_view,
    plot_wavelet_view,
    plot_density_view,
    plot_psd_view,
)
from pandas import DataFrame


set_page_config(page_title="Local Analysis")

if "data_loaded" in session_state:
    dataset = session_state["dataset"]
    features = session_state["features"]

    title(":orange[Graphical] analysis :male-detective:")

    # layout
    c1, c2 = columns(2)
    with c1:
        serie_name = selectbox(
            label="Choose the serie to plot:",
            options=dataset.unique_id.sort_values().unique(),
            index=1,
        )
    with c2:
        plot_name = selectbox(
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

    # plot
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

    plotly_chart(fig, use_container_width=True)

    # numerical analysis
    title(":blue[Numerical] analysis :male-detective:")
    write("Base analysis :")
    dataframe(DataFrame(advanced_describe(data.loc[:, serie_name])).T)

    write("Advanced analysis :")
    print_ts_features(features, serie_name)

else:
    title(
        ":warning: You must load your dataset first in the :orange[Dataset Management] page !"
    )
