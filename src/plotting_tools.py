from plotly.graph_objects import Figure, Scatter, Surface, Heatmap
from plotly.figure_factory import create_distplot
from plotly.express import scatter_3d, colors
from pandas import DataFrame
from numpy import abs as nabs, ndarray, arange, meshgrid, log


def plot_time_view(data: DataFrame, serie_name: str) -> Figure:
    fig = Figure()
    fig.add_traces(
        Scatter(
            x=data.loc[:, "ds"],
            y=data.loc[:, serie_name],
            name=serie_name,
            line={"color": "rgb(30, 144, 250)", "width": 1.0},
            fill="tonexty",
            fillcolor="rgba(30, 144, 250, .3)",
        )
    )

    fig.update_layout(
        title=f"Time representation of the serie: {serie_name}",
        xaxis_title="Time index",
        yaxis_title=f"{serie_name}",
        showlegend=True,
    )

    return fig


def plot_fft_view(frequencies: ndarray, fft_result: ndarray, serie_name: str) -> Figure:
    fig = Figure()
    fig.add_trace(
        Scatter(
            x=frequencies,
            y=nabs(fft_result),
            mode="lines",
            name=f"{serie_name}",
            line={"color": "rgb(255, 223, 0)", "width": 1.0},
            fill="tonexty",
            fillcolor="rgba(255, 223, 0, .3)",
        )
    )
    fig.update_layout(
        title=f"Fast Fourier Transform of the serie: {serie_name}",
        xaxis_title="Frequency",
        yaxis_title=f"FFT",
        showlegend=True,
        legend=dict(x=1.0, y=1.0),
    )
    return fig


def plot_wavelet_view(widths, t, cwt_result: ndarray, serie_name: str) -> Figure:
    T, S = meshgrid(arange(t), widths)
    fig = Figure(
        data=[
            Surface(x=T, y=S, z=nabs(cwt_result), showscale=False, colorscale="Viridis")
        ]
    )

    fig.update_traces(
        contours_z=dict(
            show=True, usecolormap=True, highlightcolor="chartreuse", project_z=True
        )
    )

    fig.update_layout(
        title=f"Wavelet analysis - {serie_name}",
        autosize=True,
        width=1000,
        height=800,
        scene_camera_eye=dict(x=1.87, y=0.88, z=-0.64),
        margin=dict(l=65, r=50, b=65, t=90),
    )

    fig.update_scenes(
        xaxis_title_text="Time",
        yaxis_title_text="Frequency",
        zaxis_title_text="Magnitude",
    )

    return fig


def plot_density_view(data: DataFrame, serie_name: str) -> Figure:
    group_labels = [serie_name]

    colors = ["orangered"]

    fig = create_distplot(
        [data.loc[:, serie_name]],
        group_labels,
        colors=colors,
        bin_size=[len(data) / 100],
        show_curve=True,
        show_rug=False,
    )

    return fig


def plot_psd_view(frequencies: ndarray, psd: ndarray, serie_name: str) -> Figure:
    fig = Figure()
    fig.add_trace(
        Scatter(
            x=frequencies,
            y=log(psd),
            mode="lines",
            name=f"{serie_name}",
            line={"color": "rgb(127, 255, 0)", "width": 1.0},
            fill="tonexty",
            fillcolor="rgba(127, 255, 0, .3)",
        )
    )
    fig.update_layout(
        title=f"Log of the Power Spectral Density (PSD) of the serie: {serie_name}",
        xaxis_title="Frequency",
        yaxis_title="Log(Power/Frequency) <=> dB/Hz",
        showlegend=True,
        legend=dict(x=1.0, y=1.0),
    )
    return fig


def plot_reducted_dim(reducted_df: DataFrame, reduc_dim_algo: str) -> Figure:
    fig = scatter_3d(
        data_frame=reducted_df,
        x="fst_dim",
        y="snd_dim",
        z="trd_dim",
        hover_name="Name",
        color="Style",
        symbol="Style",
        opacity=0.8,
        height=800,
        hover_data={
            "fst_dim": ":.2f",
            "snd_dim": ":.2f",
            "trd_dim": ":.2f",
            "Style": False,
        },
        color_discrete_sequence=colors.qualitative.Plotly,
        title=f"{reduc_dim_algo} representation",
    )

    fig.update_traces(marker_size=8)
    fig.update_layout(showlegend=True)

    return fig


def plot_correlation_heatmap(top_five: dict) -> Figure:
    fig = Figure(
        data=Heatmap(
            z=[*[x for x in top_five.values()]][::-1],
            customdata=[*[x.values for x in top_five.values()]][::-1],
            hovertemplate="Kendall's τ: %{customdata:.2f}<extra></extra>",
            text=[*[x.index for x in top_five.values()]][::-1],
            texttemplate="%{text}",
            textfont={"size": 15},
            y=list(top_five.keys())[::-1],
            colorbar={"title": "Kendall's τ range"},
            colorscale="viridis",
            x=[f"Top {i}" for i in range(1, 6)],
        ),
    )

    # Add title to the figure
    fig.update_layout(
        title_text="Top 5 Correlated Features for Each Dimension (Kendall's τ)",
        title_font_size=20,
        height=500,
    )

    return fig
