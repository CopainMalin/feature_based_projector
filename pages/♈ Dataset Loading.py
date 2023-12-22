from streamlit import (
    title,
    session_state,
    write,
    dataframe,
    toggle,
    columns,
    number_input,
    button,
    spinner,
    success,
    set_page_config,
)
from pandas import DataFrame, date_range
from numpy import arange
from numpy.random import randn
from src.space_projection import compute_tsfeatures
from src.utils import load_data, transform_dataset, inject_toy_series

set_page_config(page_title="Dataset Loading")
title(":green[Dataset management] page 💾")

tuto = toggle("Show guide")
if tuto:
    write("Dataset must be of the form :")
    dataframe(
        DataFrame(
            data={
                "unique_id": [*["serie_1"] * 5, *["serie_2"] * 5],
                "ds": [
                    *arange(1, 6),
                    *date_range(start="2023-01-01", periods=5, freq="M"),
                ],
                "y": [*randn(5), *randn(5)],
            }
        )
    )
    write(
        ":blue['unique_id'] must be a string, :blue['ds'] an int or a datetime and :blue['y'] must be numeric."
    )
    write("There is no limit to the number of series you can load.")
    write("The file format must be :orange[.xlsx] or :orange[.csv]. :wink:")

    write("The app can also handle and transform data of the form :")
    dataframe(
        DataFrame(
            data={
                "date": date_range(start="2023-01-01", periods=5, freq="M"),
                "serie_1": randn(5),
                "serie_2": randn(5),
                "serie_3": randn(5),
                "serie_4": randn(5),
            },
        )
    )
    write(
        "To do so, load your file in the app and press the :orange[transform] button !"
    )
    write(
        ":warning: The date column name must be 'date' so the algorithm recognizes it."
    )

dataset = load_data()

if dataset is not None:
    c_left, _, c_right = columns([0.35, 0.1, 0.55])
    with c_left:
        period = number_input(label="Enter the seasonal period :", min_value=1)
        transform = button("Transform and compute")
        if transform:
            dataset = transform_dataset(dataset)
    with c_right:
        dataframe(dataset)
    if (
        ("unique_id" in dataset.columns)
        & ("ds" in dataset.columns)
        & ("y" in dataset.columns)
        & (period > 0)
    ):
        with c_left:
            with spinner("Features computation"):
                session_state["dataset"] = inject_toy_series(dataset, freq=period)
                session_state["features"] = compute_tsfeatures(
                    df=session_state["dataset"], freq=period, fill_value=0
                )
                session_state["data_loaded"] = True
                session_state["next_stage"] = True
    with c_left:
        if "next_stage" in session_state:
            success(":green[Loading complete] ✅.")
