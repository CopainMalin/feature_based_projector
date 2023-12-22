from streamlit import set_page_config, title, write

set_page_config(
    page_title="Home page !",
)
title("Welcome to the :blue[Feature based projector] app")
write("> First, load your dataset in the :orange[Dataset Loading] page.")
write(
    "> Then, you can analyze your time series locally (serie per serie) on the :blue[Local Analysis] page ! :male-detective:"
)
write(
    "> Or you can also visualize the whole dataset on the :orange[Global Analysis] page ! :male-detective:"
)
write("> Have fun ! :tada:")


# TODO : DOCSTRING
# TODO : Index.py
# TODO : storing the app on docker
