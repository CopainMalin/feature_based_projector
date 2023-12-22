FROM python:3.11
WORKDIR /app


# Install dependencies required for Poetry
RUN apt-get update && apt-get install -y curl

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/usr/local python -

# Add Poetry to the system's PATH
ENV PATH="${PATH}:/usr/local/bin"

# Only copying usefull files
COPY src /app/src
COPY pages /app/pages
COPY Home_page.py /app/Home_page.py
COPY pyproject.toml poetry.lock /app/

# Using poetry to install the requirements
RUN poetry install --without dev
# run the app
CMD ["poetry", "run", "streamlit", "run", "Home_page.py"]
