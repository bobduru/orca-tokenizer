FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.6.10 /uv /usr/local/bin/uv

WORKDIR /app

ENV UV_SYSTEM_PYTHON=1

# Install dependencies first (layer is cached unless pyproject.toml/uv.lock change)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --no-dev --no-install-project

# Copy source packages
COPY token_library/ token_library/
COPY ai_classification/ ai_classification/
COPY UI/ UI/

# Copy only the data required for the catalogue page
COPY data/ford-catalogue/merged_online_catalogue_annotated_parsed.csv data/ford-catalogue/
COPY data/ford_paper_spects/ data/ford_paper_spects/
COPY data/ford-catalogue/spects/ data/ford-catalogue/spects/
COPY ["data/ford-catalogue/Northern Resident/", "data/ford-catalogue/Northern Resident/"]

# Install local packages (token_library, ai_classification)
RUN uv sync --no-dev

ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV APP_ENV=prod

EXPOSE 8501

CMD ["uv", "run", "--no-sync", "streamlit", "run", "UI/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
