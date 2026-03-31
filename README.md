# Orca Tokenizer

## Setup

```bash
uv sync --group dev
```

## Running the UI

**Dev** — all pages (Catalogue, Label Calls, Call Query):
```bash
uv run streamlit run UI/streamlit_app.py
```

**Prod** — Catalogue page only:
```bash
APP_ENV=prod uv run streamlit run UI/streamlit_app.py
```

## Secrets

Create `.streamlit/secrets.toml` for the Call Query page (dev only):
```toml
OPENAI_API_KEY = "your-key"
```
