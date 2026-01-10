FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    POETRY_HOME=/opt/poetry

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libjpeg62-turbo \
        libpng16-16 \
        libglib2.0-0 \
        libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir poetry

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --only main --no-ansi --no-root

COPY backend ./backend
COPY data ./data

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "backend.api_app:app", "--host", "0.0.0.0", "--port", "8000"]
