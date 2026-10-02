# One image for the three Python services (ingest, api, ui); the command decides which runs.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY .streamlit ./.streamlit
RUN pip install .

# Run as an unprivileged user. /app/data is created here so the named volume
# mounted on it inherits this ownership.
RUN useradd --create-home --uid 1000 app \
    && mkdir -p /app/data \
    && chown -R app:app /app
USER app

EXPOSE 8000 8501

CMD ["python", "-m", "rag_assistant.api"]
