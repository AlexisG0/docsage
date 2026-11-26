FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.5 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra api
COPY src ./src
RUN uv sync --frozen --no-dev --extra api

FROM python:3.12-slim
RUN useradd --create-home --uid 1000 docsage \
    && mkdir /data && chown docsage /data
COPY --from=build /app /app
ENV PATH="/app/.venv/bin:$PATH" DOCSAGE_HOME=/data PYTHONUNBUFFERED=1
USER docsage
WORKDIR /app
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "docsage.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
