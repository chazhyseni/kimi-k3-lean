# kimi-k3-lean router — multi-model OpenAI-compatible router.
# Sits in front of one or more model backends.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request, sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthz', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]

LABEL org.opencontainers.image.title="kimi-k3-lean-router" \
      org.opencontainers.image.source="https://github.com/chazhyseni/kimi-k3-lean" \
      org.opencontainers.image.licenses="Apache-2.0"