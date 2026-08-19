# litmoe gateway - Python service
FROM python:3.11-slim

WORKDIR /app

# Copy and install the litmoe package
COPY pyproject.toml README.md ./
COPY litmoe/ litmoe/
RUN pip install --no-cache-dir -e .

# Install runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# The gateway needs to reach engine subprocesses; it does NOT need kt-kernel or llama.cpp itself.
# Engines are installed separately in their own containers or on the host.

ENV LITMOE_CONFIG=/config/models.yaml
EXPOSE 8080

CMD ["litmoe", "serve"]
