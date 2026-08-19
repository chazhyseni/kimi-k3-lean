# litmoe - OpenAI-compatible gateway
# No inference code in this image. Install ktransformers/llama.cpp as needed.

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install
COPY pyproject.toml README.md ./
COPY litmoe ./litmoe
RUN pip install --no-cache-dir -e .

# Optional: install engines
# RUN pip install kt-kernel
# RUN git clone https://github.com/ggml-org/llama.cpp && \
#     cd llama.cpp && cmake -B build -DGGML_CUDA=ON && \
#     cmake --build build --config Release && \
#     cp build/bin/llama-server /usr/local/bin/

# Config
ENV LITMOE_CONFIG=/config/models.yaml
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["litmoe", "serve"]
