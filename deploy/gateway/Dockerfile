# litmoe gateway - Python service with llama.cpp
FROM python:3.11-slim

WORKDIR /app

# Install build deps + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl git build-essential cmake pkg-config libopenblas-dev libopenblas0 && \
    rm -rf /var/lib/apt/lists/*

# Build llama.cpp from source (CPU-only with BLAS + native CPU features)
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git /tmp/llama && \
    cd /tmp/llama && \
    cmake -B build \
        -DGGML_CUDA=OFF \
        -DGGML_BLAS=ON \
        -DGGML_BLAS_VENDOR=OpenBLAS \
        -DGGML_NATIVE=ON \
        -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --config Release -j --target llama-server && \
    cp build/bin/llama-server /usr/local/bin/ && \
    cp build/bin/lib*.so* /usr/local/lib/ && \
    ldconfig && \
    rm -rf /tmp/llama

# Copy and install the litmoe package
COPY pyproject.toml README.md ./
COPY litmoe/ litmoe/
RUN pip install --no-cache-dir -e .

ENV LITMOE_CONFIG=/config/models.yaml
EXPOSE 8080

CMD ["litmoe", "serve"]