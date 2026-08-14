# kimi-k3-lean model server — the actual inference service.
# Mounts a checkpoint directory at /data, runs serve/__main__.py against it.
#
# This image does NOT include the weights; mount them at runtime.
# The companion compose.yml uses a bind mount or named volume.
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System deps for the C engine (libgomp = OpenMP runtime; the article's
# kernels are OpenMP-parallelized).
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/kimi-k3-lean

# Just the runtime pieces (not tests, not docs, not packaging).
COPY serve/ /opt/kimi-k3-lean/serve/
COPY include/ /opt/k3-include/
COPY bin/libk3.so /opt/kimi-k3-lean/bin/libk3.so
COPY bin/k3 /opt/kimi-k3-lean/bin/k3

ENV LD_LIBRARY_PATH=/opt/kimi-k3-lean/bin

# /data is the mount point for the checkpoint.
WORKDIR /data
VOLUME ["/data"]

# Default port + args. The compose service overrides these as needed.
ENV K3_HOST=0.0.0.0 K3_PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

# The server reads K3_API_KEY from env (set by compose).
CMD ["python3", "-u", "/opt/kimi-k3-lean/serve/__main__.py", \
     "/data", \
     "--host", "0.0.0.0", "--port", "8080"]

LABEL org.opencontainers.image.title="kimi-k3-lean-server" \
      org.opencontainers.image.source="https://github.com/chazhyseni/kimi-k3-lean" \
      org.opencontainers.image.licenses="Apache-2.0"