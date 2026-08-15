# kimi-k3-lean model server — the actual inference service.
# Mounts a checkpoint directory at /data, runs serve/__main__.py against it.
#
# This image does NOT include the weights; mount them at runtime.
# The companion compose.yml uses a bind mount or named volume.
FROM python:3.11-slim-bookworm AS build

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src/
RUN LDFLAGS="-lm -pthread" make -j"$(nproc)" all

# --- runtime ---
FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/kimi-k3-lean
COPY --from=build /src/serve/ /opt/kimi-k3-lean/serve/
COPY --from=build /src/bin/libk3.so /opt/kimi-k3-lean/bin/libk3.so
COPY --from=build /src/bin/k3 /opt/kimi-k3-lean/bin/k3

ENV LD_LIBRARY_PATH=/opt/kimi-k3-lean/bin

VOLUME ["/data"]

ENV K3_HOST=0.0.0.0 K3_PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

RUN { \
    echo '#!/bin/sh'; \
    echo 'export LD_LIBRARY_PATH=/opt/kimi-k3-lean/bin:${LD_LIBRARY_PATH}'; \
    echo 'exec python3 -u /opt/kimi-k3-lean/serve/__main__.py /data --host 0.0.0.0 --port 8080 "$@"'; \
} > /opt/kimi-k3-lean/entrypoint.sh && chmod +x /opt/kimi-k3-lean/entrypoint.sh

CMD ["/opt/kimi-k3-lean/entrypoint.sh"]

LABEL org.opencontainers.image.title="kimi-k3-lean-server" \
      org.opencontainers.image.source="https://github.com/chazhyseni/kimi-k3-lean" \
      org.opencontainers.image.licenses="Apache-2.0"