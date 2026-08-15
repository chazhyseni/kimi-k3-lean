# litMoE model server — the actual inference service.
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

WORKDIR /opt/litMoE
COPY --from=build /src/serve/ /opt/litMoE/serve/
COPY --from=build /src/bin/liblitmoe.so /opt/litMoE/bin/liblitmoe.so
COPY --from=build /src/bin/litmoe /opt/litMoE/bin/litmoe

ENV LD_LIBRARY_PATH=/opt/litMoE/bin

VOLUME ["/data"]

ENV LITMOE_HOST=0.0.0.0 LITMOE_PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8080/health || exit 1

RUN { \
    echo '#!/bin/sh'; \
    echo 'export LD_LIBRARY_PATH=/opt/litMoE/bin:${LD_LIBRARY_PATH}'; \
    echo 'exec python3 -u /opt/litMoE/serve/__main__.py /data --host 0.0.0.0 --port 8080 "$@"'; \
} > /opt/litMoE/entrypoint.sh && chmod +x /opt/litMoE/entrypoint.sh

CMD ["/opt/litMoE/entrypoint.sh"]

LABEL org.opencontainers.image.title="litMoE-server" \
      org.opencontainers.image.source="https://github.com/chazhyseni/litMoE" \
      org.opencontainers.image.licenses="Apache-2.0"