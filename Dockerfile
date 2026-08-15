# Dockerfile — runtime image for kimi-k3-lean.
#
# Multi-arch: linux/amd64 + linux/arm64.
# Base: Debian 12 (bookworm) slim. Glibc 2.36, gcc-compatible.
#
# What's in it:
#   - The compiled kimi-k3-lean binaries (libk3.so, k3)
#   - Python 3.11 (for the OpenAI server)
#   - The full source tree (for the convert tool)
#
# What's NOT in it:
#   - The model weights. Mount them as a volume.
#   - The Python server code (yet — Phase C).

# --------------------------------------------------------------- syntax
# syntax=docker/dockerfile:1.7

# --------------------------------------------------------------- build
FROM --platform=$BUILDPLATFORM debian:12-slim AS build

ARG TARGETPLATFORM
ARG TARGETARCH

# Toolchain. gcc, make, cmake for completeness.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    ninja-build \
    python3 \
    python3-pip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

# Copy the source. .dockerignore keeps build artifacts out.
COPY . /src/

# Build.
RUN LDFLAGS="-lm -pthread" make -j"$(nproc)" && \
    cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && \
    cmake --build build --parallel

# Verify the build.
RUN LDFLAGS="-lm -pthread" make test

# --------------------------------------------------------------- runtime
FROM python:3.11-slim-bookworm AS runtime

# Install runtime libs we need: libgomp, libgfortran, ca-certificates.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the built binaries. The Makefile produces bin/libk3.so; CMake
# produces build/lib/libk3.so. Prefer the Makefile output (it's what
# bootstrap.sh and the launcher use).
COPY --from=build /src/bin/k3          /usr/local/bin/k3
COPY --from=build /src/bin/libk3.so    /usr/local/lib/libk3.so

# Copy the headers (for downstream embedding).
COPY --from=build /src/include/libk3/libk3.h /usr/local/include/libk3/libk3.h

# Copy the source tree (for the Python server + convert tool).
WORKDIR /opt/kimi-k3-lean
COPY --from=build /src/ /opt/kimi-k3-lean/

# Update the dynamic linker cache.
RUN ldconfig

# Default: start the OpenAI server on port 8080. Mount a checkpoint
# at /data to get real inference; without it the server still starts
# and /v1/models works (chat returns engine_error).
ENV K3_HOST=0.0.0.0 K3_PORT=8080
EXPOSE 8080
ENTRYPOINT ["python3", "-u", "serve/__main__.py", "/data", "--host", "0.0.0.0", "--port", "8080"]

# --------------------------------------------------------------- labels
LABEL org.opencontainers.image.title="kimi-k3-lean" \
      org.opencontainers.image.description="Lean OpenAI-compatible server for Kimi K3 — disk-resident, CPU-only" \
      org.opencontainers.image.source="https://github.com/chazhyseni/kimi-k3-lean" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.vendor="chazhyseni"