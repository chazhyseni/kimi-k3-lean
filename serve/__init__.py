"""kimi-k3-lean OpenAI-compatible HTTP server.

A drop-in replacement for warp's serve/ that drives libk3.so instead of
libwaste.so. The endpoint surface, request/response shape, and SSE
streaming are the same; the engine underneath is different.

See COMBINED.md for the architecture and INSTALL.md for installation.
"""

__version__ = "0.6.8"