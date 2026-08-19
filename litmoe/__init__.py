"""litmoe - OpenAI-compatible gateway for ktransformers and llama.cpp.

This is a thin orchestration layer. It does NOT contain inference code.
For inference, it dispatches to:
- ktransformers (https://github.com/kvcache-ai/ktransformers) - GPU accelerated,
  AMX/AVX2/AVX512 CPU kernels, MoE expert offloading
- llama.cpp (https://github.com/ggml-org/llama.cpp) - Cross-platform CUDA/HIP/Metal/Vulkan
"""

__version__ = "0.1.0"
