#!/usr/bin/env bash
# vLLM OpenAI-compatible server for Qwen3.5-35B-A3B (bf16)
# Used by multi_agents.py (Outline + Generation agents).
#
# Usage:
#   bash serve_qwen35.sh
# or with tmux for laptop-close safety:
#   tmux new -s qwen35 'bash 2_rag/agents/serve_qwen35.sh'
#
# Server listens on http://localhost:8000/v1 (OpenAI-compatible).
# multi_agents.py points VLLM_BASE_URL there.

set -euo pipefail

MODEL="Qwen/Qwen3.5-35B-A3B"

# 16K context is enough: 5 retrieved docs (~1500-2500 chars each) + system/user
# prompt + ~9000-char content output stays well under this.
# (Qwen3.5 default is 262K but we don't need it for synth gen.)
MAX_MODEL_LEN=16384

# 35B bf16 weights ~70GB + KV cache. 92% of 97GB VRAM = ~89GB usable.
GPU_MEM_UTIL=0.92

vllm serve "${MODEL}" \
  --host 0.0.0.0 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len ${MAX_MODEL_LEN} \
  --gpu-memory-utilization ${GPU_MEM_UTIL} \
  --tensor-parallel-size 1 \
  --enable-prefix-caching \
  --attention-backend TRITON_ATTN \
  --mm-encoder-attn-backend TORCH_SDPA \
  --max-num-seqs 256 \
  --served-model-name "${MODEL}"
