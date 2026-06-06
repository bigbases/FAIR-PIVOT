#!/usr/bin/env bash
# vLLM OpenAI-compatible server for Qwen3-30B-A3B-Instruct-2507
# Used by multi_agents.py (Outline + Generation agents) to replace gpt-oss-20b.
#
# Usage:
#   bash serve_qwen3.sh
# or with tmux for laptop-close safety:
#   tmux new -s qwen3 'bash 2_rag/agents/serve_qwen3.sh'
#
# Server listens on http://localhost:8000/v1 (OpenAI-compatible).
# multi_agents.py points VLLM_BASE_URL there.

set -euo pipefail

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# 16K context is enough: 5 retrieved docs (~1500-2500 chars each) + system/user
# prompt + ~9000-char content output stays well under this.
MAX_MODEL_LEN=16384

# Use 92% of 97GB VRAM for weights + KV cache. Leave headroom for activations.
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
  --served-model-name "${MODEL}"
