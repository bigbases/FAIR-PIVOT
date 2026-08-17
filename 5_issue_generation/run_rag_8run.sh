#!/bin/bash
# RAG issue_generation: 4 arm × 2 backbone (gemma3-4b + llama3.2-3b),
# base condition only, GENERATION_MODE = answer_plus_self_score (default).
# Each arm = one invocation; both backbones loaded sequentially inside.
set -u
# Run from this script's own directory (5_issue_generation), which holds scripts/.
cd "$(cd "$(dirname "$0")" && pwd)"

export IFV1_BACKBONES=gemma3-4b,llama3.2-3b
export IFV1_CONDITIONS=base
export IFV1_RUN_NAME=fairsynth_3arm_v1
export IFV1_RAG=true

echo "=== START $(date) ==="

echo; echo "=== ARM 1/4: pairs + original_only  [4']  $(date) ==="
IFV1_RAG_MODE=pairs IFV1_RAG_INDEX=original_only python scripts/generate_issue_eval.py

echo; echo "=== ARM 2/4: pairs + fair           [5]   $(date) ==="
IFV1_RAG_MODE=pairs IFV1_RAG_INDEX=fair          python scripts/generate_issue_eval.py

echo; echo "=== ARM 3/4: topk + original_only   [4]   $(date) ==="
IFV1_RAG_MODE=topk IFV1_RAG_INDEX=original_only  python scripts/generate_issue_eval.py

echo; echo "=== ARM 4/4: political + fair       [5']  $(date) ==="
IFV1_RAG_MODE=political IFV1_RAG_INDEX=fair      python scripts/generate_issue_eval.py

echo; echo "=== ALL DONE $(date) ==="
