#!/usr/bin/env bash
# 6 finetunings sequential: balanced(gemma,qwen) -> left(gemma,qwen) -> right(gemma,qwen)
set -u
cd /workspace/FAIR_SYNTH_Haneul/3_downstream_task/fine_tuning

run_one () {
  local backbone=$1 dataset=$2 script=$3 model_dir=$4
  local ts=$(date +%Y%m%d_%H%M%S)
  local out=checkpoints_v2_${dataset}/${model_dir}_${dataset}_${ts}
  local log=logs/v2_${dataset}_${model_dir}_${ts}.log
  echo "=== START ${model_dir} / ${dataset} @ $(date) ==="
  python3 ${script} \
    --data_path finetuning_data_v2_${dataset}/${dataset}_${backbone}_tokenchunk.jsonl \
    --output_dir ${out} \
    --use_lora --use_qlora > ${log} 2>&1
  echo "=== DONE ${model_dir} / ${dataset} @ $(date) (rc=$?) ==="
}

# balanced first
run_one gemma balanced models/gemma3-4b/train_gemma.py gemma3-4b
run_one qwen  balanced models/qwen3-4b/train_qwen.py   qwen3-4b
# left
run_one gemma left_extreme models/gemma3-4b/train_gemma.py gemma3-4b
run_one qwen  left_extreme models/qwen3-4b/train_qwen.py   qwen3-4b
# right
run_one gemma right_extreme models/gemma3-4b/train_gemma.py gemma3-4b
run_one qwen  right_extreme models/qwen3-4b/train_qwen.py   qwen3-4b

echo "=== ALL 6 DONE @ $(date) ==="
touch /tmp/v2_extremes_all_done
