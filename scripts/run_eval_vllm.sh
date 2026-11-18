#!/bin/bash
set -euo pipefail

# 单数据集 vLLM 数学评测入口。
# 用法：bash scripts/run_eval_vllm.sh MODEL DATA OUTPUT [K] [MAX_NEW_TOKENS]
MODEL="${1:?需要模型路径}"
DATA="${2:?需要评测 JSONL}"
OUTPUT="${3:?需要输出 JSONL}"
K="${4:-1}"
MAX_NEW_TOKENS="${5:-4096}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
REQUEST_BATCH_SIZE="${VLLM_REQUEST_BATCH_SIZE:-32}"

if (( MAX_NEW_TOKENS <= 4096 )); then
    MAX_MODEL_LEN=8192
else
    MAX_MODEL_LEN=12288
fi

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 tools/evaluate_math_vllm.py \
    --model "$MODEL" \
    --data "$DATA" \
    --output "$OUTPUT" \
    --enable_thinking true \
    --num_samples "$K" \
    --temperature "$(if (( K > 1 )); then echo 0.6; else echo 0; fi)" \
    --top_p 0.95 \
    --top_k 20 \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_model_len "$MAX_MODEL_LEN" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --max_num_seqs "$MAX_NUM_SEQS" \
    --request_batch_size "$REQUEST_BATCH_SIZE"
