#!/bin/bash
set -euo pipefail
# ============================================================================
# DPO 偏好优化（Qwen3-8B，单卡 48GB，LoRA）
# ============================================================================
# 在整条流水线里的位置：
#   SFT → GRPO(+AR3PO, 规则奖励) → 【本步 DPO】
#
# DPO 的偏好对来自 GRPO 的 rollout 自产，零额外标注成本：
#   先跑 tools/export_dpo_pairs.py 生成 data/reward/math_dpo_pairs.jsonl
#   它把同一道题的多次采样按 (准确性, 规范性) 排序，取最好/最差组成 chosen/rejected
#
# 与 GRPO 的分工：
#   GRPO 负责「探索 + 提升上限」（规则奖励，可验证题）
#    DPO 负责「稳定 + 压低方差」（相对比较，不依赖绝对奖励）
#
# 数据 schema（由 dpo_training.py 决定，勿改）：
#   {"conversations": [{"from":"human","value":...}], "chosen": str, "rejected": str}
# ============================================================================

GRPO_MODEL=outputs-grpo-qwen3-8b-ar3po-merged
if [[ ! -f "$GRPO_MODEL/config.json" ]]; then
    echo "缺少 $GRPO_MODEL；请先完成 GRPO 并合并 adapter。" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python3 training/dpo_training.py \
    --model_name_or_path "$GRPO_MODEL" \
    --train_file_dir ./data/reward \
    --validation_split_percentage 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples -1 \
    --max_eval_samples 20 \
    --max_steps 300 \
    --eval_steps 50 \
    --save_steps 100 \
    --save_total_limit 3 \
    --max_source_length 2048 \
    --max_target_length 2048 \
    --learning_rate 5e-7 \
    --warmup_steps 10 \
    --output_dir outputs-dpo-qwen3-8b-math \
    --target_modules all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --fp16 False \
    --report_to tensorboard \
    --remove_unused_columns False \
    --gradient_checkpointing True \
    --tool_format default \
    --cache_dir ./cache

echo "合并 DPO LoRA adapter，生成最终可直接推理的模型..."
python3 tools/merge_peft_adapter.py \
    --base_model "$GRPO_MODEL" \
    --tokenizer_path "$GRPO_MODEL" \
    --lora_model outputs-dpo-qwen3-8b-math \
    --output_dir outputs-dpo-qwen3-8b-math-merged

echo "DPO 完成并已合并! 最终模型: outputs-dpo-qwen3-8b-math-merged"
