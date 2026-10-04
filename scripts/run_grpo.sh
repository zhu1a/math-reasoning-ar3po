#!/bin/bash
set -euo pipefail
# ============================================================================
# GRPO + AR3PO 数学推理训练（Qwen3-8B，单卡 48GB）
# ============================================================================
# 算法：AR3PO (arXiv:2509.25808) 自适应 rollout
#       loss 用 DAPO 式 token-level + clip-higher
# 奖励：整体给分(accuracy) + 分层规范分(format) + DAPO 软长度惩罚(length)
#       纯规则函数 → 不需要 reward model，不需要 critic，省显存
#
# 显存要点（48GB 单卡口径）：
#   · LoRA + gradient_checkpointing 必须同时开
#     （原脚本会在用 LoRA 时强制关掉 GC，是 8B 必 OOM 的元凶，已修正）
#   · scale_rewards 必须设 none —— AR3PO 的「复制补齐」机制要求；
#     用默认的 group 会让补齐组的 advantage 幅度被统一放大 ~1.22×
#   · beta=0 —— RLVR 标准做法，省掉 reference 前向
#   · use_vllm colocate + sleep mode 省显存
#
# 数据（已下载并转换好，见 tools/build_math_dataset.py）：
#   ⚠️ GRPO 的三个数据集必须用【不同子目录】：--train_file_dir 会把目录下
#      所有 jsonl 合并加载，schema 不同会直接抛 CastError。
#      data/grpo/en/    ← 阶段一 复现论文（英文 DAPO-Math，先跑这个）
#      data/grpo/zh/    ← 阶段二 中文迁移
#      data/grpo/proof/ ← 证明题（verifiable=False，默认不进 GRPO，只作评测）
#
# 若 OOM，按此顺序降级：num_generations 8→4 → max_completion_length 2048→1024
#                      → vllm_gpu_memory_utilization 降到 0.25 → 换 Qwen3-4B
# ============================================================================

SFT_MODEL=outputs-sft-qwen3-8b-math-merged
if [[ ! -f "$SFT_MODEL/config.json" ]]; then
    echo "缺少 $SFT_MODEL；请先运行 bash scripts/run_sft.sh 完成 SFT 和 LoRA 合并。" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=0 python3 -m training.grpo_training \
    --model_name_or_path "$SFT_MODEL" \
    --train_file_dir data/grpo/en \
    --train_samples -1 \
    --max_steps -1 --num_train_epochs 1 \
    --save_steps 50 \
    --save_strategy steps \
    --save_total_limit 5 \
    --output_dir outputs-grpo-qwen3-8b-ar3po \
    --dtype bfloat16 \
    --bf16 True \
    --report_to tensorboard \
    --remove_unused_columns False \
    --logging_steps 10 \
    \
    `# ---- 显存与吞吐 ----` \
    --gradient_checkpointing True \
    --use_vllm True \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.35 \
    --vllm_enable_sleep_mode True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --generation_batch_size 32 \
    --num_generations 8 \
    \
    `# ---- 奖励（reward_funcs 由 training/math_rewards.py 提供）----` \
    --reward_weights 1.0 0.3 1.0 \
    --format_weight 0.3 \
    --length_weight 1.0 \
    --mask_truncated_completions True \
    \
    `# ---- 优化器 ----` \
    --learning_rate 1.0e-6 \
    --lr_scheduler_type cosine \
    --warmup_steps 0 \
    --beta 0.0 \
    \
    `# ---- DAPO 损失 ----` \
    --loss_type dapo \
    --epsilon 0.2 \
    --epsilon_high 0.28 \
    --scale_rewards none \
    --max_completion_length 2048 \
    \
    `# ---- AR3PO 自适应 rollout ----` \
    --ar3po_stages 2 \
    --ar3po_correctness_reward_idx 0 \
    `# TRL 0.29.1 replay mixin 会丢掉 AR3PO 停梯度 mask，必须关闭` \
    --replay_buffer_size 0 \
    \
    `# ---- LoRA ----` \
    --use_peft True \
    --qlora False \
    --load_in_4bit False \
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05

echo "合并 GRPO LoRA adapter，供 rollout 导出和 DPO 使用..."
python3 tools/merge_peft_adapter.py \
    --base_model "$SFT_MODEL" \
    --tokenizer_path "$SFT_MODEL" \
    --lora_model outputs-grpo-qwen3-8b-ar3po \
    --output_dir outputs-grpo-qwen3-8b-ar3po-merged

echo "GRPO 训练完成并已合并!"
