#!/bin/bash
set -euo pipefail
# ============================================================================
# SFT 冷启动（Qwen3-8B，单卡 48GB，LoRA）
# ============================================================================
# 为什么需要 SFT：GRPO 不能从 base/instruct 直接裸跑。
# 「Secrets of RLHF Part I」(arXiv:2307.04964) 实验证明：
#   跳过 SFT 直接做 RL 会失败，语言建模能力严重退化。
# 本项目用 data/sft/math_sft_demo.jsonl 做轻量冷启动，教会模型
# 「分步推理 + 把答案放进 \boxed{}」这个输出习惯。
#
# 注意：原脚本把注释写在 `\` 后续行里，会导致后续参数被 bash 当成注释吃掉，
#       实际上一条参数都没传进去。已修正。
# ============================================================================

CUDA_VISIBLE_DEVICES=0 python3 training/supervised_finetuning.py \
    --model_name_or_path Qwen/Qwen3-8B \
    --train_file_dir ./data/sft \
    --validation_split_percentage 5 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --do_train \
    --do_eval \
    --use_peft True \
    --max_train_samples -1 \
    --max_eval_samples 20 \
    --model_max_length 2048 \
    --num_train_epochs 2 \
    --learning_rate 2e-5 \
    --warmup_steps 10 \
    --weight_decay 0.05 \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_steps 50 \
    --eval_strategy steps \
    --save_steps 200 \
    --save_strategy steps \
    --save_total_limit 3 \
    --gradient_accumulation_steps 4 \
    --preprocessing_num_workers 4 \
    --output_dir outputs-sft-qwen3-8b-math \
    --logging_first_step True \
    --target_modules all \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --bf16 True \
    --report_to tensorboard \
    --gradient_checkpointing True \
    --tool_format default \
    --cache_dir ./cache

echo "合并 SFT LoRA adapter，供 GRPO 作为真正的起点..."
python3 tools/merge_peft_adapter.py \
    --base_model Qwen/Qwen3-8B \
    --tokenizer_path Qwen/Qwen3-8B \
    --lora_model outputs-sft-qwen3-8b-math \
    --output_dir outputs-sft-qwen3-8b-math-merged

echo "SFT 完成并已合并! 下一步: bash scripts/run_grpo.sh"
