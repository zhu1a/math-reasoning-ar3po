#!/bin/bash
set -euo pipefail

# 数学评测入口。历史文件名保留兼容（原名做的是量化，现已改为生成式评测）。
#
# 用法:
#   bash scripts/run_eval_quantize.sh <模型路径> [avg@k]
#
# 默认 avg@8（AR3PO/DAPO 论文口径）。冒烟测试可传 1（贪心、确定性、最快）。
#   bash scripts/run_eval_quantize.sh outputs-xxx-merged 1     # 快速回归
#   bash scripts/run_eval_quantize.sh outputs-xxx-merged 8     # 对齐论文
#
# 四个数据集的角色（缺任何一个都会误判）：
#   dapo_holdout   与训练集同源  → 看有没有过拟合
#   math500        英文主指标    → 可与论文直接对比
#   aime2024       高难度        → 题量少(30)，k 要大
#   agieval_gaokao 中文高考选择题 → 走独立的 MCQ 字母集合判分

MODEL_PATH="${1:-outputs-dpo-qwen3-8b-math-merged}"
K="${2:-8}"

if [[ ( "$MODEL_PATH" == /* || "$MODEL_PATH" == ./* || "$MODEL_PATH" == outputs-* ) \
      && ! -f "$MODEL_PATH/config.json" ]]; then
    echo "模型不存在: $MODEL_PATH" >&2
    echo "用法: bash scripts/run_eval_quantize.sh <模型路径> [avg@k]" >&2
    exit 1
fi

OUT_DIR="outputs-eval/$(basename "$MODEL_PATH")"
mkdir -p "$OUT_DIR"
echo "模型: $MODEL_PATH   avg@$K   明细输出: $OUT_DIR/"

run_one () {  # data_file  max_new_tokens  batch
    local data="$1" mnt="$2" bs="$3"
    local name
    name="$(basename "$data" .jsonl)"
    echo ""
    echo "==================== $name ===================="
    python3 tools/evaluate_math.py \
        --model "$MODEL_PATH" \
        --data "$data" \
        --num_samples "$K" \
        --batch_size "$bs" \
        --max_new_tokens "$mnt" \
        --output "$OUT_DIR/$name.jsonl" || true   # 截断严重时脚本返回 2，不要让 set -e 中断后续
}

#                       数据文件                          max_new  batch
run_one data/eval/dapo_holdout.jsonl                       2048     8
run_one data/eval/math500.jsonl                            2048     8
run_one data/eval/agieval_gaokao.jsonl                     2048     8
# AIME 只有 30 题，用更大的 k 和更长的输出；batch 调小防显存峰值
run_one data/eval/aime2024.jsonl                           4096     4

echo ""
echo "全部完成。明细在 $OUT_DIR/"
echo "对照建议：base / SFT / 最终模型 各跑一遍，否则无法判断提升来自哪一步。"
