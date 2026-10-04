# data/reward/ —— DPO 偏好对

这个目录**初始是空的**，不是遗漏。DPO 的偏好对需要由你自己生成。

## 为什么要自产

DPO 需要 `(prompt, chosen, rejected)` 三元组。常规做法是找强模型打分，但本项目
的约束是「只用可下载的公开数据 + 不额外常驻模型」。

好在 **GRPO 的 rollout 本身就是天然的偏好对来源**：同一道题采样多条回答，
按 `(准确性, 规范性)` 排序，取最好/最差即可组成偏好对——零额外标注成本，
而且比强模型标注**更贴近当前策略的分布**。

## 怎么生成

```bash
# 方式一（推荐）：本地采样生成，不依赖任何外部模型
python3 tools/export_dpo_pairs.py --gen \
    --model outputs-sft-qwen3-8b-math \
    --data data/grpo/en/math_dapo_train.jsonl \
    --limit 2000 \
    --out data/reward/math_dpo_pairs.jsonl

# 方式二：从已有的 rollout 文件转换（需要 log_completions 落盘）
python3 tools/export_dpo_pairs.py \
    --rollouts rollouts.jsonl \
    --out data/reward/math_dpo_pairs.jsonl

# 生成后先校验 schema
python3 tools/validate_jsonl.py --file_path data/reward/math_dpo_pairs.jsonl
```

## schema

```json
{"conversations": [{"from": "human", "value": "题目..."}],
 "chosen":   "正确的推理与答案",
 "rejected": "错误的推理与答案"}
```

由 `training/dpo_training.py` 的 `_build_prompt_from_conversations` 消费，
**不要改字段名**。

## 两个防呆机制

1. `training/dpo_training.py` 里的 `validate_preference_dataset()` 会拒绝
   「占位/退化数据」——例如所有行的 `rejected` 都是同一句话。那种数据上训 DPO
   只会让模型学会区分模板，而不是学会数学。
2. 目录为空时会直接报错并提示上面的生成命令，而不是抛出晦涩的 `KeyError`。

> ⚠️ 历史提示：本仓库曾经把这个文件提交成了用**伪造 rollout** 生成的占位数据
> （`rejected` 全部是「随便写点东西」），已删除。请按上面命令重新生成真实数据。
