# -*- coding: utf-8 -*-
"""从 GRPO 的 rollout 里自产 DPO 偏好对。

===========================================================================
为什么这样做（而不是请强模型标注）
===========================================================================
DPO 需要 (prompt, chosen, rejected) 三元组。常规做法是找强模型打分，
但本项目的约束是「只用可下载的公开数据 + 不额外常驻模型」。

好在 GRPO 的 rollout **本身就是天然的偏好对来源**：
    同一道题采样出的 8 条回答 → 按 (accuracy, format) 排序
    chosen   = 答对且格式最好的
    rejected = 答错且格式最差的
零额外标注成本，而且**比强模型标注更贴近当前策略的分布**
（这是在线偏好对相对离线标注的核心优势）。

===========================================================================
输入 / 输出 schema
===========================================================================
输入：training/grpo_training.py 训练时用 --log_completions 落盘的 rollout jsonl，
      或本脚本直接用 transformers 采样生成的 jsonl（见 --rollouts）。
      每行：{"prompt": str 或 list, "completions": [str, ...], "answer": str}

输出：data/reward/math_dpo_pairs.jsonl —— 匹配 dpo_training.py 期望的 schema
      {"conversations": [{"from":"human","value":...}],
       "chosen": str, "rejected": str}

用法：
    # 方式一：从已有 rollout 文件转换
    python3 tools/export_dpo_pairs.py --rollouts rollouts.jsonl --out data/reward/math_dpo_pairs.jsonl

    # 方式二：直接对数据集本地采样生成（最慢，但完全不依赖外部模型）
    python3 tools/export_dpo_pairs.py --gen --model outputs-grpo-qwen3-8b-ar3po-merged \
        --data data/grpo/zh/math_zh_train.jsonl --limit 2000
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.math_rewards import accuracy_reward, make_format_reward  # noqa: E402


def _as_conversations(prompt) -> list:
    """把 prompt 统一成 dpo_training.py 期望的 conversations 格式。"""
    if isinstance(prompt, str):
        return [{"from": "human", "value": prompt}]
    if isinstance(prompt, list):
        out = []
        for m in prompt:
            role = str(m.get("role", m.get("from", "user"))).lower()
            out.append({
                "from": "human" if role in ("user", "human") else "gpt",
                "value": m.get("content", m.get("value", "")),
            })
        return out
    return [{"from": "human", "value": str(prompt)}]


def build_pairs_from_group(prompt, completions: List[str], gold: str,
                           fmt_fn, min_gap: float = 0.0,
                           keep_meta: bool = False) -> Optional[dict]:
    """把一组 rollout 变成一条偏好对。

    排序键：(accuracy, format)。要求 chosen 的 accuracy 严格高于 rejected
    或（accuracy 相同时）format 分明显更高——否则这对偏好是噪声。
    """
    if not completions or len(completions) < 2:
        return None

    convs = [[{"role": "assistant", "content": c}] for c in completions]
    accs = accuracy_reward(convs, [gold] * len(convs))
    fmts = fmt_fn(convs)

    scored = sorted(
        zip(completions, accs, fmts),
        key=lambda t: (t[1], t[2]),
        reverse=True,
    )
    best_c, best_a, best_f = scored[0]
    worst_c, worst_a, worst_f = scored[-1]

    # 过滤掉「无信息量」的偏好对（两条都错且格式一样 / 两条都对且格式一样）
    if best_a == worst_a and (best_f - worst_f) < max(min_gap, 0.2):
        return None
    if best_a == 0.0 and worst_a == 0.0 and best_f == 0.0:
        return None

    pair = {
        "conversations": _as_conversations(prompt),
        "chosen": best_c,
        "rejected": worst_c,
    }
    if keep_meta:  # 默认不写：避免多余字段干扰 dpo_training.py 的预处理
        pair["_meta"] = {"acc_chosen": best_a, "acc_rejected": worst_a,
                         "fmt_chosen": best_f, "fmt_rejected": worst_f}
    return pair


def iter_rollouts(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def generate_rollouts(model_name: str, data_path: Path, limit: int, n: int,
                      max_new_tokens: int, temperature: float):
    """本地采样 rollout（不依赖外部强模型）。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    rows = [json.loads(l) for l in data_path.open(encoding="utf-8")][:limit]
    for i, r in enumerate(rows):
        messages = [{"role": "user", "content": r["question"]}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs, do_sample=True, temperature=temperature, top_p=0.95,
                num_return_sequences=n, max_new_tokens=max_new_tokens,
                pad_token_id=tok.pad_token_id,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        comps = tok.batch_decode(gen, skip_special_tokens=True)
        yield {"prompt": r["question"], "answer": r["answer"], "completions": comps}
        if (i + 1) % 50 == 0:
            print(f"  已采样 {i+1}/{len(rows)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", type=Path, default=None,
                    help="已有的 rollout jsonl（含 prompt/completions/answer）")
    ap.add_argument("--gen", action="store_true", help="本地采样生成 rollout")
    ap.add_argument(
        "--model",
        default="outputs-grpo-qwen3-8b-ar3po-merged",
        help="用于生成偏好对的已合并 GRPO 模型（不能退回原始 base model）",
    )
    ap.add_argument("--data", type=Path, default=Path("data/grpo/zh/math_zh_train.jsonl"))
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--num_samples", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", type=Path, default=Path("data/reward/math_dpo_pairs.jsonl"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--keep_meta", action="store_true",
                    help="额外写入 _meta 字段（acc/fmt 分数），便于事后分析")
    args = ap.parse_args()

    fmt_fn = make_format_reward(think_end_token_id=None)
    if args.rollouts:
        source = iter_rollouts(args.rollouts)
    elif args.gen:
        source = generate_rollouts(args.model, args.data, args.limit,
                                   args.num_samples, args.max_new_tokens, args.temperature)
    else:
        ap.error("必须指定 --rollouts 或 --gen")

    pairs, total = [], 0
    for rec in source:
        total += 1
        p = build_pairs_from_group(rec.get("prompt"), rec.get("completions", []),
                                   str(rec.get("answer", "")), fmt_fn,
                                   keep_meta=args.keep_meta)
        if p:
            pairs.append(p)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    with args.out.open("w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n输入组数 {total} → 产出偏好对 {len(pairs)} 条（过滤掉 {total-len(pairs)} 条无信息量的）")
    print(f"已写入 {args.out}")
    print("下一步:  bash scripts/run_dpo.sh")


if __name__ == "__main__":
    main()
