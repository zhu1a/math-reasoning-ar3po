# -*- coding: utf-8 -*-
"""数学评测：生成答案 + 符号等价判定，输出 avg@k 准确率。

===========================================================================
为什么需要 avg@k（而不是单次贪心）
===========================================================================
推理模型的单次评测方差极大。AR3PO / DAPO 等论文的标准口径是：
    MATH-500 / Minerva / OlympiadBench  →  avg@8
    AIME24（只有 30 题）                  →  avg@64
不按这个口径跑，你的数字**无法和论文对比**，也无法区分「方法有效」和「运气好」。

===========================================================================
用法
===========================================================================
    # 快速冒烟（单次贪心，最快）
    python3 tools/evaluate_math.py --model <path> --data data/eval/math500.jsonl \
        --num_samples 1 --batch_size 4

    # 对齐论文口径
    python3 tools/evaluate_math.py --model <path> --data data/eval/math500.jsonl \
        --num_samples 8 --batch_size 4 --output outputs/eval_math500.jsonl
    python3 tools/evaluate_math.py --model <path> --data data/eval/aime2024.jsonl \
        --num_samples 64 --batch_size 8 --max_new_tokens 4096

说明：
  · avg@k 的口径是「同一道题采样 k 次，命中率取平均，再对题目取平均」
  · num_samples 只控制每题回答数；temperature 默认 0（贪心），两者互不改写
  · 会额外报告截断比例——截断率高说明 max_new_tokens 不够，
    准确率会被系统性低估，此时应先调大它再比较
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# 允许从任意目录运行（不依赖 CWD 是仓库根目录）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from training.math_rewards import (  # noqa: E402
    accuracy_reward, extract_mcq_letters, make_format_reward, mcq_accuracy_reward,
)


def parse_args():
    p = argparse.ArgumentParser(description="Math evaluation with avg@k.")
    p.add_argument("--model", required=True, help="模型路径或 HF 名称")
    p.add_argument("--data", type=Path, required=True, help="评测集 jsonl（需 question/answer）")
    p.add_argument("--limit", type=int, default=-1, help="只评测前 N 条，-1 为全部")
    p.add_argument("--num_samples", type=int, default=1, help="每题采样次数 k（avg@k）")
    p.add_argument("--batch_size", type=int, default=4, help="每次前向的题目数")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="采样温度；0 为贪心，>0 启用采样（默认 0）")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--enable_thinking", type=str.lower, choices=("true", "false"), default="true",
                   help="显式传给 chat template（默认 true）")
    p.add_argument("--output", type=Path, help="把每题明细写成 jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    temperature = args.temperature
    if temperature < 0:
        raise ValueError("--temperature 必须大于等于 0")
    if args.num_samples < 1:
        raise ValueError("--num_samples 必须大于等于 1")
    if not 0.0 <= args.top_p <= 1.0:
        raise ValueError("--top_p 必须在 [0, 1] 范围内")
    if args.top_k < 0:
        raise ValueError("--top_k 必须大于等于 0")
    do_sample = temperature > 0
    enable_thinking = args.enable_thinking == "true"
    if not do_sample and args.num_samples > 1:
        print("提示：temperature=0 使用贪心解码，num_samples>1 的多个回答可能相同。")

    rows = [json.loads(line) for line in args.data.open(encoding="utf-8") if line.strip()]
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError(f"评测集为空: {args.data}")

    print(f"model={args.model}")
    print(f"enable_thinking={enable_thinking}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # decoder-only 的批量生成必须左填充，才能用同一个偏移切出 completion
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto", trust_remote_code=True,
    ).eval()
    device = model.get_input_embeddings().weight.device


    # 只有 thinking 模式开启且首尾 token 都有效时才评估完整思考块。
    think_start_token_id = None
    think_end_token_id = None
    if enable_thinking:
        try:
            start_id = tokenizer.convert_tokens_to_ids("<think>")
            end_id = tokenizer.convert_tokens_to_ids("</think>")
            valid_start = (
                isinstance(start_id, int) and not isinstance(start_id, bool) and start_id >= 0
                and start_id != tokenizer.unk_token_id
            )
            valid_end = (
                isinstance(end_id, int) and not isinstance(end_id, bool) and end_id >= 0
                and end_id != tokenizer.unk_token_id
            )
            if valid_start and valid_end and start_id != end_id:
                think_start_token_id = start_id
                think_end_token_id = end_id
        except Exception:
            pass
    format_reward = make_format_reward(
        think_end_token_id=think_end_token_id,
        think_start_token_id=think_start_token_id,
    )
    template_kwargs = {"enable_thinking": enable_thinking}

    def add_eos_ids(value, target):
        values = value if isinstance(value, (list, tuple, set)) else [value]
        for token_id in values:
            if isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0:
                target.add(token_id)

    eos_token_ids = set()
    add_eos_ids(tokenizer.eos_token_id, eos_token_ids)
    add_eos_ids(getattr(model.generation_config, "eos_token_id", None), eos_token_ids)
    effective_eos_token_ids = sorted(eos_token_ids)

    generation_config = {
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": args.top_p if do_sample else None,
        "top_k": args.top_k if do_sample else None,
        "num_samples": args.num_samples,
        "num_return_sequences": args.num_samples if do_sample else 1,
        "max_new_tokens": args.max_new_tokens,
        "eos_token_id": effective_eos_token_ids,
        "pad_token_id": tokenizer.pad_token_id,
    }
    print("generation_config=" + json.dumps(generation_config, ensure_ascii=False, sort_keys=True))

    records = []
    done = 0
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": r["question"]}],
                tokenize=False, add_generation_prompt=True, **template_kwargs,
            )
            for r in batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": do_sample,
        }
        # Use the same merged EOS set for generation and truncation accounting.
        # The pad id remains padding-only unless it is independently a real EOS.
        if effective_eos_token_ids:
            gen_kwargs["eos_token_id"] = effective_eos_token_ids
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=args.top_p, top_k=args.top_k)
        # 每题采样 k 条；HF 的输出顺序是「第0题的全部k条, 第1题的全部k条, ...」
        if do_sample and args.num_samples > 1:
            gen_kwargs["num_return_sequences"] = args.num_samples

        with torch.inference_mode():
            output = model.generate(**inputs, **gen_kwargs)

        generated = output[:, prompt_len:]
        if not do_sample and args.num_samples > 1:
            # HF greedy decoding rejects num_return_sequences>1. Repeat the one
            # deterministic completion so num_samples remains an output-count knob.
            generated = generated.repeat_interleave(args.num_samples, dim=0)
        texts_out = tokenizer.batch_decode(generated, skip_special_tokens=True)

        k = args.num_samples

        for i, row in enumerate(batch):
            completions = texts_out[i * k:(i + 1) * k]
            ids_list = generated[i * k:(i + 1) * k]
            convs = [[{"role": "assistant", "content": c}] for c in completions]

            # ---- 按题目类型分派判分 ----
            # MCQ（选择题）：答案是选项字母集合，必须走集合比较。
            #   用 math_verify 会有 100% 假阴性（模型答字母、gold 是下标）。
            # 其余：符号等价判定，gold 包进 \boxed{} 后交给 math_verify。
            is_mcq = str(row.get("question_type", "")).lower() in ("mcq", "mcqa", "gaokao-mcqa")
            gold = str(row["answer"])
            if is_mcq:
                accs = mcq_accuracy_reward(convs, [gold] * len(convs))
                preds = ["".join(sorted(extract_mcq_letters(c))) for c in completions]
            else:
                accs = accuracy_reward(convs, [gold] * len(convs))
                preds = [None] * len(convs)
            raw_id_lists = [ids.tolist() if hasattr(ids, "tolist") else list(ids) for ids in ids_list]
            sample_truncated = []
            completion_tokens = []
            completion_id_lists = []
            for ids in raw_id_lists:
                eos_positions = [j for j, token_id in enumerate(ids) if token_id in eos_token_ids]
                if eos_positions:
                    effective_ids = ids[:eos_positions[0] + 1]
                    completion_id_lists.append(effective_ids)
                    completion_tokens.append(len(effective_ids))
                    sample_truncated.append(False)
                    continue
                effective_len = len(ids)
                # A distinct pad token is padding only, never a successful EOS.
                if tokenizer.pad_token_id not in eos_token_ids:
                    while effective_len and ids[effective_len - 1] == tokenizer.pad_token_id:
                        effective_len -= 1
                completion_id_lists.append(ids[:effective_len])
                completion_tokens.append(effective_len)
                sample_truncated.append(effective_len >= args.max_new_tokens)

            fmts = format_reward(convs, completion_ids=completion_id_lists)

            sample_correct = [a == 1.0 for a in accs]
            strict_correct = [ok and not truncated for ok, truncated in zip(sample_correct, sample_truncated)]
            n_trunc = sum(sample_truncated)
            hit = sum(1 for a in accs if a == 1.0)
            strict_hit = sum(strict_correct)

            records.append({
                "question": row["question"],
                "answer": gold,
                "source": row.get("source", ""),
                "question_type": row.get("question_type", ""),
                "judge": "mcq_set" if is_mcq else "math_verify",
                "n_samples": len(completions),
                "n_correct": hit,
                "avg_at_k": hit / len(completions),
                "raw_avg_at_k": hit / len(completions),
                "strict_avg_at_k": strict_hit / len(completions),
                "raw_pass_at_k": hit > 0,
                "strict_pass_at_k": strict_hit > 0,
                "mean_format": sum(fmts) / len(fmts),
                "n_truncated": n_trunc,
                "sample_correct": sample_correct,
                "sample_truncated": sample_truncated,
                "completion_tokens": completion_tokens,
                "pred_letters": preds if is_mcq else None,   # 诊断：预测到了哪些字母
                "completions": completions,
            })
            done += 1

        if done % 20 < args.batch_size or done == len(rows):
            acc = sum(r["avg_at_k"] for r in records) / len(records)
            print(f"  {done}/{len(rows)}  avg@{k}={acc:.2%}", flush=True)

    # ---------------- 汇总 ----------------
    n = len(records)
    raw_avg_at_k = sum(r["raw_avg_at_k"] for r in records) / n
    strict_avg_at_k = sum(r["strict_avg_at_k"] for r in records) / n
    mean_format = sum(r["mean_format"] for r in records) / n
    trunc_ratio = sum(r["n_truncated"] for r in records) / sum(r["n_samples"] for r in records)
    # 单次贪心时也报「命中率」，等价于 pass@k 的上界口径
    raw_pass_at_k = sum(r["raw_pass_at_k"] for r in records) / n
    strict_pass_at_k = sum(r["strict_pass_at_k"] for r in records) / n

    print("\n" + "=" * 62)
    print(f"model            = {args.model}")
    print(f"data             = {args.data}")
    print(f"n (题目数)        = {n}")
    print(f"raw_avg_at_k      (k={args.num_samples}) = {raw_avg_at_k:.2%}")
    print(f"strict_avg_at_k   (k={args.num_samples}) = {strict_avg_at_k:.2%}")
    print(f"raw_pass_at_k     (k={args.num_samples}) = {raw_pass_at_k:.2%}")
    print(f"strict_pass_at_k  (k={args.num_samples}) = {strict_pass_at_k:.2%}")
    print(f"mean_format      = {mean_format:.4f}")
    print(f"截断比例          = {trunc_ratio:.2%}"
          + ("   ⚠️ 偏高，准确率被系统性低估，请调大 --max_new_tokens"
             if trunc_ratio > 0.1 else ""))

    # 按判分器分组（混装数据集时能看出是哪一类在拉低/拉高）
    by_judge = {}
    for r in records:
        by_judge.setdefault(r.get("judge", "unknown"), []).append(r)
    if len(by_judge) > 1:
        print("-" * 62)
        print("按判分器分组：")
        for judge, rs in sorted(by_judge.items()):
            sub = sum(x["avg_at_k"] for x in rs) / len(rs)
            print(f"  {judge:14} n={len(rs):4}  avg@{args.num_samples}={sub:.2%}")
    elif by_judge:
        judge = next(iter(by_judge))
        print(f"判分器            = {judge}"
              + ("（选择题：字母集合完全匹配）" if judge == "mcq_set"
                 else "（math_verify 符号等价，gold 包 boxed）"))
    print("=" * 62)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"明细已写入 {args.output}")

    # 退出码：截断严重时返回 2，方便 CI/脚本区分「结果不可信」
    return 2 if trunc_ratio > 0.3 else 0


if __name__ == "__main__":
    raise SystemExit(main())
