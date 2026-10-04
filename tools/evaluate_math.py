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
  · k>1 时采样是随机的（temperature 默认 1.0，Qwen 官方推荐值）；
    k=1 时默认贪心，可用 --temperature 覆盖
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
    p.add_argument("--temperature", type=float, default=None,
                   help="默认：k>1 时 1.0，k=1 时 0.0（贪心）")
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--enable_thinking", default=None,
                   help="传给 chat template；Qwen3 传 false 可关闭思考模式")
    p.add_argument("--output", type=Path, help="把每题明细写成 jsonl")
    return p.parse_args()


def main():
    args = parse_args()
    temperature = args.temperature
    if temperature is None:
        temperature = 1.0 if args.num_samples > 1 else 0.0
    do_sample = temperature > 0

    rows = [json.loads(line) for line in args.data.open(encoding="utf-8") if line.strip()]
    if args.limit > 0:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError(f"评测集为空: {args.data}")

    print(f"model={args.model}")
    print(f"data={args.data}  n={len(rows)}  avg@{args.num_samples}  "
          f"batch={args.batch_size}  temperature={temperature}  "
          f"max_new_tokens={args.max_new_tokens}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # decoder-only 的批量生成必须左填充，才能用同一个偏移切出 completion
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto", trust_remote_code=True,
    ).eval()
    device = model.get_input_embeddings().weight.device

    format_reward = make_format_reward()

    # 取 Qwen3 的 thinking 结束 token（用于判断思考块是否完整）
    think_end_token_id = None
    try:
        tid = tokenizer.convert_tokens_to_ids(" response")
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            think_end_token_id = tid
    except Exception:
        pass

    template_kwargs = {}
    if args.enable_thinking is not None:
        template_kwargs["enable_thinking"] = str(args.enable_thinking).lower() != "false"

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
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=args.top_p)
        # 每题采样 k 条；HF 的输出顺序是「第0题的全部k条, 第1题的全部k条, ...」
        if args.num_samples > 1:
            gen_kwargs["num_return_sequences"] = args.num_samples

        with torch.inference_mode():
            output = model.generate(**inputs, **gen_kwargs)

        generated = output[:, prompt_len:]
        texts_out = tokenizer.batch_decode(generated, skip_special_tokens=True)

        eos_and_pad = {tokenizer.eos_token_id, tokenizer.pad_token_id}
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

            fmts = format_reward(convs, completion_ids=None)

            # 截断：最后 token 既不是 eos 也不是 pad
            n_trunc = sum(
                1 for ids in ids_list
                if len(ids) > 0 and int(ids[-1]) not in eos_and_pad
            )
            hit = sum(1 for a in accs if a == 1.0)

            records.append({
                "question": row["question"],
                "answer": gold,
                "source": row.get("source", ""),
                "question_type": row.get("question_type", ""),
                "judge": "mcq_set" if is_mcq else "math_verify",
                "n_samples": len(completions),
                "n_correct": hit,
                "avg_at_k": hit / len(completions),
                "mean_format": sum(fmts) / len(fmts),
                "n_truncated": n_trunc,
                "pred_letters": preds if is_mcq else None,   # 诊断：预测到了哪些字母
                "completions": completions,
            })
            done += 1

        if done % 20 < args.batch_size or done == len(rows):
            acc = sum(r["avg_at_k"] for r in records) / len(records)
            print(f"  {done}/{len(rows)}  avg@{k}={acc:.2%}", flush=True)

    # ---------------- 汇总 ----------------
    n = len(records)
    accuracy = sum(r["avg_at_k"] for r in records) / n
    mean_format = sum(r["mean_format"] for r in records) / n
    trunc_ratio = sum(r["n_truncated"] for r in records) / sum(r["n_samples"] for r in records)
    # 单次贪心时也报「命中率」，等价于 pass@k 的上界口径
    any_correct = sum(1 for r in records if r["n_correct"] > 0) / n

    print("\n" + "=" * 62)
    print(f"model            = {args.model}")
    print(f"data             = {args.data}")
    print(f"n (题目数)        = {n}")
    print(f"avg@{args.num_samples} (主指标)   = {accuracy:.2%}")
    if args.num_samples > 1:
        print(f"pass@{args.num_samples} (至少对1条) = {any_correct:.2%}")
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
