#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast math evaluation with vLLM and the project's existing verifiers.

The output schema intentionally mirrors ``tools/evaluate_math.py``.  Records
are appended after every request chunk, so an interrupted run can resume
without regenerating completed questions.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

from training.math_rewards import (  # noqa: E402
    accuracy_reward,
    extract_mcq_letters,
    make_format_reward,
    mcq_accuracy_reward,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Math evaluation with vLLM.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument(
        "--enable_thinking", type=str.lower, choices=("true", "false"), default="true"
    )
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.75)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_num_seqs", type=int, default=8)
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=32,
        help="每次提交给 vLLM 的题目数；只影响增量落盘频率，不是显存并发数。",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--enforce_eager", action="store_true")
    parser.add_argument("--disable_prefix_caching", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出；默认读取已有记录并断点续评。",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < 1:
        raise ValueError("--num_samples 必须大于等于 1")
    if args.max_new_tokens < 1:
        raise ValueError("--max_new_tokens 必须大于等于 1")
    if args.max_model_len <= args.max_new_tokens:
        raise ValueError("--max_model_len 必须大于 --max_new_tokens，以容纳 prompt")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu_memory_utilization 必须在 (0, 1] 范围内")
    if args.max_num_seqs < 1 or args.request_batch_size < 1:
        raise ValueError("--max_num_seqs 和 --request_batch_size 必须大于等于 1")
    if args.temperature < 0:
        raise ValueError("--temperature 必须大于等于 0")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top_p 必须在 (0, 1] 范围内")
    if args.top_k == 0 or args.top_k < -1:
        raise ValueError("vLLM 的 --top_k 必须为 -1（关闭）或正整数")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_existing(path: Path, overwrite: bool) -> dict[int, dict]:
    if not path.exists() or overwrite:
        return {}
    records = read_jsonl(path)
    completed: dict[int, dict] = {}
    for record in records:
        if "sample_index" not in record:
            raise ValueError(
                f"已有输出不含 sample_index，无法安全断点续评：{path}。"
                "请更换输出路径或显式传 --overwrite。"
            )
        index = int(record["sample_index"])
        if index in completed:
            raise ValueError(f"已有输出包含重复 sample_index={index}: {path}")
        completed[index] = record
    return completed


def valid_token_id(tokenizer, token: str):
    token_id = tokenizer.convert_tokens_to_ids(token)
    if (
        isinstance(token_id, int)
        and not isinstance(token_id, bool)
        and token_id >= 0
        and token_id != tokenizer.unk_token_id
    ):
        return token_id
    return None


def score_record(
    sample_index: int,
    row: dict,
    completions: list[str],
    token_ids: list[list[int]],
    finish_reasons: list[str | None],
    format_reward,
) -> dict:
    conversations = [[{"role": "assistant", "content": text}] for text in completions]
    question_type = str(row.get("question_type", "")).lower()
    is_mcq = question_type in ("mcq", "mcqa", "gaokao-mcqa")
    gold = str(row["answer"])
    if is_mcq:
        accuracies = mcq_accuracy_reward(conversations, [gold] * len(completions))
        predictions = ["".join(sorted(extract_mcq_letters(text))) for text in completions]
    else:
        accuracies = accuracy_reward(conversations, [gold] * len(completions))
        predictions = [None] * len(completions)

    formats = format_reward(conversations, completion_ids=token_ids)
    truncated = [reason == "length" for reason in finish_reasons]
    correct = [score == 1.0 for score in accuracies]
    strict_correct = [ok and not cut for ok, cut in zip(correct, truncated)]
    hit = sum(correct)
    strict_hit = sum(strict_correct)
    count = len(completions)

    return {
        "sample_index": sample_index,
        "question": row["question"],
        "answer": gold,
        "source": row.get("source", ""),
        "question_type": row.get("question_type", ""),
        "judge": "mcq_set" if is_mcq else "math_verify",
        "n_samples": count,
        "n_correct": hit,
        "avg_at_k": hit / count,
        "raw_avg_at_k": hit / count,
        "strict_avg_at_k": strict_hit / count,
        "raw_pass_at_k": hit > 0,
        "strict_pass_at_k": strict_hit > 0,
        "mean_format": sum(formats) / count,
        "n_truncated": sum(truncated),
        "sample_correct": correct,
        "strict_correct": strict_correct,
        "sample_truncated": truncated,
        "completion_tokens": [len(ids) for ids in token_ids],
        "finish_reasons": finish_reasons,
        "pred_letters": predictions if is_mcq else None,
        "completions": completions,
    }


def print_summary(records: list[dict], args: argparse.Namespace, elapsed: float, new_tokens: int) -> int:
    count = len(records)
    total_samples = sum(record["n_samples"] for record in records)
    total_tokens = sum(sum(record.get("completion_tokens", [])) for record in records)
    truncated = sum(record["n_truncated"] for record in records)
    raw_avg = sum(record["raw_avg_at_k"] for record in records) / count
    strict_avg = sum(record["strict_avg_at_k"] for record in records) / count
    raw_pass = sum(record["raw_pass_at_k"] for record in records) / count
    strict_pass = sum(record["strict_pass_at_k"] for record in records) / count
    mean_format = sum(record["mean_format"] for record in records) / count
    truncation_ratio = truncated / total_samples

    print("\n" + "=" * 66)
    print(f"backend          = vLLM")
    print(f"model            = {args.model}")
    print(f"data             = {args.data}")
    print(f"n (题目数)        = {count}")
    print(f"raw_avg_at_k      (k={args.num_samples}) = {raw_avg:.2%}")
    print(f"strict_avg_at_k   (k={args.num_samples}) = {strict_avg:.2%}")
    print(f"raw_pass_at_k     (k={args.num_samples}) = {raw_pass:.2%}")
    print(f"strict_pass_at_k  (k={args.num_samples}) = {strict_pass:.2%}")
    print(f"mean_format      = {mean_format:.4f}")
    print(f"截断比例          = {truncation_ratio:.2%}")
    print(f"completion tokens = {total_tokens}")
    if elapsed > 0:
        print(f"本次生成耗时      = {elapsed:.1f}s")
        print(f"本次生成吞吐      = {new_tokens / elapsed:.1f} tokens/s")
    print(f"输出             = {args.output}")
    print("=" * 66)
    return 2 if truncation_ratio > 0.3 else 0


def main() -> int:
    args = parse_args()
    validate_args(args)
    rows = read_jsonl(args.data)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"评测集为空: {args.data}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_existing(args.output, args.overwrite)
    if args.overwrite:
        args.output.unlink(missing_ok=True)
    pending = [(index, row) for index, row in enumerate(rows) if index not in completed]
    print(
        f"总题数={len(rows)}，已完成={len(completed)}，待生成={len(pending)}，"
        f"k={args.num_samples}"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    think_start_id = valid_token_id(tokenizer, "<think>") if args.enable_thinking == "true" else None
    think_end_id = valid_token_id(tokenizer, "</think>") if args.enable_thinking == "true" else None
    format_reward = make_format_reward(
        think_start_token_id=think_start_id,
        think_end_token_id=think_end_id,
    )

    if not pending:
        records = [completed[index] for index in range(len(rows))]
        return print_summary(records, args, elapsed=0.0, new_tokens=0)

    print(
        "vLLM config="
        + json.dumps(
            {
                "dtype": args.dtype,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len": args.max_model_len,
                "max_num_seqs": args.max_num_seqs,
                "enable_prefix_caching": not args.disable_prefix_caching,
                "enforce_eager": args.enforce_eager,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    load_start = time.perf_counter()
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=not args.disable_prefix_caching,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    print(f"vLLM 模型加载耗时={time.perf_counter() - load_start:.1f}s")

    do_sample = args.temperature > 0
    if not do_sample and args.num_samples > 1:
        print("提示：temperature=0 时多条输出可能相同。")
    sampling = SamplingParams(
        n=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p if do_sample else 1.0,
        top_k=args.top_k if do_sample else -1,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    generated_tokens = 0
    generation_seconds = 0.0
    mode = "a" if completed and not args.overwrite else "w"
    with args.output.open(mode, encoding="utf-8") as output_file:
        for start in range(0, len(pending), args.request_batch_size):
            chunk = pending[start : start + args.request_batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["question"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=args.enable_thinking == "true",
                )
                for _, row in chunk
            ]
            started = time.perf_counter()
            generated = llm.generate(prompts, sampling, use_tqdm=True)
            generation_seconds += time.perf_counter() - started
            if len(generated) != len(chunk):
                raise RuntimeError(
                    f"vLLM 返回 {len(generated)} 个请求，预期 {len(chunk)} 个"
                )

            for (sample_index, row), request_output in zip(chunk, generated):
                outputs = request_output.outputs
                completions = [item.text for item in outputs]
                ids = [list(item.token_ids) for item in outputs]
                finish_reasons = [item.finish_reason for item in outputs]
                generated_tokens += sum(len(value) for value in ids)
                record = score_record(
                    sample_index,
                    row,
                    completions,
                    ids,
                    finish_reasons,
                    format_reward,
                )
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                completed[sample_index] = record
            output_file.flush()
            print(
                f"已完成 {len(completed)}/{len(rows)}，"
                f"本次平均吞吐={generated_tokens / generation_seconds:.1f} tokens/s",
                flush=True,
            )

    records = [completed[index] for index in range(len(rows))]
    return print_summary(records, args, generation_seconds, generated_tokens)


if __name__ == "__main__":
    raise SystemExit(main())
