#!/usr/bin/env python3
"""Build a non-destructive, structurally validated Qwen3 thinking SFT dataset.

The source dataset stores a worked solution as a plain assistant message.  This
tool preserves that solution as reasoning, adds explicit Qwen3 thinking
boundaries, and repeats only the last balanced ``\\boxed{...}`` expression as
the visible final answer.  Rows that cannot satisfy the structural contract are
written to a rejection log instead of being silently repaired.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


BAD_FINALS = {"nan", "none", "null", "undefined", "n/a", "unknown"}
SUSPICIOUS_PHRASES = (
    "assuming we made a mistake",
    "translate the text above into english",
    "although the final answer is not in the exact format",
)
RESERVED_CHAT_TOKENS = ("<|im_start|>", "<|im_end|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a separate Qwen3 thinking-formatted SFT JSONL file."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--min-reasoning-chars", type=int, default=80)
    return parser.parse_args()


def _is_escaped(text: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and text[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def balanced_boxed_expressions(text: str) -> list[str]:
    """Return complete ``\\boxed{...}`` expressions, including nested braces."""
    results: list[str] = []
    for match in re.finditer(r"\\boxed\s*\{", text):
        depth = 0
        for index in range(match.end() - 1, len(text)):
            char = text[index]
            if char == "{" and not _is_escaped(text, index):
                depth += 1
            elif char == "}" and not _is_escaped(text, index):
                depth -= 1
                if depth == 0:
                    results.append(text[match.start() : index + 1])
                    break
    return results


def boxed_payload(boxed: str) -> str:
    start = boxed.find("{")
    return boxed[start + 1 : -1].strip() if start >= 0 else ""


def reject_reasons(row: object, min_reasoning_chars: int) -> tuple[list[str], str, str]:
    reasons: list[str] = []
    if not isinstance(row, dict):
        return ["row_not_object"], "", ""

    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        return ["not_single_turn"], "", ""

    user, assistant = conversations
    if not isinstance(user, dict) or user.get("from") not in {"human", "user"}:
        reasons.append("invalid_user_message")
    if not isinstance(assistant, dict) or assistant.get("from") not in {"gpt", "assistant"}:
        reasons.append("invalid_assistant_message")

    question = user.get("value", "") if isinstance(user, dict) else ""
    answer = assistant.get("value", "") if isinstance(assistant, dict) else ""
    if not isinstance(question, str) or not question.strip():
        reasons.append("empty_question")
        question = ""
    if not isinstance(answer, str) or not answer.strip():
        reasons.append("empty_answer")
        answer = ""
    if len(answer.strip()) < min_reasoning_chars:
        reasons.append("reasoning_too_short")
    if "<think>" in answer or "</think>" in answer:
        reasons.append("preexisting_thinking_tags")
    if any(token in question or token in answer for token in RESERVED_CHAT_TOKENS):
        reasons.append("reserved_chat_token")
    lowered = answer.lower()
    if any(phrase in lowered for phrase in SUSPICIOUS_PHRASES):
        reasons.append("suspicious_content")

    boxes = balanced_boxed_expressions(answer)
    if not boxes:
        reasons.append("missing_balanced_boxed_answer")
        final_box = ""
    else:
        final_box = boxes[-1]
        payload = re.sub(r"\s+", "", boxed_payload(final_box)).lower()
        if not payload:
            reasons.append("empty_boxed_answer")
        if payload in BAD_FINALS:
            reasons.append("invalid_boxed_answer")

    return sorted(set(reasons)), question, final_box


def render_target(original_answer: str, final_box: str) -> str:
    return (
        "<think>\n"
        + original_answer.strip()
        + "\n</think>\n\n"
        + "Final answer: "
        + final_box
    )


def encoded_length(tokenizer, question: str, target: str) -> tuple[int, list[int]]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    source_ids = tokenizer.encode(prompt, add_special_tokens=True)
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    if source_ids and source_ids[0] == tokenizer.eos_token_id:
        source_ids = source_ids[1:]
    if target_ids and target_ids[-1] == tokenizer.eos_token_id:
        target_ids = target_ids[:-1]
    return len(source_ids) + len(target_ids) + 1, target_ids


def refuse_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing output(s): " + ", ".join(existing)
        )


def main() -> int:
    args = parse_args()
    source = args.input.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("--output must differ from --input")
    if args.max_length < 128:
        raise ValueError("--max-length must be at least 128")

    # The SFT loader recursively consumes every ``*.jsonl`` in train_file_dir.
    # Keep the audit log in line-delimited JSON but use a non-training suffix.
    rejected_path = output.with_suffix(".rejected.ndjson")
    report_path = output.with_suffix(".report.json")
    refuse_existing([output, rejected_path, report_path])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    think_start_id = tokenizer.convert_tokens_to_ids("<think>")
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")
    invalid_ids = {None, tokenizer.unk_token_id}
    if think_start_id in invalid_ids or think_end_id in invalid_ids or think_start_id == think_end_id:
        raise ValueError("Tokenizer does not expose distinct <think> and </think> token ids")

    accepted: list[dict] = []
    rejected: list[dict] = []
    reason_counts: Counter[str] = Counter()
    input_sha256 = hashlib.sha256()

    with source.open("rb") as raw_file:
        raw_lines = raw_file.readlines()
    for raw_line in raw_lines:
        input_sha256.update(raw_line)

    for source_index, raw_line in enumerate(raw_lines, 1):
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            reasons = ["invalid_json"]
            rejected.append({"source_index": source_index, "reasons": reasons, "detail": str(exc)})
            reason_counts.update(reasons)
            continue

        reasons, question, final_box = reject_reasons(row, args.min_reasoning_chars)
        original_answer = ""
        if isinstance(row, dict) and isinstance(row.get("conversations"), list):
            conversations = row["conversations"]
            if len(conversations) == 2 and isinstance(conversations[1], dict):
                original_answer = conversations[1].get("value", "")

        target = render_target(original_answer, final_box) if not reasons else ""
        if not reasons:
            total_length, target_ids = encoded_length(tokenizer, question, target)
            if total_length > args.max_length:
                reasons.append("over_max_length")
            if target_ids.count(think_start_id) != 1 or target_ids.count(think_end_id) != 1:
                reasons.append("invalid_thinking_token_count")
            elif target_ids.index(think_start_id) >= target_ids.index(think_end_id):
                reasons.append("invalid_thinking_token_order")

        if reasons:
            reasons = sorted(set(reasons))
            reason_counts.update(reasons)
            rejected.append(
                {
                    "source_index": source_index,
                    "reasons": reasons,
                    "question_preview": question[:160],
                }
            )
            continue

        accepted.append(
            {
                "conversations": [
                    {"from": "human", "value": question},
                    {"from": "gpt", "value": target},
                ]
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as file:
        for row in accepted:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    with rejected_path.open("x", encoding="utf-8") as file:
        for row in rejected:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = {
        "input": str(source),
        "input_sha256": input_sha256.hexdigest(),
        "output": str(output),
        "tokenizer": args.tokenizer,
        "max_length": args.max_length,
        "total_rows": len(raw_lines),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "rejection_reasons": dict(sorted(reason_counts.items())),
        "think_start_token_id": think_start_id,
        "think_end_token_id": think_end_id,
    }
    with report_path.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
