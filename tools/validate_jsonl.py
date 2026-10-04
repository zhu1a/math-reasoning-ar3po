# -*- coding: utf-8 -*-
"""Validate the GRPO/eval, SFT, and DPO JSONL schemas used by this project."""

import argparse
import json
from pathlib import Path


CONVERSATION_ROLES = {"system", "human", "gpt", "user", "assistant", "function_call", "observation"}


def _validate_conversations(conversations):
    if not isinstance(conversations, list) or not conversations:
        return "'conversations' 必须是非空列表"
    for index, message in enumerate(conversations):
        if not isinstance(message, dict) or "from" not in message or "value" not in message:
            return f"conversations[{index}] 缺少 'from' 或 'value'"
        if message["from"] not in CONVERSATION_ROLES:
            return f"conversations[{index}].from={message['from']!r} 不受支持"
        if not isinstance(message["value"], str):
            return f"conversations[{index}].value 必须是字符串"
    return None


def validate_record(data):
    """Return ``(schema, error)`` for one decoded JSON object."""
    if not isinstance(data, dict):
        return "unknown", "顶层必须是 JSON object"

    if "conversations" in data:
        error = _validate_conversations(data["conversations"])
        if error:
            return "conversation", error
        has_chosen = "chosen" in data
        has_rejected = "rejected" in data
        if has_chosen or has_rejected:
            if not (has_chosen and has_rejected):
                return "dpo", "DPO 数据必须同时包含 chosen 和 rejected"
            if not isinstance(data["chosen"], str) or not isinstance(data["rejected"], str):
                return "dpo", "chosen/rejected 必须是字符串"
            if not data["chosen"].strip() or not data["rejected"].strip():
                return "dpo", "chosen/rejected 不能为空"
            if data["chosen"] == data["rejected"]:
                return "dpo", "chosen 和 rejected 不能相同"
            return "dpo", None
        return "sft", None

    if "question" in data or "answer" in data:
        if not isinstance(data.get("question"), str) or not data["question"].strip():
            return "grpo/eval", "question 必须是非空字符串"
        if "answer" not in data or not str(data["answer"]).strip():
            return "grpo/eval", "answer 不能为空"
        return "grpo/eval", None

    return "unknown", "无法识别 schema；需要 conversations 或 question+answer"


def validate_jsonl(file_path, max_errors=20):
    path = Path(file_path)
    total = valid = 0
    schemas = {}
    errors = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            total += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append((line_number, f"JSON 无效: {exc.msg}"))
                continue
            schema, error = validate_record(data)
            schemas[schema] = schemas.get(schema, 0) + 1
            if error:
                errors.append((line_number, error))
            else:
                valid += 1

    for line_number, error in errors[:max_errors]:
        print(f"第 {line_number} 行: {error}")
    if len(errors) > max_errors:
        print(f"... 其余 {len(errors) - max_errors} 条错误已省略")
    print(f"schema: {schemas}")
    print(f"总行数: {total}，有效: {valid}，无效: {total - valid}")
    return total > 0 and valid == total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate project JSONL schemas.")
    parser.add_argument("--file_path", required=True, help="Path to JSONL file")
    parser.add_argument("--max_errors", type=int, default=20)
    args = parser.parse_args()
    raise SystemExit(0 if validate_jsonl(args.file_path, args.max_errors) else 1)
