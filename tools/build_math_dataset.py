# -*- coding: utf-8 -*-
"""把下载的原始数据集转换成本仓库训练脚本需要的三种 schema。

原始数据 → 目标目录
--------------------------------------------------------------------------
DAPO-Math-17k (去重后 17,186 题)   → data/grpo/en/math_dapo_train.jsonl
NuminaMath-1.5 (proof 子集)        → data/grpo/proof/numina_proof.jsonl
MathInstruct-Chinese (中文)        → data/grpo/zh/math_zh_train.jsonl
MATH-500 / AIME2024 / AGIEval      → data/eval/*.jsonl
NuminaMath-1.5 (真实解题过程)        → data/sft/math_sft_demo.jsonl   (SFT 冷启动)

三种 schema（由仓库现有脚本决定，不要随意改）
--------------------------------------------------------------------------
data/grpo/*.jsonl   : {"question": str, "answer": str, "source": str, "question_type": str}
                       ↑ grpo_training.py 读 question / answer
data/sft/*.jsonl    : {"conversations": [{"from":"human","value":...},
                                         {"from":"gpt","value":...}]}
                       ↑ supervised_finetuning.py 读 conversations
data/reward/*.jsonl : {"conversations": [...], "chosen": str, "rejected": str}
                       ↑ dpo_training.py 读 conversations/chosen/rejected

关于 answer 字段：必须能被 math_verify 做符号等价判定（数值、表达式、LaTeX）。
证明题（question_type == "proof"）结论不可机械化验证，单独打标，不进 GRPO。
"""

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import pandas as pd
from loguru import logger

# 允许 `python3 tools/build_math_dataset.py` 直接运行时能 import training.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

RAW = Path("data/_raw")
# ⚠️ GRPO 的三个数据集必须放在【不同子目录】。
# 原因：grpo_training.py 用 load_dataset("json", data_dir=...) 加载，
# 它会把目录下【所有】jsonl 合并，而三者 schema 不同（证明题多一个
# reference_solution 字段）→ 直接抛 CastError。
# 分目录后 --train_file_dir data/grpo/en 即可精确指定。
OUT_GRPO = Path("data/grpo")
OUT_GRPO_EN = Path("data/grpo/en")
OUT_GRPO_ZH = Path("data/grpo/zh")
OUT_GRPO_PROOF = Path("data/grpo/proof")
OUT_SFT = Path("data/sft")
OUT_EVAL = Path("data/eval")

SYSTEM_HINT = "请逐步推理，并把最终答案放在 \\boxed{} 中。"


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _dedupe_hash(text: str) -> str:
    return hashlib.md5(re.sub(r"\s+", " ", str(text)).strip().lower().encode()).hexdigest()


def _write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"[写] {path}  {len(rows)} 行")


def _looks_verifiable(answer: str) -> bool:
    """粗筛：答案是否像可机械化验证的形式（数值/短表达式）。

    证明题的 answer 常是长段落或"证明见解析"，这类会被判为不可验证。
    """
    a = str(answer).strip()
    if not a or len(a) > 80:
        return False
    if re.search(r"(证明|证毕|解析|略|prove|Proof|QED|shown)", a, re.I):
        return False
    return True


# --------------------------------------------------------------------------- #
# 1. DAPO-Math-17k（verl 格式 → 仓库 schema）
# --------------------------------------------------------------------------- #
def build_dapo(n_train: int, n_eval: int, seed: int) -> None:
    src = RAW / "dapo-math-17k.parquet"
    if not src.exists():
        logger.warning(f"跳过 DAPO：{src} 不存在")
        return

    df = pd.read_parquet(src, columns=["prompt", "reward_model"])
    logger.info(f"DAPO 原始 {len(df)} 行，开始去重…")

    rows, seen = [], set()
    for prompt, rm in zip(df["prompt"], df["reward_model"]):
        try:
            text = prompt[0]["content"]
            gt = rm["ground_truth"]
        except Exception:
            continue
        h = _dedupe_hash(text)
        if h in seen:
            continue
        seen.add(h)
        rows.append({"problem": text, "answer": gt})
    logger.info(f"DAPO 去重后 {len(rows)} 道唯一题")

    rng = random.Random(seed)
    rng.shuffle(rows)
    eval_rows = rows[:n_eval]
    train_rows = rows[n_eval:n_eval + n_train]

    def to_schema(r):
        return {
            "question": r["problem"],
            "answer": str(r["answer"]),
            "source": "dapo_math_17k",
            "question_type": "math-word-problem",
        }

    _write_jsonl(OUT_GRPO_EN / "math_dapo_train.jsonl", [to_schema(r) for r in train_rows])
    _write_jsonl(OUT_EVAL / "dapo_holdout.jsonl", [to_schema(r) for r in eval_rows])

# --------------------------------------------------------------------------- #
# 2. NuminaMath-1.5（真实 SFT 解答 + 证明题子集 + 中文子集说明）
# --------------------------------------------------------------------------- #
def build_numina(max_proof: int, max_sft: int, seed: int, zh_ratio_check: bool = True) -> None:
    files = sorted(RAW.glob("numina-train-*.parquet"))
    if not files:
        logger.warning("跳过 NuminaMath：未找到 parquet")
        return

    parts = [
        pd.read_parquet(f, columns=["problem", "answer", "solution", "source",
                                    "question_type", "problem_is_valid", "solution_is_valid"])
        for f in files
    ]
    df = pd.concat(parts, ignore_index=True)
    logger.info(f"NuminaMath-1.5 共 {len(df)} 行")

    if zh_ratio_check:
        cn = df[df["source"].isin(["cn_k12", "cn_contest"])]
        if len(cn):
            sample = cn.sample(min(300, len(cn)), random_state=seed)
            cjk = sample["problem"].map(
                lambda s: len(re.findall(r"[\u4e00-\u9fff]", str(s))) / max(1, len(str(s)))
            )
            ratio = float((cjk > 0.10).mean())
            logger.warning(
                f"⚠️ NuminaMath 的 cn_k12/cn_contest 子集（{len(cn)} 行）中，"
                f"真正含中文的比例仅 {ratio:.1%} —— 该数据集把中文题翻译成了英文，"
                f"不能作为中文高中题训练源。"
            )

    # SFT 必须使用真实解题过程。旧实现从只有最终答案的 DAPO 数据凭空拼出
    # “逐步推理后得到结果”占位文本，实际上没有任何推理监督，会教坏模型。
    sft = df[(df["question_type"] != "proof")
             & (df["problem_is_valid"] == "Yes")
             & (df["solution_is_valid"] == "Yes")].copy()
    sft = sft[sft["answer"].map(_looks_verifiable)]
    sft = sft[sft["solution"].map(lambda value: len(str(value).strip()) >= 80)]
    sft = sft.sample(frac=1, random_state=seed).head(max_sft)
    sft_rows = []
    for _, row in sft.iterrows():
        answer = str(row["answer"]).strip()
        solution = str(row["solution"]).strip()
        if "\\boxed" not in solution and "\\fbox" not in solution:
            solution += f"\n\n\\boxed{{{answer}}}"
        sft_rows.append({
            "conversations": [
                {"from": "human", "value": str(row["problem"]).strip()},
                {"from": "gpt", "value": solution},
            ]
        })
    logger.info(f"NuminaMath 真实解题 SFT 样本 {len(sft_rows)} 条")
    _write_jsonl(OUT_SFT / "math_sft_demo.jsonl", sft_rows)

    # 证明题：结论不可机械化验证，单独打标，默认不进 GRPO
    proof = df[(df["question_type"] == "proof")
               & (df["problem_is_valid"] == "Yes")
               & (df["solution_is_valid"] == "Yes")]
    # 必须过滤空答案：否则 tools/validate_jsonl.py 会判 answer 不能为空，
    # 而且下游评测/DPO 拿到空答案会直接失效。
    before = len(proof)
    proof = proof[proof["answer"].astype(str).str.strip().str.len() > 0]
    logger.info(
        f"证明题（有效）{before} 道 → 过滤空答案后 {len(proof)} 道，取前 {max_proof} 道"
    )
    proof = proof.head(max_proof)

    rows = [{
        "question": str(r["problem"]),
        "answer": str(r["answer"]),
        "source": f"numina_{r['source']}",
        "question_type": "proof",
        "verifiable": False,          # ← 关键标记：GRPO 跳过，转 DPO/评测
        "reference_solution": str(r["solution"])[:8000],
    } for _, r in proof.iterrows()]
    _write_jsonl(OUT_GRPO_PROOF / "numina_proof.jsonl", rows)


# --------------------------------------------------------------------------- #
# 3. MathInstruct-Chinese（中文训练集，英译中）
# --------------------------------------------------------------------------- #
def _extract_final_from_solution(text: str) -> str:
    """从解答文本里抽最终答案。

    覆盖本仓库用到的几种写法：\\boxed{} > 「答案是X」 > 「答案：X」 > 最后一句。
    """
    if not text:
        return ""
    m = re.search(r"\\boxed\{([^{}]{1,80})\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"答案\s*[是为:：]\s*([^\s。，,；;]{1,40})", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"[Tt]he\s+answer\s+is\s+([^\s.,;]{1,40})", text)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    return lines[-1][:80] if lines else ""


def build_math_zh(n_train: int, seed: int) -> None:
    src = RAW / "mathinstruct-zh.parquet"
    if not src.exists():
        logger.warning(f"跳过 MathInstruct-Chinese：{src} 不存在")
        return

    df = pd.read_parquet(src)
    logger.info(f"MathInstruct-Chinese 共 {len(df)} 行 | 列: {list(df.columns)}")

    qcol = next((c for c in df.columns if c.lower() in ("instruction", "question", "problem")), None)
    acol = next((c for c in df.columns if c.lower() in ("output", "answer", "response")), None)
    mcol = next((c for c in df.columns if c.lower() in ("messages", "conversations")), None)

    if not (mcol or (qcol and acol)):
        logger.warning(f"无法识别列名，跳过。列={list(df.columns)}")
        return

    rows, n_unverifiable = 0, 0
    out = []
    for _, r in df.iterrows():
        if mcol:
            msgs = list(r[mcol])
            user = next((m.get("content", "") for m in msgs
                         if str(m.get("role", m.get("from", ""))).lower() in ("user", "human")), "")
            asst = next((m.get("content", "") for m in msgs
                         if str(m.get("role", m.get("from", ""))).lower() in ("assistant", "gpt")), "")
        else:
            user, asst = str(r[qcol]), str(r[acol])
        if not user or not asst:
            continue
        final = _extract_final_from_solution(asst)
        if not _looks_verifiable(final):
            n_unverifiable += 1
            continue
        out.append({
            "question": user.strip(),
            "answer": final,
            "source": "mathinstruct_zh",
            "question_type": "math-word-problem",
            "reference_solution": asst[:4000],
        })
        rows += 1
        if len(out) >= n_train * 2:      # 多收一些，后面 shuffle 再截断
            break

    logger.info(f"MathInstruct-Chinese 可验证样本 {len(out)} 条（跳过不可验证 {n_unverifiable} 条）")
    random.Random(seed).shuffle(out)
    _write_jsonl(OUT_GRPO_ZH / "math_zh_train.jsonl", out[:n_train])


# --------------------------------------------------------------------------- #
# 4. 评测集
# --------------------------------------------------------------------------- #
def build_evals() -> None:
    # MATH-500
    p = RAW / "math500-test.jsonl"
    if p.exists():
        rows = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                rows.append({
                    "question": d.get("problem", ""),
                    "answer": d.get("answer", ""),
                    "source": "math500",
                    "question_type": "math-word-problem",
                })
        _write_jsonl(OUT_EVAL / "math500.jsonl", rows)

    # AIME 2024
    p = RAW / "aime2024.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        rows = [{
            "question": str(r.get("Problem", "")),
            "answer": str(r.get("Answer", "")),
            "source": "aime2024",
            "question_type": "math-word-problem",
        } for _, r in df.iterrows()]
        _write_jsonl(OUT_EVAL / "aime2024.jsonl", rows)

    # AGIEval 高考数学（真实中文高考题）
    p = RAW / "agieval-gaokao.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        logger.info(f"AGIEval 高考数学 {len(df)} 行 | 列: {list(df.columns)}")
        cols = {c.lower(): c for c in df.columns}
        qcol = cols.get("query") or cols.get("question") or cols.get("problem")
        acol = cols.get("answer") or cols.get("gold")

        # ⚠️ AGIEval 的 gold 是【选项下标集合】，且是 **0-based**
        #    （证据：出现过 [0]，若是 1-based 不可能有 0），并存在多选题。
        #    题面里的选项标注是 (A)(B)(C)(D)。
        #    而模型最自然的作答是输出选项字母 —— 实测用 math_verify 判分
        #    会有 100% 假阴性。所以这里统一转成字母集合，并打 question_type="mcq"
        #    让评测脚本走集合比较。
        from training.math_rewards import MCQ_LETTERS

        def _indices_to_letters(raw) -> tuple:
            nums = sorted({int(x) for x in re.findall(r"\d+", str(raw))})
            nums = [n for n in nums if 0 <= n < len(MCQ_LETTERS)]
            return "".join(MCQ_LETTERS[n] for n in nums), nums

        rows, n_bad, n_multi = [], 0, 0
        for _, r in df.iterrows():
            letters, nums = _indices_to_letters(r[acol])
            if not letters:
                n_bad += 1
                continue
            if len(nums) > 1:
                n_multi += 1
            rows.append({
                "question": str(r[qcol]),
                "answer": letters,               # ← 字母集合，如 "ABD"
                "answer_indices": nums,          # 保留原始下标，便于回溯
                "source": "agieval_gaokao",
                "question_type": "mcq",          # ← 评测据此走 MCQ 判分
            })
        logger.info(
            f"AGIEval 转换：{len(df)} 行 → {len(rows)} 有效"
            f"（跳过 {n_bad} 行无有效下标，其中多选 {n_multi} 题）"
        )
        _write_jsonl(OUT_EVAL / "agieval_gaokao.jsonl", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dapo_train", type=int, default=12000)
    ap.add_argument("--dapo_eval", type=int, default=500)
    ap.add_argument("--numina_proof", type=int, default=5000)
    ap.add_argument("--numina_sft", type=int, default=2000)
    ap.add_argument("--math_zh", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    build_dapo(args.dapo_train, args.dapo_eval, args.seed)
    build_numina(args.numina_proof, args.numina_sft, args.seed)
    build_math_zh(args.math_zh, args.seed)
    build_evals()
    logger.info("数据构建完成 ✅")


if __name__ == "__main__":
    main()
