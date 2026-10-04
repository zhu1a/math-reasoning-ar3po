# -*- coding: utf-8 -*-
"""数学推理任务的奖励函数（GRPO / AR3PO 用）。

===========================================================================
一、设计原则
===========================================================================
总分 = w_acc · accuracy  +  w_fmt · format  +  w_len · length

· accuracy  —— **整体给分（outcome）**，用 math_verify 做符号等价判定 → 0 / 1
               不可 hack，是整个奖励的地基。**必须放在 reward_funcs 的第 0 位**，
               因为 AR3POTrainer 用 ar3po_correctness_reward_idx=0 单独靠它判定「全错」。

· format    —— **分层给分**（不是二值！）。这是缓解 GRPO「组内奖励全同 → advantage 全 0」
               的第一道防线，比任何算法改动都更根本。
               注意：分档是工程判断，文献里 R1-Zero/DAPO 用的是二值 format reward，
               建议做一次 A/B 对比（看 TRL 自动记录的 frac_reward_zero_std 指标）。

· length    —— DAPO (arXiv:2503.14476) Eq.13 的 **软长度惩罚**：
               只在接近截断阈值时才惩罚，不惩罚正常长度。
               原文：不恰当的长度奖励会引入 reward noise 并显著干扰训练。

===========================================================================
二、两个必须注意的实现细节（都是实测踩出来的）
===========================================================================
【坑 1】gold 也必须包进 \\boxed{} 再 parse
    math_verify 的 LatexExtractionConfig 默认需要 anchor（\\boxed{} 或 "answer is"）。
    直接 parse("\\sqrt{51}") 会抽出空值或只抓到一个数字，
    导致**模型答对了却被判 0 分**。
    实测（MATH-500 + DAPO 共 800 条）自洽率：
        直接 parse gold      → 86.5%（13.5% 的奖励噪声！）
        把 gold 包进 \\boxed{} → **100.0%**
    → 本模块统一对 gold 包 boxed。

【坑 2】TRL 用 skip_special_tokens=True 解码，thinking 标记会被剥掉
    Qwen3 的 ` thinking` / `response` 是 special token，解码后**不会出现在文本里**。
    所以 format_reward **不能靠正则检查  thinking 标签**。
    正确做法：用 completion_ids 里是否含 think-end token id 来判断
    （通过 make_format_reward(think_end_token_id=...) 注入）。

【坑 3】选择题（MCQ）不能用 math_verify 判分
    AGIEval 高考这类数据集的答案是「选项下标」（0-based，如 "[0 1 3]"），
    而题面里选项标注是 (A)(B)(C)(D)。模型最自然的作答是输出选项字母
    （`\\boxed{C}`），但 math_verify 不懂「字母 ↔ 下标」的映射。
    实测：模型输出字母被判 0 的假阴性率 **100/100**。
    → MCQ 必须走独立的集合比较判分，见本模块的 extract_mcq_letters /
      mcq_accuracy_reward。
"""

import re
from typing import Callable, List, Optional

from loguru import logger
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

# --------------------------------------------------------------------------- #
# 文本抽取
# --------------------------------------------------------------------------- #
_ANSWER_MARKERS = [
    re.compile(r"####\s*(.+?)(?:\n|$)"),                        # GSM8K 风格
    re.compile(r"(?:答案|所以|因此|故|答)\s*[:：]?\s*(.+?)(?:\n|$)"),
    re.compile(r"(?:the\s+answer\s+is|answer\s*[:：])\s*(.+?)(?:\n|$)", re.I),
]

_FILLER = re.compile(r"^\s*(好的|当然|没问题|让我来|首先|Sure|Certainly|Of course|Okay)", re.U)

#: 解析配置（pred 侧：优先 boxed；gold 侧：先用 boxed 包裹再走同一路径）
_LATEX_PRED = [
    LatexExtractionConfig(boxed_match_priority=0, try_extract_without_anchor=False),
    ExprExtractionConfig(),
]


def strip_thinking(text: str) -> str:
    """去掉思考块，只留最终回答。

    兼容三种写法：
      · Qwen3 原生        思考...<end_of_thinking>答案   （闭合标记可能已被 skip_special_tokens 剥掉）
      · DeepSeek-R1       <think>思考</think>答案
      · 本仓库历史数据     <think>思考</think><answer>答案</answer>
    """
    if not text:
        return ""
    out = re.sub(r" thinking.*? response", " ", text, flags=re.DOTALL)
    out = re.sub(r"<think>.*?</think>", " ", out, flags=re.DOTALL)
    # 闭合标记被剥掉的情况：只剩开头标记 + 内容，取 `\n\n` 之后的部分
    if " thinking" in out:
        parts = out.split(" thinking", 1)[-1]
        tail = parts.rsplit("\n\n", 1)[-1] if "\n\n" in parts else parts
        out = tail
    return out.strip()


def extract_final_answer(text: str) -> str:
    """从模型输出里抽最终答案。

    优先级：\\boxed{} > #### > 「答案：」等标记 > 最后一行
    注意：带 \\boxed 时直接返回原文，交给 math_verify 的 boxed 优先策略处理。
    """
    if not text:
        return ""
    body = strip_thinking(text)
    if "\\boxed" in body or "\\fbox" in body:
        return body
    for pat in _ANSWER_MARKERS:
        m = pat.search(body)
        if m:
            return m.group(1).strip()
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    return lines[-1] if lines else body


# --------------------------------------------------------------------------- #
# 1) 准确性 —— 整体给分（不可 hack）
# --------------------------------------------------------------------------- #
def accuracy_reward(completions, answer, **kwargs) -> List[float]:
    """整体给分：符号等价判定，返回 0 / 1。

    math_verify 会做符号化简（1/2 == 0.5 == \\frac{1}{2}），
    不依赖任何模型 → 无法被 hack。
    """
    rewards: List[float] = []
    for completion, gold in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        try:
            # 关键：gold 也包进 \boxed{}，让两侧走同一条 anchor 抽取路径。
            # 实测这能把自洽率从 86.5% 提到 100.0%。
            gold_parsed = parse(
                f"\\boxed{{{gold}}}",
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()],
            )
            pred_parsed = parse(
                extract_final_answer(text),
                extraction_mode="first_match",
                extraction_config=_LATEX_PRED,
            )
            rewards.append(float(verify(gold_parsed, pred_parsed)))
        except Exception as e:  # 解析失败按错处理，绝不让异常打断训练
            logger.debug(f"accuracy_reward 解析失败: {type(e).__name__}: {e}")
            rewards.append(0.0)
    return rewards


# --------------------------------------------------------------------------- #
# 1b) 选择题（MCQ）—— 独立的集合比较判分
# --------------------------------------------------------------------------- #
#: 选项字母表（A~H 足够覆盖 8 选项的选择题）
MCQ_LETTERS = "ABCDEFGH"

#: LaTeX 里包裹纯文本的命令，先剥掉再抽字母（否则 \text{C} 会抽出 T/E/X/C）
_LATEX_TEXT_WRAP = re.compile(r"\\(?:text|mathrm|mathbf|mbox|operatorname)\s*\{([^{}]*)\}")

#: 显式答案标记：「答案：C」「应选 ABD」「故选 C」
_MCQ_MARKER = re.compile(
    r"(?:答案|应选|故选|选择|选)\s*(?:是|为)?\s*[:：]?\s*"
    r"([A-Ha-h](?:\s*[,，、和及]?\s*[A-Ha-h])*)"
)


def extract_mcq_letters(text: str, valid: str = MCQ_LETTERS) -> set:
    """从模型输出里抽出选项字母集合。

    优先级：最后一个 ``\\boxed{}`` > 显式答案标记 > 最后一个 ``(X)`` > 最后一行。

    返回大写字母的 **set**（选择题的答案是集合，多选天然支持）。
    抽不到任何合法字母时返回空 set —— 判分处会把它当错处理。
    """
    if not text:
        return set()
    body = strip_thinking(text)
    body = _LATEX_TEXT_WRAP.sub(r"\1", body)  # 剥掉 \text{...} 外壳

    candidate = None

    # 1) 最后一个 \boxed{...}（最可靠）
    boxed = re.findall(r"\\boxed\s*\{([^{}]{0,40})\}", body)
    if boxed:
        candidate = boxed[-1]

    # 2) 显式答案标记
    if candidate is None:
        m = _MCQ_MARKER.search(body)
        if m:
            candidate = m.group(1)

    # 3) 最后一个 (X)
    if candidate is None:
        parens = re.findall(r"\(\s*([A-Ha-h])\s*\)", body)
        if parens:
            candidate = parens[-1]

    # 4) 兜底：最后一行 —— 仅在它足够短、像"答案行"时才用。
    #    否则一段长推理文本会被抽出满屏随机字母（假阳性）。
    if candidate is None:
        lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
        if lines and len(lines[-1]) <= 20:
            candidate = lines[-1]

    if not candidate:
        return set()
    return {ch.upper() for ch in candidate if ch.upper() in valid}


def make_mcq_reward(valid_letters: str = MCQ_LETTERS) -> Callable:
    """构造选择题判分函数：预测字母集合与 gold 字母集合**完全相等**才给 1。

    注意：多选要求完全匹配（不一致的选项组合算错），这是评测惯例。
    诊断用的部分重合信息由 evaluate_math.py 在明细里单独输出。
    """

    def mcq_accuracy_reward(completions, answer, **kwargs) -> List[float]:
        rewards: List[float] = []
        for completion, gold in zip(completions, answer):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            gold_set = {c.upper() for c in str(gold) if c.upper() in valid_letters}
            pred_set = extract_mcq_letters(text, valid_letters)
            # 必须非空：抽不到字母 = 没作答 = 错
            rewards.append(1.0 if pred_set and pred_set == gold_set else 0.0)
        return rewards

    mcq_accuracy_reward.__name__ = "mcq_accuracy_reward"
    return mcq_accuracy_reward


#: 默认实例（4 选项），供直接调用 / 单元测试
mcq_accuracy_reward = make_mcq_reward()


# --------------------------------------------------------------------------- #
# 2) 规范性 —— 分层给分
# --------------------------------------------------------------------------- #
#: 分项权重（总和 1.0）
FMT_WEIGHTS = {
    "answer_marked": 0.5,      # 有最终答案标记（最重要）
    "has_structure": 0.3,      # 有可见推理结构
    "thinking_used": 0.1,      # 走了思考流程（需注入 think_end_token_id 才生效）
    "no_filler": 0.1,          # 无冗余套话
}

#: 「有推理结构」的判定阈值：正文至少这么多处换行
_MIN_NEWLINES = 3


def make_format_reward(think_end_token_id: Optional[int] = None) -> Callable:
    """构造规范性子分函数。

    Args:
        think_end_token_id: Qwen3 的 `response` token id。传了就额外奖励「走了思考流程」。
            **必须用 token id 判断**，因为 TRL 用 skip_special_tokens=True 解码，
            文本里根本不会出现 thinking 标记。

    返回 0 ~ 1 的连续值。为什么分档而不是二值：
        二值 format reward 在模型学会格式后恒为 1 → 退化成常数，不再提供梯度；
        分档能让「格式好坏」持续产生组内方差，直接缓解 GRPO 的零梯度问题。
    """

    def format_reward(completions, completion_ids=None, **kwargs) -> List[float]:
        rewards: List[float] = []
        for idx, completion in enumerate(completions):
            text = completion[0]["content"] if isinstance(completion, list) else str(completion)
            score = 0.0

            marked = ("\\boxed" in text or "####" in text
                      or re.search(r"(?:答案|答)\s*[:：]", text) is not None)
            structured = text.count("\n") >= _MIN_NEWLINES

            if marked:
                score += FMT_WEIGHTS["answer_marked"]
            if structured:
                score += FMT_WEIGHTS["has_structure"]

            # 思考流程：靠 token id 判断，不靠文本（见 docstring）
            if think_end_token_id is not None and completion_ids is not None:
                ids = completion_ids[idx]
                if hasattr(ids, "__len__") and think_end_token_id in list(ids):
                    score += FMT_WEIGHTS["thinking_used"]

            # 冗余套话：只在「有实际内容」时才给干净分，
            # 否则空回答/垃圾回答也会白拿 0.1 分，削弱组内区分度。
            if (marked or structured) and not _FILLER.match(strip_thinking(text)):
                score += FMT_WEIGHTS["no_filler"]

            rewards.append(round(score, 4))
        return rewards

    format_reward.__name__ = "format_reward"
    return format_reward


#: 默认实例（不检测 thinking），供直接调用 / 单元测试
format_reward = make_format_reward()


# --------------------------------------------------------------------------- #
# 3) 长度 —— DAPO Eq.13 软长度惩罚
# --------------------------------------------------------------------------- #
def _overlong_penalty(length: int, max_len: int, cache_len: int) -> float:
    """DAPO (arXiv:2503.14476) Eq.13。

        0                                    length <= max_len - cache_len
        (max_len - cache_len - length)/cache  max_len-cache < length <= max_len
        -1                                    length > max_len
    """
    if cache_len <= 0:
        return 0.0
    if length <= max_len - cache_len:
        return 0.0
    if length <= max_len:
        return (max_len - cache_len - length) / cache_len
    return -1.0


def make_length_reward(max_len: int, cache_len: int) -> Callable:
    """构造长度惩罚函数（需闭包捕获 max_len，所以用工厂）。

    Args:
        max_len:   生成的最大 token 数，应与 GRPOConfig.max_completion_length 一致
        cache_len: 惩罚区间宽度，论文常用 max_len 的 1/4 左右
    """

    def length_reward(completions, completion_ids=None, **kwargs) -> List[float]:
        if completion_ids is not None and len(completion_ids) > 0:
            lengths = [len(ids) for ids in completion_ids]
        else:  # 兜底：用字符数近似（不精确，仅防御）
            lengths = [
                len(c[0]["content"] if isinstance(c, list) else str(c)) for c in completions
            ]
        return [_overlong_penalty(n, max_len, cache_len) for n in lengths]

    length_reward.__name__ = "length_reward"
    return length_reward


# --------------------------------------------------------------------------- #
# 组装
# --------------------------------------------------------------------------- #
def build_reward_funcs(
    max_completion_length: int,
    format_weight: float = 0.3,
    length_weight: float = 1.0,
    length_cache_ratio: float = 0.25,
    think_end_token_id: Optional[int] = None,
) -> tuple:
    """构造奖励函数列表与对应权重。

    Returns:
        (reward_funcs, reward_weights)
        **注意顺序**：accuracy 必须在第 0 位，供 AR3POTrainer 判定「全错」。
    """
    cache_len = max(1, int(max_completion_length * length_cache_ratio))
    funcs = [
        accuracy_reward,                                              # index 0 ← AR3PO 依赖
        make_format_reward(think_end_token_id),                       # index 1
        make_length_reward(max_completion_length, cache_len),         # index 2
    ]
    weights = [1.0, format_weight, length_weight]
    logger.info(
        f"[rewards] accuracy(×1.0) + format(×{format_weight}) + "
        f"length(×{length_weight}, L_max={max_completion_length}, L_cache={cache_len}, "
        f"think_end_id={think_end_token_id})"
    )
    return funcs, weights
