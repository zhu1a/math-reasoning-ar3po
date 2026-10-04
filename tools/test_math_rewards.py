# -*- coding: utf-8 -*-
"""奖励函数单元测试：真实数据 + 构造输出 + 负向对照。

这是最关键的一道验证 —— 奖励函数写错 = 训练零梯度或奖励噪声。
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.math_rewards import (  # noqa: E402
    _overlong_penalty, accuracy_reward, build_reward_funcs,
    extract_final_answer, make_format_reward, strip_thinking,
)

FAIL = []


def check(name, got, want):
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {name}: got={got!r} want={want!r}")
    if not ok:
        FAIL.append(name)


def conv(text):
    return [[{"role": "assistant", "content": text}]]


print("=" * 78)
print("1) 答案抽取")
print("=" * 78)
check("strip Qwen3（闭合被剥）", strip_thinking(" thinking推理\n\n最终答案42"), "最终答案42")
check("strip R1 <think>", strip_thinking("<think>推理</think>答案7"), "答案7")
check("boxed 保留给 math_verify", extract_final_answer("推理...\\boxed{42}"), "推理...\\boxed{42}")
check("#### 风格", extract_final_answer("推理\n#### 18"), "18")
check("答案：风格", extract_final_answer("推理过程\n答案：3/4"), "3/4")
check("兜底取最后一行", extract_final_answer("第一步\n第二步\n最终是 99"), "最终是 99")

print()
print("=" * 78)
print("2) accuracy_reward：真实数据自洽率（模型答对必须给 1）")
print("=" * 78)
rows = ([json.loads(l) for l in open("data/eval/math500.jsonl", encoding="utf-8")]
        + [json.loads(l) for l in open("data/grpo/en/math_dapo_train.jsonl", encoding="utf-8")][:300])
n_pos = n_ok = 0
for r in rows:
    g = str(r["answer"]).strip()
    if not g:
        continue
    n_pos += 1
    if accuracy_reward(conv(f"思考过程…\n\n\\boxed{{{g}}}"), [g])[0] == 1.0:
        n_ok += 1
rate = n_ok / n_pos
print(f"  答对 → 给 1 分：{n_ok}/{n_pos} = {rate:.1%}")
if rate < 0.99:
    FAIL.append(f"答对率仅 {rate:.1%}（奖励噪声过大）")
else:
    print("  ✓ 无奖励噪声（修复前是 86.5%，会白扣 13.5% 的正确答案）")

print()
print("=" * 78)
print("3) accuracy_reward：负向对照（不能把所有答案都判对）")
print("=" * 78)
n_neg = n_neg_ok = 0
for r in rows[:400]:
    g = str(r["answer"]).strip()
    if not g:
        continue
    n_neg += 1
    if accuracy_reward(conv("思考…\n\n\\boxed{987654321}"), [g])[0] == 0.0:
        n_neg_ok += 1
nr = n_neg_ok / n_neg
print(f"  答错 → 给 0 分：{n_neg_ok}/{n_neg} = {nr:.1%}")
if nr < 0.999:
    FAIL.append(f"答错未给0的比例过高：{nr:.1%}")
else:
    print("  ✓ 判别力完好（没有因为修复而全部判对）")

print()
print("  符号等价边界：")
for text, gold, want, desc in [
    ("\\boxed{42}", "42.0", 1.0, "42 vs 42.0"),
    ("\\boxed{1/2}", "0.5", 1.0, "分数 vs 小数"),
    ("\\boxed{\\frac{1}{2}}", "\\frac{1}{2}", 1.0, "LaTeX 分数"),
    ("\\boxed{43}", "42", 0.0, "答错"),
    ("我算不出来", "42", 0.0, "无答案"),
    ("", "42", 0.0, "空输出"),
]:
    check(f"{desc}", accuracy_reward(conv(text), [gold])[0], want)

print()
print("=" * 78)
print("4) format_reward：分层给分（必须连续，提供组内方差）")
print("=" * 78)
fmt_cases = [
    ("结构+boxed+干净", "推理步骤\n第二行\n第三行\n\\boxed{42}", 0.9),
    ("只有 boxed", "\\boxed{42}", 0.6),
    ("boxed+套话", "好的，让我来解答\n\\boxed{42}", 0.5),
    ("裸答案", "42", 0.0),
    ("多行无 boxed", "第一步\n第二步\n第三步\n所以是 42", 0.4),
]
for desc, text, want in fmt_cases:
    got = make_format_reward()(conv(text))[0]
    ok = abs(got - want) < 1e-6
    print(f"  {'✓' if ok else '✗'} {desc}: got={got} want={want}")
    if not ok:
        FAIL.append(f"fmt:{desc}")

# thinking 检测必须靠 token id，不能靠文本（TRL 会剥掉 special token）
THINK_END = 151668
fr = make_format_reward(think_end_token_id=THINK_END)
got = fr(conv("\\boxed{42}"), completion_ids=[[11, 22, THINK_END, 33]])[0]
check("thinking 检测（含 think_end token）", got, 0.7)
got = fr(conv("\\boxed{42}"), completion_ids=[[11, 22, 33]])[0]
check("thinking 检测（不含）", got, 0.6)

scores = {make_format_reward()(conv(t))[0] for _, t, _ in fmt_cases}
print(f"\n  format_reward 取值集合: {sorted(scores)} → 共 {len(scores)} 档")
if len(scores) < 3:
    FAIL.append("format_reward 档位太少，无法提供组内方差")
else:
    print("  ✓ 分层给分有效：能提供组内方差（缓解零方差组的第一道防线）")

print()
print("=" * 78)
print("4b) MCQ 选择题判分（AGIEval 高考）")
print("=" * 78)

from training.math_rewards import extract_mcq_letters, mcq_accuracy_reward  # noqa: E402

mcq_extract_cases = [
    ("\\boxed{C}", "C"),
    ("推理…\n\\boxed{A}", "A"),
    ("**答案：D**", "D"),
    ("故选 B", "B"),
    ("答案是 (C)", "C"),
    ("\\boxed{\\text{C}}", "C"),            # LaTeX 外壳必须剥掉，否则会抽出 T/E/X
    ("多选：\\boxed{ABD}", "ABD"),
    ("多选：\\boxed{A, C}", "AC"),
    ("\\boxed{A、B、D}", "ABD"),
    ("我认为 \\boxed{x^2+1} 是对的", ""),     # 数学表达式不应被误判成字母
    ("", ""),
]
for text, want in mcq_extract_cases:
    got = "".join(sorted(extract_mcq_letters(text)))
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {text[:26]!r:30} → {got!r} (want {want!r})")
    if not ok:
        FAIL.append(f"mcq_extract:{text[:20]}")

mcq_judge_cases = [
    ("\\boxed{C}", "C", 1.0, "单选：字母正确"),
    ("\\boxed{A}", "C", 0.0, "单选：字母错误"),
    ("\\boxed{ABD}", "ABD", 1.0, "多选：完全匹配"),
    ("\\boxed{AB}", "ABD", 0.0, "多选：漏选算错"),
    ("\\boxed{ABCD}", "ABD", 0.0, "多选：多选算错"),
    ("我不会", "C", 0.0, "没作答"),
    ("\\boxed{}", "C", 0.0, "空 boxed"),
]
for text, gold, want, desc in mcq_judge_cases:
    got = mcq_accuracy_reward(conv(text), [gold])[0]
    ok = got == want
    print(f"  {'✓' if ok else '✗'} {desc:18} gold={gold:4} → {got}")
    if not ok:
        FAIL.append(f"mcq_judge:{desc}")

# --- 真实数据自洽 + 回归对比 ---
try:
    agi = [json.loads(l) for l in open("data/eval/agieval_gaokao.jsonl", encoding="utf-8")]
    n_ok = sum(
        1 for r in agi
        if mcq_accuracy_reward(conv(f"推理…\n\\boxed{{{r['answer']}}}"), [r["answer"]])[0] == 1.0
    )
    rate = n_ok / len(agi)
    print(f"\n  真实 AGIEval 自洽率: {n_ok}/{len(agi)} = {rate:.1%}")
    if rate < 1.0:
        FAIL.append(f"AGIEval 自洽率 {rate:.1%}")
    else:
        print("  ✓ 全部自洽（gold 字母一定得 1 分）")

    # 回归：修复前的失败模式 —— 模型输出选项字母被判 0
    # 用「完整字母集合」作答（单选=1个字母，多选=多个字母），这是最自然的方式
    n_old = sum(
        1 for r in agi
        if accuracy_reward(conv(f"\\boxed{{{r['answer']}}}"), [str(r["answer_indices"])])[0] == 1.0
    )
    n_new = sum(
        1 for r in agi
        if mcq_accuracy_reward(conv(f"\\boxed{{{r['answer']}}}"), [r["answer"]])[0] == 1.0
    )
    print("\n  【回归对比】模型按最自然方式作答（输出选项字母集合）：")
    print(f"     旧路径 math_verify（拿字母比下标）: {n_old}/{len(agi)}   ← 修复前会全部判 0")
    print(f"     新路径 MCQ 集合比较              : {n_new}/{len(agi)}   ← 修复后")
    if n_new < len(agi) * 0.99:
        FAIL.append("MCQ 修复未完全生效")
except FileNotFoundError:
    print("  (跳过：data/eval/agieval_gaokao.jsonl 不存在)")

print()
print("=" * 78)
print("5) length_reward：DAPO Eq.13 软长度惩罚")
print("=" * 78)
L, C = 2048, 512
for length, want in [(100, 0.0), (1536, 0.0), (1792, -0.5), (2048, -1.0), (3000, -1.0)]:
    check(f"len={length}", _overlong_penalty(length, L, C), want)
print("  ✓ 正常长度不惩罚（软着陆），只在接近截断时惩罚")

print()
print("=" * 78)
print("6) 组装：顺序 + 权重")
print("=" * 78)
funcs, weights = build_reward_funcs(max_completion_length=2048, think_end_token_id=THINK_END)
names = [f.__name__ for f in funcs]
print(f"  reward_funcs   = {names}")
print(f"  reward_weights = {weights}")
check("accuracy 在 index 0（AR3PO 依赖）", names[0], "accuracy_reward")
check("函数名可用于日志", names[1], "format_reward")
check("权重数 == 函数数", len(weights), len(funcs))
check("accuracy 权重最大", weights[0] == max(weights), True)

print()
print("=" * 78)
print("结论")
print("=" * 78)
if FAIL:
    print(f"❌ {len(FAIL)} 项失败:")
    for f in FAIL:
        print("   -", f)
    sys.exit(1)
print("✅ 全部通过：奖励链路在真实数据上工作正常")
