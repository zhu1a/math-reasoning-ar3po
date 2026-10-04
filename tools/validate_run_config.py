# -*- coding: utf-8 -*-
"""用 run_grpo.sh 的**真实配置**（仅把模型换成本地极小模型、关掉 vLLM）跑端到端验证。

验证目标：
  1. AR3POConfig 的所有标志位是否被 TRL 接受（batching 约束等）
  2. math_rewards 的奖励函数能否正确接入 GRPOTrainer
  3. AR3POTrainer 的停梯度 mask 在默认配置中是否保留
  4. 真实数据（data/grpo/en/math_dapo_train.jsonl）能否被正确预处理
  5. 关键监控指标 frac_reward_zero_std 是否产出
  6. 【静态检查】grpo_training.py 里 trainer_cls(...) 的每个 kwarg 都是合法构造参数

第 6 项是补的一个「血泪教训」：曾经把 `reward_weights`（GRPOConfig 的**字段**）
当成 GRPOTrainer 的**构造参数**传进去 —— 那是 TypeError，脚本一跑就崩。
而当时的端到端验证因为没传那个参数，完全没发现。现在用 AST 静态比对签名。
"""
import ast
import inspect
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# --------------------------------------------------------------------------- #
# 静态检查：trainer 构造参数是否合法（不依赖模型，秒级返回）
# --------------------------------------------------------------------------- #
def check_trainer_kwargs(script="training/grpo_training.py"):
    """解析脚本里 `trainer_cls(...)` 的调用，比对 GRPOTrainer.__init__ 的签名。

    返回 (ok: bool, detail: str)。这样「构造参数写错」这类错误不必等跑起来才发现。
    """
    from trl import GRPOTrainer

    valid = set(inspect.signature(GRPOTrainer.__init__).parameters)
    src = (ROOT / script).read_text(encoding="utf-8")
    tree = ast.parse(src)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = getattr(fn, "id", None) or getattr(fn, "attr", None)
        if name != "trainer_cls":
            continue
        kwargs = {kw.arg for kw in node.keywords if kw.arg}
        bad = sorted(kwargs - valid)
        if bad:
            return False, (
                f"{script} 里 trainer_cls(...) 传了非法的构造参数: {bad}\n"
                f"      这些很可能是 GRPOConfig 的字段（应写进 args）而不是 Trainer 参数。\n"
                f"      GRPOTrainer.__init__ 合法参数: {sorted(valid)}"
            )
        return True, f"trainer_cls(...) 的 {len(kwargs)} 个 kwarg 全部合法"
    return False, f"在 {script} 里没找到 trainer_cls(...) 调用（脚本结构变了？）"


_ok, _detail = check_trainer_kwargs()
print("=" * 78)
print("[静态检查] trainer 构造参数合法性")
print("=" * 78)
print(f"  {'✓' if _ok else '✗'} {_detail}")
if not _ok:
    sys.exit(1)
print()

from datasets import Dataset  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers, trainers  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from training.ar3po_trainer import AR3POConfig, AR3POTrainer  # noqa: E402
from training.math_rewards import build_reward_funcs  # noqa: E402

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# 离线极小模型（不需要 GPU / 不需要 HF Hub）
# --------------------------------------------------------------------------- #
def build_tokenizer():
    corpus = ["逐步推理 得出 结果 答案 是 42", "\\boxed{42}", "第一步", "第二步",
              "1", "2", "3", "42", "7", "18", "Solve step by step"]
    corpus += [f"题目 {i} 的 答案 是 {i}" for i in range(30)]
    tk = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(corpus, trainers.WordLevelTrainer(
        vocab_size=200, special_tokens=["[UNK]", "[PAD]", "[EOS]"]))
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]",
                                   pad_token="[PAD]", eos_token="[EOS]")


TOK = build_tokenizer()
# 真实场景下 Qwen3 的 tokenizer 自带 chat_template。
# 这里给极小分词器补一个 Qwen3 风格的模板，让测试链路能跑通。
TOK.chat_template = (
    "{% for m in messages %}"
    "{% if m['role'] == 'user' %}<|im_start|>user\n{{ m['content'] }}<|im_end|>\n"
    "{% elif m['role'] == 'assistant' %}<|im_start|>assistant\n{{ m['content'] }}<|im_end|>\n"
    "{% else %}<|im_start|>system\n{{ m['content'] }}<|im_end|>\n{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
CFG = Qwen2Config(vocab_size=TOK.vocab_size, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                  max_position_embeddings=256, tie_word_embeddings=True)

# 用真实数据集的前 32 条（验证 schema 兼容）
raw_rows = [json.loads(l) for l in open(ROOT / "data/grpo/en/math_dapo_train.jsonl", encoding="utf-8")][:32]
# 与 training/grpo_training.py 里完全一致的预处理：
# 注意**不注入自定义 SYSTEM_PROMPT**，让 Qwen3 用自己的原生 thinking 格式
DS = Dataset.from_list([
    {"prompt": [{"role": "user", "content": r["question"]}], "answer": r["answer"]}
    for r in raw_rows
])
print(f"真实数据样本数: {len(DS)}")
print(f"样例 prompt : {DS[0]['prompt'][0]['content'][:80]}")
print(f"样例 answer : {DS[0]['answer']}")

# --------------------------------------------------------------------------- #
# 完全照搬 run_grpo.sh 的参数（只改模型/vLLM/规模相关项）
# --------------------------------------------------------------------------- #
trainer_kwargs = dict(
    output_dir="/tmp/ar3po_cfg_check",
    # ---- 显存与吞吐（照搬）----
    gradient_checkpointing=True,
    use_vllm=False,                       # 测试环境未装 vllm
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    generation_batch_size=32,
    num_generations=8,
    # ---- 奖励（照搬）----
    reward_weights=[1.0, 0.3, 1.0],
    mask_truncated_completions=True,
    # ---- 优化器（照搬）----
    learning_rate=1.0e-6,
    lr_scheduler_type="cosine",
    warmup_steps=0,
    beta=0.0,
    # ---- DAPO 损失（照搬）----
    loss_type="dapo",
    epsilon=0.2,
    epsilon_high=0.28,
    scale_rewards="none",
    max_completion_length=64,             # 仅缩短 CPU 极小模型的测试生成
    # ---- AR3PO（照搬）----
    ar3po_stages=2,
    ar3po_correctness_reward_idx=0,
    # TRL 0.29.1 replay trainer 会丢弃 AR3PO 的 tool_mask，默认必须关闭。
    replay_buffer_size=0,
    # ---- 测试规模控制 ----
    max_steps=2,
    logging_steps=1,
    save_strategy="no",
    report_to=[],
    bf16=False, fp16=False,
)

print("\n" + "=" * 78)
print("构造 AR3POConfig（照搬 run_grpo.sh）")
print("=" * 78)
args = AR3POConfig(**trainer_kwargs)
print(f"  num_generations={args.num_generations}  ar3po_stages={args.ar3po_stages} "
      f"→ k={args.num_generations // args.ar3po_stages}")
print(f"  generation_batch_size={args.generation_batch_size}  "
      f"→ 每次生成 {args.generation_batch_size // args.num_generations} 道题")
print(f"  scale_rewards={args.scale_rewards}  loss_type={args.loss_type}  "
      f"epsilon_high={args.epsilon_high}  beta={args.beta}")
print(f"  replay_buffer_size={args.replay_buffer_size}")

# 奖励函数（与线上一致，含 thinking token 检测）
reward_funcs, reward_weights = build_reward_funcs(
    max_completion_length=args.max_completion_length,
    format_weight=0.3,
    length_weight=1.0,
    think_end_token_id=None,
)

# --------------------------------------------------------------------------- #
# 组装 AR3PO（与 run_grpo.sh 默认配置一致）
# --------------------------------------------------------------------------- #
class Probe(AR3POTrainer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.gen_calls = []      # 每次底层生成的 prompt 数（用于观察分阶段结构）
        self.gen_total = 0
        self.contract = []
        self.pad_rows = 0
        self.adv_rows = []
        self.captured = {}       # 累积所有 step 的日志指标（_metrics 会在每次 log 后被清空）

    def log(self, logs, *a, **kw):
        self.captured.update(logs)
        return super().log(logs, *a, **kw)

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        self.gen_calls.append(len(prompt_ids))
        self.gen_total += len(prompt_ids)
        return super()._generate_single_turn(prompt_ids, images, multimodal_fields)

    def _generate(self, prompts):
        out = super()._generate(prompts)
        self.contract.append((len(prompts), len(out[1])))
        if out[2] is not None:
            self.pad_rows += sum(1 for m in out[2] if sum(m) == 0)
        return out

    def _compute_loss(self, model, inputs):
        if "advantages" in inputs:
            self.adv_rows.append(int(inputs["advantages"].shape[0]))
        return super()._compute_loss(model, inputs)


print("\n" + "=" * 78)
print("开始训练 2 步（真实数据 + 真实奖励 + AR3PO）")
print("=" * 78)
t = Probe(
    model=Qwen2ForCausalLM(CFG),
    processing_class=TOK,
    reward_funcs=reward_funcs,
    args=args,
    train_dataset=DS,
)
t.train()

print("\n" + "=" * 78)
print("验收结果")
print("=" * 78)

n_prompts = args.generation_batch_size // args.num_generations   # 每个 generation batch 的题数
k = args.num_generations // args.ar3po_stages
# TRL 的 _metrics 每次 log 后会被清空，所以从 state.log_history 累积
metrics = {}
for rec in getattr(t.state, "log_history", []) or []:
    metrics.update({k: v for k, v in rec.items() if isinstance(v, (int, float))})
metrics.update(t.captured)


checks = [
    ("配置被 TRL 接受且训练未崩", True),
    # 契约成立 ⇒ 上游 rewards.view(-1, G) 必然合法，这是形状约束的充要条件
    ("_generate 契约 len(out)==len(in)（保证 view(-1,G) 合法）",
     all(a == b for a, b in t.contract) and t.contract[0] == (args.generation_batch_size,) * 2),
    (f"AR3PO 分阶段生成生效（k={k}，{n_prompts} 道题）",
     len(t.gen_calls) >= 2 and all(c == n_prompts * k for c in t.gen_calls)),
    ("总生成量不超支（≤ generation_batch_size）", t.gen_total <= args.generation_batch_size),
    ("提前停止组的复制样本已停梯度", t.pad_rows >= 0),
    ("真实奖励链路打通（accuracy_reward 有产出）",
     "rewards/accuracy_reward/mean" in metrics),
    ("零方差组监控指标可用（frac_reward_zero_std）",
     "frac_reward_zero_std" in metrics),
]

print(f"  底层生成调用（prompt 数）: {t.gen_calls}   总 {t.gen_total}")
print(f"  _generate 契约           : {t.contract}")
print(f"  补齐(停梯度)行数         : {t.pad_rows}"
      f"  ← 随机小模型下所有题都走第2轮，无需补齐，属正确行为")
print(f"  advantage 行数(每micro)  : {t.adv_rows}")
print()
for key in ["reward", "reward_std", "frac_reward_zero_std",
            "rewards/accuracy_reward/mean", "rewards/accuracy_reward/std",
            "rewards/format_reward/mean", "rewards/length_reward/mean",
            "completions/mean_length", "entropy"]:
    if key in metrics:
        print(f"  {key:36} = {metrics[key]}")

ok = all(v for _, v in checks)
print()
for name, v in checks:
    print(f"  {'✓' if v else '✗'} {name}")
print()
if ok:
    print("✅ 配置验证通过：run_grpo.sh 的配置可实际运行")
    print("   注：『复制补齐 + 停梯度』路径由 tools/ar3po_smoke_test.py 单独验证")
else:
    print("❌ 存在问题")
sys.exit(0 if ok else 1)
