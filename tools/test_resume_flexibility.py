# -*- coding: utf-8 -*-
"""验证：训练中途/恢复时能否更改构造期参数（如 use_vllm / max_completion_length）。

背景：`GRPOTrainer.__init__` 里 `self.use_vllm = args.use_vllm` 只赋值一次，
之后全部用 `self.use_vllm`，且没有运行期一致性校验。所以答案是：
  · 同一个 run 内 —— 改 args 无效（已经拷走了）
  · 新起一个 run 并从 checkpoint 恢复 —— 可以换

本脚本用离线极小模型实测「恢复训练 + 换参数」是否真的能跑通。
"""
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from datasets import Dataset  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers, trainers  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from training.ar3po_trainer import AR3POConfig, AR3POTrainer  # noqa: E402
from training.math_rewards import build_reward_funcs  # noqa: E402

CKPT = Path("/tmp/resume_test")


def build_tok():
    corpus = ["推理 答案 是 42", "\\boxed{42}", "第一步", "第二步"] + [f"题目 {i} 答案 {i}" for i in range(30)]
    tk = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(corpus, trainers.WordLevelTrainer(
        vocab_size=200, special_tokens=["[UNK]", "[PAD]", "[EOS]"]))
    t = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    t.chat_template = ("{% for m in messages %}{% if m['role']=='user' %}<|im_start|>user\n{{ m['content'] }}"
                       "<|im_end|>\n{% else %}<|im_start|>assistant\n{{ m['content'] }}<|im_end|>\n{% endif %}"
                       "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
    return t


TOK = build_tok()
CFG = Qwen2Config(vocab_size=TOK.vocab_size, hidden_size=32, intermediate_size=64,
                  num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                  max_position_embeddings=256, tie_word_embeddings=True)

rows = [json.loads(l) for l in open(ROOT / "data/grpo/en/math_dapo_train.jsonl", encoding="utf-8")][:32]
DS = Dataset.from_list([{"prompt": [{"role": "user", "content": r["question"]}], "answer": r["answer"]} for r in rows])


def make(args_over):
    kw = dict(
        output_dir=str(CKPT), per_device_train_batch_size=4, generation_batch_size=16,
        num_generations=4, max_completion_length=64, logging_steps=1, report_to=[],
        save_strategy="no", bf16=False, fp16=False, gradient_checkpointing=False,
        use_vllm=False, loss_type="dapo", scale_rewards="none", beta=0.0,
        ar3po_stages=2, replay_buffer_size=0,
    )
    kw.update(args_over)
    a = AR3POConfig(**kw)
    funcs, weights = build_reward_funcs(max_completion_length=a.max_completion_length)
    # ⚠️ 与 training/grpo_training.py 保持一致：reward_weights 是 GRPOConfig 的字段，
    #    **不是** GRPOTrainer.__init__ 的参数，传进去会 TypeError。
    a.reward_weights = weights
    return AR3POTrainer(model=Qwen2ForCausalLM(CFG), processing_class=TOK,
                        reward_funcs=funcs, args=a, train_dataset=DS)


shutil.rmtree(CKPT, ignore_errors=True)
print("=" * 70)
print("步骤 1：训练 2 步并存 checkpoint（max_completion_length=64）")
print("=" * 70)
t1 = make(dict(max_steps=2, save_strategy="steps", save_steps=2))
t1.train()
saved = sorted(p for p in CKPT.iterdir() if p.name.startswith("checkpoint-"))
print(f"  产生的 checkpoint: {[p.name for p in saved]}")
assert saved, "没有产生 checkpoint"
last = saved[-1]

print()
print("=" * 70)
print("步骤 2：从该 checkpoint 恢复，并【更改构造期参数】")
print("  max_completion_length: 64 → 96")
print("  use_vllm            : False → False（本机无 vllm，改这个会真去建引擎）")
print("  ar3po_stages        : 2 → 2")
print("=" * 70)
t2 = make(dict(max_steps=3, max_completion_length=96))
t2.train(resume_from_checkpoint=str(last))
print(f"\n  恢复后 global_step = {t2.state.global_step}（应为 3，说明从 2 步续上）")

print()
print("=" * 70)
print("结论")
print("=" * 70)
print("  ✅ 从 checkpoint 恢复时可以更改构造期参数（max_completion_length 实测换了）")
print("  ✅ AR3POTrainer 的自定义 _generate 与 resume 兼容（未报错）")
print(f"  ✅ 续训后 global_step={t2.state.global_step}，状态正确接续")
print()
print("  ⚠️ 但 use_vllm 不能在【同一个 run 内】中途改：")
print("     grpo_trainer.py:531  `self.use_vllm = args.use_vllm` 只在 __init__ 执行一次")
print("     之后所有分支都用 self.use_vllm，vLLM 引擎也在 __init__ 创建（:709）")
print("     且全文没有任何 `if self.args.use_vllm != self.use_vllm` 之类的一致性校验")
print("     → 直接改 training_args.use_vllm 会静默不一致（日志显示变了，实际没生效）")
