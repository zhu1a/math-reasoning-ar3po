"""AR3PO × TRL 端到端验证（离线极小模型，不需要 GPU / 不需要 HF Hub）。

验证五件事：
  1. AR3PO 能在 TRL GRPOTrainer 上跑通完整训练步（不炸）
  2. 组装后 completion 数量 == 入参 prompts 数量（满足上游 view(-1, G) 约束）
  3. 补齐样本的 tool_mask 全 0（真正停梯度）
  4. 生成量下降（自适应 rollout 的核心收益）
  5. advantage 仍在完整的 G 条上计算（补齐样本参与组内归一化）
"""
import sys

import torch

sys.path.insert(0, "/Users/liu/Downloads/MedicalGPT-main")

from datasets import Dataset  # noqa: E402
from tokenizers import Tokenizer, models, pre_tokenizers, trainers  # noqa: E402
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from training.ar3po_trainer import AR3POConfig, AR3POTrainer  # noqa: E402

torch.manual_seed(0)


# --------------------------------------------------------------------- #
# 离线极小分词器 + 模型
# --------------------------------------------------------------------- #
def build_tokenizer():
    corpus = [f"What is {i} plus zero ? the answer is {i} ." for i in range(20)]
    corpus += ["What", "is", "plus", "zero", "?", "the", "answer", ".", "EASY", "HARD"]
    corpus += [str(i) for i in range(20)]
    tk = Tokenizer(models.WordLevel(unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator(corpus, trainers.WordLevelTrainer(
        vocab_size=200, special_tokens=["[UNK]", "[PAD]", "[EOS]"]))
    return PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="[UNK]",
                                   pad_token="[PAD]", eos_token="[EOS]")


TOKENIZER = build_tokenizer()


def build_model():
    cfg = Qwen2Config(vocab_size=TOKENIZER.vocab_size, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                      max_position_embeddings=128, tie_word_embeddings=True)
    return Qwen2ForCausalLM(cfg)


def spoof_reward(completions, answer, **kwargs):
    """机制测试用奖励：EASY → 全对（触发补齐分支）；HARD → 全错（触发第 2 轮）。"""
    return [1.0 if a == "EASY" else 0.0 for a in answer]


ROWS = [{"prompt": f"What is {i} plus zero ?",
         "answer": "EASY" if i % 2 == 0 else "HARD"} for i in range(8)]
DS = Dataset.from_list(ROWS)


# --------------------------------------------------------------------- #
# 探针
# --------------------------------------------------------------------- #
class Probe:
    """记录：真实生成量、_generate 契约、tool_mask、advantage 形状。"""

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        self.gen_count += len(prompt_ids)
        return super()._generate_single_turn(prompt_ids, images, multimodal_fields)

    def _generate(self, prompts):
        out = super()._generate(prompts)
        n_in, n_out = len(prompts), len(out[1])
        assert n_out == n_in, f"契约破坏: {n_out} != {n_in}"
        self.contract.append((n_in, n_out))
        self.masks.append(None if out[2] is None else [list(m) for m in out[2]])
        return out

    def _compute_loss(self, model, inputs):
        if "advantages" in inputs:
            self.adv_rows.append(int(inputs["advantages"].shape[0]))
            self.adv_vals.append(inputs["advantages"].detach().flatten().tolist())
        return super()._compute_loss(model, inputs)


class BaseProbe(Probe, GRPOTrainer):
    pass


class AR3POProbe(Probe, AR3POTrainer):
    pass


def make_args(cls, **over):
    kw = dict(
        output_dir="/tmp/ar3po_test", per_device_train_batch_size=16,
        generation_batch_size=16, num_generations=4, max_completion_length=12,
        max_steps=1, logging_steps=1, save_strategy="no", report_to=[],
        use_vllm=False, bf16=False, fp16=False, gradient_checkpointing=False,
    )
    kw.update(over)
    return cls(**kw)


def run(trainer_cls, args):
    t = trainer_cls(model=build_model(), processing_class=TOKENIZER,
                    reward_funcs=[spoof_reward], args=args, train_dataset=DS)
    t.gen_count, t.contract, t.masks, t.adv_rows, t.adv_vals = 0, [], [], [], []
    t.train()
    return t


print("=" * 78)
print("A) 基线 GRPO  (G=4, 无自适应)")
base = run(BaseProbe, make_args(GRPOConfig))
print(f"  真实生成条数 = {base.gen_count}")
print(f"  _generate 契约 = {base.contract}")
print(f"  tool_mask = {'None（无）' if base.masks[0] is None else '有'}")

print("\n" + "=" * 78)
print("B) AR3PO      (G=4, S=2 → k=2)")
ar = run(AR3POProbe, make_args(AR3POConfig, ar3po_stages=2))
print(f"  真实生成条数 = {ar.gen_count}")
print(f"  _generate 契约 = {ar.contract}")

masks = ar.masks[0]
n_rows = len(masks)
zero_rows = sum(1 for m in masks if m is not None and sum(m) == 0)
nonzero_rows = n_rows - zero_rows
print(f"  tool_mask 总行数 = {n_rows}（应等于 generation_batch_size = 16）")
print(f"  全 0 mask 的行数 = {zero_rows}   ← 这些就是「复制补齐、停梯度」的样本")
print(f"  正常 mask 的行数 = {nonzero_rows}")
print(f"  advantage 行数 = {ar.adv_rows}（应在完整 G 条上计算）")

print("\n" + "=" * 78)
print("结论")
print("=" * 78)
print(f"[1] 能在 TRL 上跑通              : 是（基线 + AR3PO 都完成训练步）")
print(f"[2] 契约 len(out) == len(in)     : "
      f"{all(a == b for a, b in ar.contract)}   {ar.contract}")
print(f"[3] 补齐样本确实被停梯度          : "
      f"{zero_rows} 行 tool_mask 全 0（>0 即为通过）")
print(f"[4] 生成量下降                    : 基线 {base.gen_count} → AR3PO {ar.gen_count}"
      f"（节省 {1 - ar.gen_count / max(1, base.gen_count):.0%}）")
print(f"[5] advantage 仍在完整 G=4 条上算 : {ar.adv_rows}")
