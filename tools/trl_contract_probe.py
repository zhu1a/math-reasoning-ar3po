"""探针脚本（离线版）：从零构造极小模型，实测 TRL 0.29.1 GRPOTrainer 的数据契约。

不依赖 HF Hub：本地随机初始化一个 2 层 Qwen2，并用 tokenizers 现场训练一个极小分词器。
目的：确认 prompts 长度 / completion_ids 长度 / num_generations 的关系，
      以及 tool_mask=0 能否屏蔽某条序列的梯度。
"""
import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import (
    PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM,
)
from trl import GRPOConfig, GRPOTrainer

torch.manual_seed(0)

# ---------------- 1. 极小分词器（词表级，现场训练）----------------
CORPUS = [
    f"What is {i} plus zero ? the answer is {i} ." for i in range(30)
] + ["What", "is", "plus", "zero", "?", "the", "answer", ".", "0", "1", "2", "3",
     "4", "5", "6", "7", "8", "9", "10", "11", "12", "16", "17", "18", "19",
     "20", "21", "22", "23", "24", "25", "26", "27", "28", "29"]

tk = Tokenizer(models.WordLevel(unk_token="[UNK]"))
tk.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.WordLevelTrainer(vocab_size=200, special_tokens=["[UNK]", "[PAD]", "[EOS]"])
tk.train_from_iterator(CORPUS, trainer)
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
)
print(f"vocab_size = {tokenizer.vocab_size}", flush=True)

# ---------------- 2. 极小模型（随机初始化）----------------
cfg = Qwen2Config(
    vocab_size=tokenizer.vocab_size,
    hidden_size=32, intermediate_size=64, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128,
    tie_word_embeddings=True,
)
model = Qwen2ForCausalLM(cfg)
print(f"model params = {sum(p.numel() for p in model.parameters())/1e3:.1f}K", flush=True)

# ---------------- 3. 数据：非对话式，prompt 为纯字符串 ----------------
rows = [{"prompt": f"What is {i} plus zero ?", "answer": str(i)} for i in range(12)]
ds = Dataset.from_list(rows)


def answer_reward(completions, answer, **kwargs):
    """可验证奖励：回复里出现正确数字给 1，否则 0。"""
    out = []
    for c, a in zip(completions, answer):
        text = c if isinstance(c, str) else str(c)
        out.append(1.0 if a in text.split() else 0.0)
    return out


class ProbeTrainer(GRPOTrainer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.probe_calls = []

    def _generate(self, prompts):
        result = super()._generate(prompts)
        (prompt_ids, completion_ids, tool_mask, completions,
         n_tokens, logprobs, extra_fields) = result
        self.probe_calls.append({
            "len_prompts": len(prompts),
            "self.num_generations": self.num_generations,
            "len_completion_ids": len(completion_ids),
            "tool_mask_is_none": tool_mask is None,
            "len_completions": len(completions),
            "n_tokens": int(n_tokens),
            "generation_batch_size": self.args.generation_batch_size,
        })
        print("\n>>> PROBE _generate:", self.probe_calls[-1], flush=True)
        return result


args = GRPOConfig(
    output_dir="/tmp/ar3po_probe",
    per_device_train_batch_size=2,
    generation_batch_size=8,
    num_generations=4,
    max_completion_length=16,
    max_steps=1,
    logging_steps=1,
    save_strategy="no",
    report_to=[],
    use_vllm=False,
    bf16=False,
    fp16=False,
    gradient_checkpointing=False,
)

t = ProbeTrainer(
    model=model, processing_class=tokenizer,
    reward_funcs=[answer_reward], args=args, train_dataset=ds,
)

print("\n=== 开始训练 1 步 ===", flush=True)
t.train()

print("\n=== 探针结果 ===")
for c in t.probe_calls:
    print(c)
c0 = t.probe_calls[0]
print("\n[契约] len_completion_ids == len_prompts ?",
      c0["len_completion_ids"] == c0["len_prompts"])
print("[契约] len_completion_ids == len_prompts * num_generations ?",
      c0["len_completion_ids"] == c0["len_prompts"] * c0["self.num_generations"])
