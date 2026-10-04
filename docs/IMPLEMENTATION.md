# 项目实现逻辑与排查手册

> 目的：让你能**按症状定位问题**。先看第二章的流程和第三章的技术清单建立全局认识，
> 出问题时直接跳到第六章「排查手册」按症状索引。

---

## 一、全局架构：三条流水线

```
data/_raw/*.parquet                       ← tools/download_math_datasets.py（走 hf-mirror）
        │
        ▼ tools/build_math_dataset.py     ← 转成三种 schema
data/{grpo,sft,reward,eval}/*.jsonl
        │
        ├─▶ ① SFT       scripts/run_sft.sh    training/supervised_finetuning.py
        │        outputs-sft-qwen3-8b-math-merged   （LoRA 合并后交给 GRPO）
        │
        ├─▶ ② GRPO+AR3PO scripts/run_grpo.sh  training/grpo_training.py
        │        outputs-grpo-qwen3-8b-ar3po-merged （LoRA 合并后用于 rollout / DPO）
        │            │
        │            └─▶ 导出 rollout ─▶ tools/export_dpo_pairs.py
        │                                        data/reward/math_dpo_pairs.jsonl
        │
        └─▶ ③ DPO       scripts/run_dpo.sh    training/dpo_training.py
                 outputs-dpo-qwen3-8b-math-merged   （最终可直接推理模型）
```

**为什么顺序不能换**：
- GRPO 不能从 base/instruct 裸跑（[Secrets of RLHF](https://arxiv.org/abs/2307.04964) 实验证明会失败、语言能力退化）→ 必须先 SFT
- DPO 的偏好对来自 GRPO 的 rollout → 必须后于 GRPO

---

## 二、数据流转与 schema（每一步都不同，最容易搞错）

### 2.1 三种 schema 严格区分

| 目录 | JSONL schema | 谁消费 | 转换位置 |
|---|---|---|---|
| `data/grpo/` | `{"question": str, "answer": str, "source": str, "question_type": str}` | `grpo_training.py` | 脚本内 `.map` → `{"prompt": [{"role":"user","content":...}], "answer": ...}` |
| `data/sft/` | `{"conversations": [{"from":"human"/"gpt", "value": str}]}` | `supervised_finetuning.py` | 脚本内 `preprocess_function` → `{input_ids, attention_mask, targets}` |
| `data/reward/` | `{"conversations": [...], "chosen": str, "rejected": str}` | `dpo_training.py` | `_build_prompt_from_conversations` → `{prompt, chosen, rejected}` |
| `data/eval/` | `{"question", "answer", "source", "question_type"}` | 评估脚本 | 无 |

⚠️ **GRPO 和 SFT 用的都是「问题」但格式完全不同**：GRPO 要 `question`+`answer`（问答对），SFT 要 `conversations`（多轮对话）。`tools/build_math_dataset.py` 负责分别生成。

### 2.2 已就绪的数据

| 文件 | 行数 | 说明 |
|---|---|---|
| `data/grpo/en/math_dapo_train.jsonl` | 12,000 | 阶段一（英文，DAPO-Math 去重，真实唯一题 17,186） |
| `data/grpo/zh/math_zh_train.jsonl` | 20,000 | 阶段二（中文，MathInstruct-Chinese 英译中） |
| `data/grpo/proof/numina_proof.jsonl` | 5,000 | 证明题，`verifiable=false`，**默认不进 GRPO** |
| `data/sft/math_sft_demo.jsonl` | 2,000 | NuminaMath 真实完整解题过程，用于 SFT 冷启动 |
| `data/eval/*.jsonl` | 500/30/351/500 | MATH-500 / AIME24 / AGIEval高考 / DAPO holdout |
| `data/reward/math_dpo_pairs.jsonl` | 动态 | DPO 偏好对 |

⚠️ **`--train_file_dir` 会把目录下【所有】jsonl 合并加载，且要求 schema 完全一致**。
所以三类数据**必须分目录**（这是实测踩出来的坑：混放会直接抛 `CastError`）：

```
data/grpo/en/     ← 阶段一英文（schema: question,answer,source,question_type）
data/grpo/zh/     ← 阶段二中文（同上 + reference_solution）
data/grpo/proof/  ← 证明题（同上 + verifiable + reference_solution）
```

`run_grpo.sh` 默认 `--train_file_dir data/grpo/en`。切中文改这一个参数即可。
`grpo_training.py` 里已加 **schema 预检**，混放时会给出清晰报错而不是晦涩的 CastError。

---

## 三、每一层用了什么技术

### ① SFT 层（`training/supervised_finetuning.py`）
| 组件 | 说明 |
|---|---|
| Trainer | transformers `Trainer` 子类 `SavePeftModelTrainer` |
| 微调方式 | LoRA（peft），`--target_modules all` |
| 输入构造 | chat 模板渲染 → tokenize；**只对 assistant 段计算 loss**（`targets` 里非 assistant 位置是 -100） |
| 显存 | `gradient_checkpointing=True` + bf16 |

### ② GRPO 层（`training/grpo_training.py` + `ar3po_trainer.py`）
| 组件 | 用了什么 |
|---|---|
| 基类 | TRL `GRPOTrainer`（0.29.1） |
| 子类 | `AR3POTrainer`（自适应 rollout，自己实现） |
| replay buffer | **默认关闭**：TRL 0.29.1 实验实现会丢掉 AR3PO 的 `tool_mask`，导致复制样本错误地产生梯度 |
| advantage | 组内归一化；`scale_rewards="none"` → `A = R − mean(R)`（不除 std） |
| 损失 | `loss_type="dapo"` → token-level + `clip(r, 1−0.2, 1+0.28)`（clip-higher） |
| KL | `beta=0` → **无 KL、无 reference 模型**（省一份前向） |
| 采样 | AR3PO 分阶段生成（S=2，k=G/2） |
| 停梯度 | 复用 `tool_mask`（置 0 即该序列不产生梯度） |

### ③ 奖励层（`training/math_rewards.py`）
```
R = 1.0 · accuracy  +  0.3 · format  +  1.0 · length
```
| 奖励 | 技术 | 关键点 |
|---|---|---|
| `accuracy_reward` | `math_verify.parse` + `verify` | **gold 也包 `\boxed{}`**；符号等价（`1/2 == 0.5 == \frac{1}{2}`） |
| `format_reward` | 4 分项加权：答案标记 0.5 / 结构 0.3 / thinking 0.1 / 无套话 0.1 | **分层不是二值**；thinking 靠 **token id** 判断 |
| `length_reward` | DAPO Eq.13 软长度惩罚 | 只在接近截断时罚 |
| **`mcq_accuracy_reward`** | 选项**字母集合**完全匹配（仅评测用） | ⚠️ 选择题不能用 math_verify，见下 |

**⚠️ 选择题（MCQ）必须走独立判分**：AGIEval 高考的 gold 是**选项下标集合**（0-based），
而题面选项标注是 (A)(B)(C)(D)，模型最自然的作答是输出**字母**。
用 `math_verify` 拿字母比下标 → 实测**假阴性率 100%**。
`evaluate_math.py` 按 `question_type == "mcq"` 自动分派到 `mcq_set` 判分；
抽取器优先级：`\boxed{}` > 「答案：」标记 > `(X)` > 短的最后一行（有长度护栏防误抽）。


### ④ DPO 层（`training/dpo_training.py`）
| 组件 | 说明 |
|---|---|
| Trainer | TRL `DPOTrainer` + `DPOConfig` |
| reference | `ref_model=None`（PEFT 下复用底座，不额外占显存） |
| 偏好对来源 | **自产**：GRPO rollout 按 (accuracy, format) 排序取最好/最差 |

---

## 四、GRPO 的完整执行时序（最复杂的一块，排查必看）

```
[1] 采样
    RepeatSampler(mini_repeat_count=G)      ← 每个 prompt 重复 G 次
    → generation_batch 有 generation_batch_size 条
    → 含 generation_batch_size / G 道唯一题，同一题的 G 个副本【连续排列】

[2] _generate_and_score_completions(inputs)
    │
    ├─ AR3POTrainer：把 inputs 挂到 self._ar3po_inputs（供判对错用）
    │
    ├─ GRPOTrainer：取 prompts，调 self._generate(prompts)
    │      │
    │      └─ AR3POTrainer._generate          ★ 自适应 rollout 在这里
    │           ├─ 按「连续 G 个相同」切组
    │           ├─ Round 1：每题取 1 个副本 × k 条 → super()._generate()
    │           ├─ 调 accuracy_reward 判「这 k 条里有没有正确答案」
    │           │     有 → 该题退出，不再生成
    │           │     无 → 进入 Round 2
    │           ├─ Round 2：只对仍全错的题再生成 k 条
    │           ├─ 组装：每组拼到恰好 G 条
    │           │     不足的用 Round 1 的 k 条【整体复制】补齐
    │           │     复制件的 tool_mask 置 0（停梯度）
    │           └─ 返回 7 元组（prompt_ids, completion_ids, tool_mask,
    │                              completions, num_items_in_batch, logprobs, extra)
    │
    ├─ 算 rewards：3 个函数 × reward_weights
    ├─ 组内归一化 → advantages（reward.view(-1, G)）
    ├─ replay buffer：默认关闭（TRL 0.29.1 与 AR3PO 停梯度 mask 不兼容）
    └─ 返回 dict（含 advantages / completion_mask / tool_mask / num_items_in_batch）

[3] _prepare_inputs 切片 → 按 steps_per_generation 分成训练 batch

[4] _compute_loss：
    mask = completion_mask * tool_mask        ← tool_mask=0 的序列梯度为 0
    DAPO token-level loss → 反传
```

**核心不变式（写代码时反复用到的）**：
- `len(completion_ids) == len(prompts)`（`tools/trl_contract_probe.py` 实测确认）
- 有了上一条，上游的 `rewards.view(-1, G)` 才合法
- `tool_mask` 是**唯一**的「按序列停梯度」机制

---

## 五、TRL 自动记录的指标（排查的主要依据）

| 指标 | 含义 | 健康范围 |
|---|---|---|
| `frac_reward_zero_std` | **零方差组比例**（最重要） | 越低越好；>0.8 说明大部分组白采样 |
| `rewards/accuracy_reward/mean` | 准确率 | 应从低往高走 |
| `rewards/format_reward/mean` | 规范分 | 应快速上升后稳定 |
| `rewards/length_reward/mean` | 长度惩罚 | 应接近 0（接近 -1 = 大量截断） |
| `reward` / `reward_std` | 加权总分 | std 不能长期为 0 |
| `completions/mean_length` | 回复长度 | 单调暴涨 = 长度失控 |
| `completions/clipped_ratio` | 被截断比例 | >0.3 说明 `max_completion_length` 太小 |
| `entropy` | 策略熵 | 持续下降 = 熵坍缩 |
| `clip_ratio/region_mean` | 裁剪触发比例 | 长期 >0.3 说明更新过猛 |
| `grad_norm` | 梯度范数 | 突增 = 不稳 |

---

## 五之二、评测策略（五层，缺一层就会误判）

### 第 1 层 数据集分层

| 数据集 | 题量 | 角色 | 判分器 |
|---|---|---|---|
| `data/eval/dapo_holdout.jsonl` | 500 | **同分布留出**（与训练集同源）→ 看有没有过拟合 | `math_verify` |
| `data/eval/math500.jsonl` | 500 | 英文主指标 → **可与论文直接对比** | `math_verify` |
| `data/eval/aime2024.jsonl` | 30 | 高难度；题量太少，**必须 avg@64** 才有意义 | `math_verify` |
| `data/eval/agieval_gaokao.jsonl` | 351 | 中文迁移 | **`mcq_set`**（字母集合比较） |

### 第 2 层 指标（不能只看准确率）

| 指标 | 回答什么 |
|---|---|
| **avg@k** | 主指标（论文口径：MATH-500 avg@8、AIME24 avg@64） |
| pass@k | 能力上限（"模型到底会不会做这题"） |
| `mean_format` | 「规范性」这个奖励设计**到底有没有被学到** |
| **截断率** | 结果是否可信；>10% 说明准确率被系统性低估。脚本 >30% 时返回 exit 2 |
| 长度分布 | 有没有长度爆炸 |

### 第 3 层 训练期指标（见第五章）

### 第 4 层 对照实验（最容易漏，也最致命）

**只看最终模型的分数没有意义。** 必须：

```
① 三类模型都测：base → SFT → 最终（GRPO/DPO）
② 消融：  --ar3po_stages 1 vs 2        （AR3PO 到底有没有用）
         --format_weight 0 vs 0.3     （分档规范分是帮还是害）
③ 至少 2~3 个随机种子
```

没有 ①②③，任何数字都无法归因。

### 第 5 层 证明题（当前缺失）

5,000 条证明题**没有任何脚本消费**（不进 GRPO / DPO / 评测），需要人工或 LLM-judge 单独处理。

### 贪心 vs 采样的选择（重要）

| 目的 | 用什么 | 为什么 |
|---|---|---|
| 训练中监控、回归测试 | **贪心**（`--num_samples 1`，temperature=0） | 确定性、低方差 —— 否则分不清"改动有效"和"随机种子走运" |
| 能力测量、论文对比 | **avg@k + 采样** | 贪心锁死最高概率路径，thinking 模型容易陷入重复循环；**贪心不等于能力上限** |

⚠️ **k>1 时必须让 temperature>0**：否则 `num_return_sequences=k` 会返回 k 条完全相同的结果，
`avg@k` 退化成假的 `pass@1`。脚本已自动处理（k>1 → temperature=1.0）。

---

## 六、排查手册（按症状索引）

### 🔴 症状 1：reward 一直是 0 / 不涨

| 检查顺序 | 命令 / 看什么 | 判断标准 |
|---|---|---|
| 1. 奖励函数本身 | `python3 tools/test_math_rewards.py` | 必须全绿；「答对→给1分」要 100% |
| 2. 数据 schema | `head -1 data/grpo/en/math_dapo_train.jsonl` | 必须有 `question` + `answer` 两列 |
| 3. 训练日志 | `rewards/accuracy_reward/mean` | 若恒 0 → 见下 |
| 4. 模型是否在输出 `\boxed{}` | `--log_completions True` 后看 `self._logs` | 若不用 boxed → **SFT 没教好**，回去跑 SFT |
| 5. `frac_reward_zero_std` | 训练日志 | ≈1.0 → 题目对模型太难/太易，不是奖励坏了 |

**最常见根因**：`max_completion_length` 太小导致长 CoT 全被截断（默认曾配 256，现在 2048）。

### 🔴 症状 2：`frac_reward_zero_std` 接近 1

这是 **GRPO 的结构性问题**，不是 bug。三种处理：
1. 确认 `ar3po_stages=2` 已开（难题会拿双倍采样机会）
2. 当前保持 `replay_buffer_size=0`；TRL 0.29.1 的实验 replay 实现与 AR3PO mask 不兼容
3. 若仍无效 → **题目难度分布问题**：该题对当前模型要么全对要么全错。检查题库难度分布。

### 🔴 症状 3：reward 在涨但人工看效果变差（reward hacking）

| 检查 | 说明 |
|---|---|
| 看 `format_reward` 权重 | 是否 > 0.3×accuracy？太高会让模型刷格式分 |
| 看实际 completion | `--log_completions True`，抽样人工读 |
| 看长度 | `completions/mean_length` 是否暴涨 |

**这是「Secrets of RLHF」里的 pattern collapse**：模型学会用特定模式骗奖励分。

### 🔴 症状 4：回复越来越长

| 检查 | 期望 |
|---|---|
| `rewards/length_reward/mean` | 应该开始变负 |
| `--length_weight` | 是否为 1.0（若被改成 0 就没惩罚） |
| `--max_completion_length` | 2048 是否够；不够会大量截断 |
| `--mask_truncated_completions` | 必须 `True`，否则截断样本仍产生梯度 |

### 🔴 症状 5：格式分恒为 0

| 检查 | 说明 |
|---|---|
| 模型输出长什么样 | 是否完全不用 `\boxed{}` / 答案标记 |
| SFT 是否真的教会格式 | SFT 数据 `math_sft_demo.jsonl` 里答案都带 `\boxed{}` |
| Qwen3 thinking token 检测 | 训练启动日志里应有 `检测到 Qwen3 thinking 结束 token` 一行；若没有 → tokenizer 变了 |
| ⚠️ 是否注入了自定义 SYSTEM_PROMPT | **不应该有**。我们已删除它，让 Qwen3 用自己的原生格式 |

### 🔴 症状 6：OOM

按顺序试（见 README 第六章）：
```
① vllm_enable_sleep_mode=True（已开）
② num_generations 8 → 4
③ max_completion_length 2048 → 1024
④ vllm_gpu_memory_utilization 0.35 → 0.25
⑤ 换 Qwen3-4B 先验证流程
⑥ 上第 2 张卡（vllm_mode 改 server）
```
**先确认 `gradient_checkpointing=True`**（原代码在用 LoRA 时会强制关掉它，已修）。

### 🔴 症状 7：训练崩 / loss 变 NaN

| 检查 | 处理 |
|---|---|
| `grad_norm` 突增 | 降低 `learning_rate`（当前 1e-6） |
| `clip_ratio/region_mean` 长期高 | 说明更新过猛；检查 `epsilon`/`epsilon_high` |
| `entropy` 快速下降 | 熵坍缩；检查 `epsilon_high=0.28` 是否生效 |
| `beta` | 必须 0（RLVR 标准）；若被改成非 0，需重新引入 reference |

### 🔴 症状 8：报形状/契约相关的错

**根因**：TRL 版本变了，内部数据契约改了。

```
python3 tools/trl_contract_probe.py     # 重新探测契约
```
若输出 `len_completion_ids == len_prompts` 变成 False → `ar3po_trainer.py` 的 `_assemble()` 需要跟着改。
建议 `pip install trl==0.29.1` 锁版本。

### 🔴 症状 9：AR3PO 好像没生效

| 检查 | 期望 |
|---|---|
| `num_generations % ar3po_stages == 0` | 8 % 2 = 0 ✓；不满足会在 `__init__` 直接报错 |
| 训练启动日志 | 应有 `[rewards] ...`；默认 `replay_buffer_size=0` |
| 跑 `python3 tools/ar3po_smoke_test.py` | 应显示「分阶段生成生效」+「补齐样本停梯度」 |
| `model.training` | AR3PO 只在训练模式生效，eval 会自动回退 |
| 是否带 tools | `self.tools` 非空时**自动回退**到原始实现（安全设计） |

### 🔴 症状 10：DPO 没效果

| 检查 | 说明 |
|---|---|
| 导出时的过滤比例 | `export_dpo_pairs.py` 会打印「过滤掉 N 条无信息量的」；若过滤掉 90%+，说明 rollout 区分度太低 |
| `chosen`/`rejected` 是否真的不同 | `head -1 data/reward/math_dpo_pairs.jsonl` 人工看 |
| `ref_model` | PEFT 下是 `None`（复用底座），这是对的 |
| `learning_rate` | 5e-7，比 SFT 低很多；太高会崩 |

### 🔴 症状 11：中文（AGIEval 高考）评测分数**恒为 0**

**这是已修复的历史问题，但如果换数据集后复发，按这个顺序查：**

| 检查 | 说明 |
|---|---|
| 参考实现 | 本题库已知的坑：AGIEval 的 gold 是**选项下标**（0-based，如 `[0 1 3]`），而模型输出的是**选项字母**（`\boxed{C}`）。用 `math_verify` 拿字母比下标 → **假阴性 100%** |
| 数据格式 | `head -1 data/eval/agieval_gaokao.jsonl` 应看到 `"answer": "C"`（字母）而不是 `"[3]"`（下标），并且 `"question_type": "mcq"` |
| 判分器分派 | 评测输出应有 `判分器 = mcq_set（选择题：字母集合完全匹配）`；若显示 `math_verify` 说明 `question_type` 没写对 |
| 明细诊断 | 加 `--output xxx.jsonl`，看 `pred_letters` 字段——模型到底抽出了哪些字母 |
| 回归测试 | `python3 tools/test_math_rewards.py` 的 4b 节会打印「旧路径 0/351 vs 新路径 351/351」 |

**其他 MCQ 类数据集接入注意**：必须把答案转成字母集合并设 `question_type="mcq"`，
多选题要求**完全匹配**（漏选/多选都算错）。

### 🔴 症状 12：想中途开/关 vLLM（比如显存不够想关掉救急）

| 场景 | 能否改 `use_vllm` |
|---|---|
| **同一个 run 的训练过程中** | ❌ **不能**。`GRPOTrainer.__init__:531` 的 `self.use_vllm = args.use_vllm` 只赋值一次，之后所有分支都用 `self.use_vllm`；vLLM 引擎也在 `__init__:709` 创建。而且**全文没有一致性校验**（搜不到 `if self.args.use_vllm != self.use_vllm`），直接改 `training_args.use_vllm` 会**静默不一致** —— 日志说变了，实际没生效。 |
| **停下来，从 checkpoint 恢复** | ✅ **能**（已实测：改 `max_completion_length` 后 resume，`global_step` 正确接续）。`use_vllm` 不在 checkpoint 里，也没有运行期校验。 |
| 新起一个 run 从头训 | ✅ 能（就是个命令行开关） |

**⚠️ 但别把它当省显存的中途手段**，两个原因：

1. **来不及**：显存不够的表现是**第一次生成就 OOM**。那时一步都没训完，
   没有 checkpoint 可恢复，切不了。
2. **会污染 A/B 对比**：TRL 源码注释（`grpo_trainer.py:1787`）明确写着
   *"the distribution mismatch between vLLM and the training model **can be large and harm the training**"*，
   所以开 vLLM 时 TRL 会额外算 `old_per_token_logps` 做 **importance sampling 修正**。
   ⇒ 开/关 vLLM 的**训练动态不同**，两次 run 的分数不能直接比较。

**正确做法**：显存不够就在**开始训练之前**决定（`--use_vllm False`），
接受 rollout 慢 3~10 倍的代价。要对比 vLLM 开关的影响，就固定其他条件做两组 run。

**验证脚本**：`python3 tools/test_resume_flexibility.py`（极小模型实测 resume + 换参数）。

---

## 七、验证工具清单（改动后必跑）

| 脚本 | 验证什么 | 何时跑 |
|---|---|---|
| `tools/test_math_rewards.py` | 奖励函数在**真实数据**上的正确性 + 负向对照 + MCQ 判分 | **每次改奖励后必跑** |
| `tools/validate_run_config.py` | 含 **AST 静态检查**：`trainer_cls(...)` 的 kwarg 是否都是合法构造参数 | 每次改 trainer 构造后 |
| `tools/test_resume_flexibility.py` | resume + 换构造期参数是否可行 | 改动训练循环后 |
| `tools/memory_planner.py` | **显存硬地板** = 权重×份数 + KV（**精确算术**）。只判定"是否数学上不可能"，**不猜激活** | 换模型/换卡前 |
| `tools/ar3po_smoke_test.py` | AR3PO 的形状契约 / 停梯度 / 生成量下降 | 每次改 `ar3po_trainer.py` 后 |
| `tools/validate_run_config.py` | `run_grpo.sh` 的真实配置能否跑通 | 每次改启动脚本后 |

### 显存结论（实测算术，可直接引用）

Qwen3-8B + vLLM colocate 的**硬地板 = 32.75 GiB**（权重 15.25×2 + KV 2.25）：

| 显卡 | 保守可用 | 地板占比 | 结论 |
|---|---|---|---|
| A100-40G | 39.0 GiB | 84.0% | 剩 5.2 GiB 放激活 —— **值得一试，必须实测** |
| 5090-32G | 31.0 GiB | **105.7%** | ❌ **确定不可能**（不用装环境/下模型就能排除） |
| 4090-24G | 23.5 GiB | 139.4% | ❌ 确定不可能 |

**Qwen3-8B 关掉 vLLM 后地板降到 15.25 GiB**（1 份权重）→ 32G 卡上反而可行，代价是 rollout 慢 3~10×。

⚠️ 另注意：**Qwen3-8B 与 Qwen3-4B 的 KV/token 完全相同**（144 KB，都是 36 层 × 8 KV头 × 128 维）
—— 换 4B 只省权重、不省 KV。而 5090 是 Blackwell **sm_120**，torch/vLLM/flash-attn 的轮子支持是独立风险。

| `tools/trl_contract_probe.py` | TRL 内部数据契约 | 升级 TRL 后 |
| `python3 -c "import ast;ast.parse(open('f').read())"` | 语法 | 随手 |

**这四个脚本都用「本地从零构造的极小模型」**，不需要 GPU、不需要 HF Hub，秒级返回。

### 评测工具

```bash
# 快速冒烟（贪心，快，确定性）
python3 tools/evaluate_math.py --model <path> --data data/eval/math500.jsonl \
    --num_samples 1 --batch_size 8 --max_new_tokens 2048

# 对齐论文口径（avg@8）
python3 tools/evaluate_math.py --model <path> --data data/eval/math500.jsonl \
    --num_samples 8 --batch_size 8 --output outputs/eval_math500.jsonl

# 一键跑四个数据集（含中文高考）
bash scripts/run_eval_quantize.sh <模型路径> 8
```

`--output` 的明细里含 `judge` / `pred_letters` / `n_truncated` / 每条 completion，
是排查评测问题时最有用的东西。

---

## 八、关键参数速查（改什么影响什么）

| 参数 | 当前值 | 作用 | 改动风险 |
|---|---|---|---|
| `num_generations` | 8 | 组内样本数；显存的主要驱动 | ↓ 省显存但方差变小 |
| `ar3po_stages` | 2 | 分阶段数；k = G/S | 1 = 退回标准 GRPO（A/B 用） |
| `replay_buffer_size` | **0** | TRL 0.29.1 下必须关闭，避免复制样本错误地产生梯度 | 高 |
| `scale_rewards` | **none** | ⚠️ **AR3PO 必须 none**；改 group 会让补齐组 advantage 幅度放大 | 高 |
| `beta` | 0.0 | KL 系数；0 = 无 reference | 非 0 会多一份前向 |
| `loss_type` | dapo | 损失形式 | 改 grpo = 原始 GRPO |
| `epsilon` / `epsilon_high` | 0.2 / 0.28 | 裁剪范围（clip-higher） | 低 |
| `max_completion_length` | 2048 | 生成长度上限 | 太小 → 长 CoT 全截断 |
| `format_weight` | 0.3 | 规范分权重 | >0.3×acc 会诱发刷分 |
| `length_weight` | 1.0 | 长度惩罚权重 | 0 = 关闭惩罚 |
| `learning_rate` | 1e-6 | | 高 → 崩 |
| `warmup_steps` | 0 | AR3PO 论文明确不用 warmup | — |

---

## 九、已知边界（不要当成 bug）

1. **`format_reward` 的分档设计是工程判断**。文献里 DAPO/R1-Zero 用的是二值。建议做一次 A/B（`--format_weight 0` vs `0.3`），对比 `frac_reward_zero_std` 和最终分数。
2. **AR3PO 的精度收益尚未在真实 7B/8B 上验证**。机制已验证，但「追平/超过 DAPO」必须在你的真实数据上复现。
3. **中文训练数据是英译中**（NuminaMath 的 cn_k12 实测汉字占比 0.0%）。原生中文题需自建 OCR 流水线。
4. **证明题不做 RL**。`numina_proof.jsonl` 只是评测集；`verifiable=false` 标记了这个边界。
5. **`trl.experimental` 是实验性 API**，升级 TRL 可能失效 → 建议 pin `trl==0.29.1`。
6. **`--train_file_dir` 会加载目录下所有 jsonl**，英文/中文/证明题混在一起时要注意隔离。

---

## 十、一句话总览

> **用规则奖励（整体给分 + 分层规范 + 长度惩罚）在 Qwen3-8B 上做无 critic 的 GRPO，
> 用 AR3PO 的自适应 rollout 解决「组内奖励全同 → 零梯度」这个结构性问题，
> 再用 rollout 自产的偏好对做 DPO 蒸馏。全程只驻留一份模型 + 一份 vLLM，48GB 单卡可跑。**
