# 数学推理项目实验与量化验收方案

> 目标：用一套可复现、可归因、可汇报的实验回答三个问题：
> 1. 最终模型是否真的提升了数学推理能力？
> 2. 提升是否来自本项目的 AR3PO 与奖励设计，而不是训练步数或随机波动？
> 3. AR3PO 是否在相近效果下减少了生成成本？

## 1. 项目主张与可验收指标

| 项目要解决的问题 | 方法 | 主指标 | 辅助指标 | 建议验收线 |
|---|---|---|---|---|
| GRPO 全对/全错组无梯度，采样浪费 | AR3PO adaptive rollout | `strict_avg@k`、真实 rollout 数 | 二阶段触发率、非零方差组比例、有效组/秒 | 相对标准 GRPO：准确率差值 95% CI 下界高于预注册的非劣界，真实生成数降低 ≥20%；或生成数相近时准确率显著提升 |
| 二值格式奖励很快退化为常数 | 0~1 分层格式奖励 | `mean_format`、`frac_reward_zero_std` | accuracy reward、总 reward、格式/准确率相关性 | 相对二值/无格式奖励：格式分提升，且 accuracy 不下降；格式权重不能主导总奖励 |
| 单次输出不稳定、答案不规范 | SFT → GRPO → DPO | 贪心 `strict_acc`、采样 `strict_avg@k` | `strict_pass@k`、答案长度、截断率 | 最终模型在主测试集优于 SFT；DPO 后贪心稳定性提升且 avg@k 不明显下降 |
| 英文训练可能无法迁移到中文 | 中英文分阶段训练 | MATH-500 与 AGIEval 高考分别报告 | DAPO holdout、AIME24 | 英文主指标提升，中文指标不低于 SFT；若做中文阶段，应单独报告增益 |
| 48GB 单卡训练成本高 | LoRA、无 critic、无 reference、vLLM colocate | 峰值显存、训练 wall time | tokens/s、有效正确样本/GPU-hour | 单卡可完成；所有方法用同一硬件和预算比较 |

验收时不要把多个目标压成一个“总分”。能力、效率、稳定性和格式质量分别报告，最后再给结论。

## 2. 实验原则

### 2.1 只改变一个因素

所有 GRPO/AR3PO 对照固定以下条件：

- 同一 SFT checkpoint、训练数据顺序、训练样本数、optimizer steps。
- 同一 `num_generations` 上限、采样温度、`top_p`、`top_k` 和最大生成长度。
- 同一 DAPO loss、LoRA 配置、batch、学习率和硬件。
- `scale_rewards=none`、`replay_buffer_size=0` 固定不变。
- 评测使用完全相同的 prompt/chat template、解码参数和判分器。

AR3PO 的目标之一是节省生成，因此同时采用两种公平口径：

1. **等训练步数**：比较最终能力与实际生成成本，回答“同样更新次数，是否更省”。
2. **等生成预算**：当某个方法达到同样累计 completion tokens/rollout 数时比较能力，回答“同样成本，是否更强”。

只做其中一种会让效率或效果的结论偏向某一方。

### 2.2 数据隔离

- SFT：`data/sft/`，只用于冷启动。
- RL 训练：`data/grpo/en/`；中文阶段作为独立实验，不与英文主实验混写。
- DPO：只使用训练集 prompt 的 rollout 构造偏好对。
- 测试：`data/eval/` 四个集合禁止参与调参、筛 checkpoint 或构造 DPO 对。
- 从训练集固定划出一个 development set，用于选 checkpoint 和调超参；正式 test 每个模型只在方案冻结后跑一次。
- 保存划分文件或样本 ID/hash，所有 run 共用同一划分。

## 3. 分阶段流程

### 阶段 0：链路与判分器验收（CPU，可频繁运行）

每次改数据、奖励或 trainer 后执行：

```bash
for f in data/grpo/en/*.jsonl data/grpo/zh/*.jsonl data/sft/*.jsonl data/eval/*.jsonl; do
  python3 tools/validate_jsonl.py --file_path "$f"
done
python3 tools/test_math_rewards.py
python3 tools/trl_contract_probe.py
python3 tools/ar3po_smoke_test.py
python3 tools/validate_run_config.py
python3 -m pytest -q
```

通过标准：全部退出码为 0；正确答案奖励为 1；MCQ 使用 `mcq_set`；补齐样本 mask 为 0；无 schema 混装。

### 阶段 1：固定评测基线

对以下 checkpoint 做一次贪心评测（`k=1, temperature=0`）：

1. Qwen3-8B 起始模型。
2. SFT merged 模型。
3. 后续每个候选训练 checkpoint。

贪心评测用于快速回归和选择 checkpoint，不作为最终论文口径。先确认所有模型截断率 ≤10%；超过 10% 时增加 `max_new_tokens` 后重跑，不能直接比较。

### 阶段 2：小规模 pilot（排错，不下效果结论）

使用 50~200 个训练样本、1 个种子，分别跑 GRPO 和 AR3PO。目标仅为确认：

- 显存不溢出，训练 loss/reward 为有限值。
- accuracy reward 不长期为 0，`frac_reward_zero_std` 不是由判分 bug 导致。
- AR3PO 的第二阶段只对第一阶段全错题触发。
- checkpoint、日志、merged 模型路径前后一致。
- 吞吐和 wall time 能被记录。

当前 `scripts/run_grpo.sh` 中的 `--train_samples 50` 属于此阶段，不能据此宣称算法有效。

### 阶段 3：核心消融实验

最小可交付矩阵如下。所有行至少运行 3 个随机种子；资源紧张时可先用 4B/数据子集筛选，再只对保留配置跑 8B 正式实验。

| ID | 起点 | `ar3po_stages` | format reward | length reward | 目的 |
|---|---|---:|---:|---:|---|
| B0 | SFT | - | - | - | RL 前基线 |
| B1 | SFT | 1 | 0.0 | 1.0 | 标准 GRPO 基线 |
| A1 | SFT | 2 | 0.0 | 1.0 | 只验证 adaptive rollout |
| A2 | SFT | 2 | 0.3（分层） | 1.0 | 项目完整 GRPO 配置 |
| A3 | A2 merged | - | - | - | 在相同偏好对规则下继续 DPO |

增强版消融增加两行：

| ID | 配置 | 回答的问题 |
|---|---|---|
| F1 | stages=2，二值 format reward | 分层格式奖励是否优于常见二值奖励 |
| L0 | stages=2，format=0.3，length=0 | 长度 shaping 是否减少截断/长度爆炸且不损害准确率 |

不要把 DPO 只接在 A2 后就声称其有效。要验证 DPO 的独立贡献，比较同一个 A2 checkpoint 的“DPO 前 vs DPO 后”，且偏好数据、训练步数固定。

### 阶段 4：正式能力评测

冻结配置后，逐模型运行：

```bash
# 快速、确定性回归
bash scripts/run_eval_quantize.sh <MODEL_PATH> 1

# MATH-500 / DAPO holdout / AGIEval：正式 avg@8
python3 tools/evaluate_math.py --model <MODEL_PATH> \
  --data data/eval/math500.jsonl --num_samples 8 \
  --temperature 0.6 --top_p 0.95 --top_k 20 \
  --max_new_tokens 2048 --output <RUN_DIR>/math500.jsonl

# AIME24 只有 30 题，正式使用 avg@64
python3 tools/evaluate_math.py --model <MODEL_PATH> \
  --data data/eval/aime2024.jsonl --num_samples 64 \
  --temperature 0.6 --top_p 0.95 --top_k 20 \
  --max_new_tokens 4096 --output <RUN_DIR>/aime2024.jsonl
```

注意：当前 `scripts/run_eval_quantize.sh <model> 8` 会让 AIME 也使用 k=8；正式结果需按上面的 AIME `k=64` 单独运行。所有模型的采样配置必须完全一致。

### 阶段 5：统计分析与结论冻结

- 准确率：以“题目”为重采样单位做 paired bootstrap（建议 10,000 次），报告两模型差值、95% CI 和胜/平/负题数。
- 多种子训练：报告 `mean ± std`；不要只挑最好 seed。
- AIME24：题目只有 30 道，必须同时给原始命中数、avg@64 和区间，不只给百分比。
- 多数据集不求简单平均；MATH-500 为英文主指标，AGIEval 为中文主指标，AIME 为高难度诊断，DAPO holdout 为同分布诊断。
- 如果 95% CI 跨 0，应写“未观察到稳定提升”，不能写“显著提升”。
- 若要声称“效果不降但更省”，需预先注册非劣界（建议主指标绝对下降不超过 0.5 个百分点）；差值 CI 下界高于 `-0.5pp` 才能判为非劣，不能用“CI 跨 0”替代非劣检验。

## 4. 每个 run 必须记录的字段

### 4.1 配置与可复现信息

```text
run_id, git_commit, dirty_worktree, date, host, gpu_name
base_checkpoint, dataset_hash, seed
train_samples, epochs, optimizer_steps, effective_batch_size
learning_rate, max_completion_length, num_generations, ar3po_stages
temperature, top_p, top_k
format_weight, length_weight, loss_type, epsilon, epsilon_high
```

### 4.2 训练能力与稳定性

从 TensorBoard/Trainer 日志保存：

- `rewards/accuracy_reward/mean`
- `rewards/format_reward/mean`
- `rewards/length_reward/mean`
- `reward`、`reward_std`、`frac_reward_zero_std`
- `completion_length/mean`、`completion_length/max`
- `clip_ratio/region_mean`、`entropy`、`grad_norm`、`loss`

### 4.3 AR3PO 效率指标（正式实验前必须补齐采集）

当前实现尚未完整暴露以下指标；没有它们就不能证明“采样效率提升”：

- `rollout/real_completions`：排除复制补齐后的真实生成条数。
- `rollout/real_completion_tokens`：真实生成 token 总数，作为首选成本口径。
- `ar3po/stage2_prompt_ratio`：进入第二阶段的 prompt 比例。
- `ar3po/mean_real_completions_per_prompt`。
- `ar3po/early_exit_ratio`。
- `throughput/real_completion_tokens_per_second`。
- 每 run 的 wall time、峰值 GPU 显存、GPU-hours。

推荐效率派生指标：

```text
sampling_saving = 1 - AR3PO_real_completions / GRPO_real_completions
token_saving    = 1 - AR3PO_real_tokens / GRPO_real_tokens
effective_groups_per_gpu_hour = nonzero_variance_groups / GPU_hours
correct_samples_per_gpu_hour  = correct_real_completions / GPU_hours
```

真实 token 数比 completion 条数更可靠，因为各方法的输出长度可能不同。

## 5. 结果表模板

### 5.1 能力主表

| 模型/方法 | seed | MATH-500 strict avg@8 | DAPO strict avg@8 | AGIEval strict avg@8 | AIME24 strict avg@64 | mean format | truncation |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | - | | | | | | |
| SFT | - | | | | | | |
| GRPO | 1/2/3 | | | | | | |
| AR3PO | 1/2/3 | | | | | | |
| AR3PO + rewards | 1/2/3 | | | | | | |
| + DPO | 1/2/3 | | | | | | |

最终报告把三个种子聚合为 `mean ± std`，并在 AR3PO 与 GRPO 的差值旁报告 paired bootstrap 95% CI。

### 5.2 效率主表

| 方法 | 实际 responses/prompt | 实际 completion tokens | stage2 比例 | 非零方差组比例 | wall time | 峰值显存 | strict MATH-500 / GPU-hour |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO | | | - | | | | |
| AR3PO | | | | | | | |

### 5.3 阶段增益表

| 对比 | 要归因的模块 | Δ主指标 | 95% CI | 成本变化 | 结论 |
|---|---|---:|---:|---:|---|
| SFT − Base | 冷启动 | | | | |
| AR3PO − GRPO | 自适应 rollout | | | | |
| 分层格式 − 无格式 | 奖励设计 | | | | |
| DPO − DPO前 | 偏好优化 | | | | |

## 6. 决策规则

按以下顺序判断，避免“只挑有利指标”：

1. **结果有效性**：截断率 ≤10%，判分器正确，配置一致；否则结果作废并重跑。
2. **能力目标**：A2 相对 B1 的 MATH-500/主任务差值 CI 不应稳定为负。
3. **效率目标**：A2 的真实生成 token 至少降低 20%，或等 token 预算下主指标提升。
4. **奖励安全性**：format 上升时 accuracy 不能下降，且人工抽检不能出现刷 `boxed`/模板的 reward hacking。
5. **完整流水线**：DPO 后贪心正确率或跨 seed 方差改善，同时 avg@k 不明显下降。
6. **泛化目标**：holdout 提升但 MATH-500 不提升，判为同分布拟合，不能宣称泛化能力提升。

项目最终可以用一句可审计的话总结：

> 在相同 SFT 起点和训练配置下，AR3PO 相比标准 GRPO 将真实 rollout token 降低 X%（或在等 token 预算下将 MATH-500 strict avg@8 提升 Y 个百分点，paired bootstrap 95% CI [L, U]）；完整 SFT→AR3PO→DPO 流水线相对 Base 在英文、中文和高难度集合上的阶段增益分别为……，峰值显存为……GB。

## 7. 当前脚本在正式实验前需确认的事项

- `scripts/run_grpo.sh` 当前是 `train_samples=50`、`num_generations=4`、`max_completion_length=1024`，与文档中的正式 8B 口径不同，应明确标记为 pilot 或另建正式配置。
- 该脚本的训练 `output_dir`、merge 时读取的 `lora_model`、最终 merged 目录命名目前不完全一致，正式运行前必须统一，否则可能合并错误 checkpoint 或直接失败。
- `scripts/run_dpo.sh` 默认寻找 `outputs-grpo-qwen3-8b-ar3po-merged`，也需与 GRPO 正式输出路径一致。
- 正式对照必须支持显式 `seed`、独立 `run_id` 和不可覆盖的输出目录。
- 测试集的逐题 JSONL 是统计分析的原始证据，不能只保留终端汇总。
