# 数学推理解题模型（GRPO + AR3PO / DPO）

基于 [MedicalGPT](https://github.com/shibing624/MedicalGPT) 的训练框架改造而来，**已剥离全部医疗数据与无关算法**，聚焦于**用强化学习提升数学解题能力**。

> 原项目文档已归档为 [docs/README_original_medical.md](docs/README_original_medical.md)。

---

## 一、这是什么

一条 **SFT → GRPO(+AR3PO) → DPO** 的数学推理训练流水线，跑在 **Qwen3-8B + 单卡 48GB** 上。

| 层次 | 采用方案 | 说明 |
|---|---|---|
| **奖励设计** | **整体给分 + 分层规范分 + 长度 shaping** | 规则函数，**不需要奖励模型**（省显存） |
| **损失层** | DAPO 式 token-level + clip-higher | TRL 内置（`loss_type=dapo`、`epsilon_high`） |
| **采样层** | **AR3PO** 自适应 rollout | 本仓库自行实现；TRL 0.29.1 replay buffer 因 mask 不兼容而关闭 |
| **偏好层** | DPO，偏好对从 GRPO rollout **自产** | 零额外标注成本 |

**核心算法论文**：[AR3PO — Improving Sampling Efficiency in RLVR through Adaptive Rollout and Response Reuse](https://arxiv.org/abs/2509.25808)（2025-09）

选它的理由：它是候选里**唯一同时改善"效果"和"吞吐"**的方案——解决 GRPO 的结构性缺陷（组内奖励全同 → advantage 全 0 → 该组白采样），生成成本比 DAPO 低 **4.2×**。

---

## 二、快速开始

运行环境要求：**Python 3.10+、Linux、NVIDIA CUDA GPU**。项目固定的
`transformers==5.17.0` 不支持 Python 3.9；默认 GRPO 配置还会使用 vLLM，
因此不能直接在 macOS 上进行正式训练。

```bash
python3 --version  # 必须 >= 3.10
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt  # 运行测试时需要
```

安装完成后先执行 `python3 -m pip check`。如果只在无 CUDA 的机器上做 CPU
逻辑测试，可把 GRPO 参数设为 `--use_vllm False`；正式训练仍建议使用脚本中的
Linux/CUDA + vLLM 配置。

```bash
# 0) 下载并转换数据集（首次执行，约 800MB，走 hf-mirror 镜像）
python3 tools/download_math_datasets.py
python3 tools/build_math_dataset.py

# 1) 验证奖励链路（强烈建议每次改奖励后都跑）
python3 tools/test_math_rewards.py

# 2) SFT 冷启动（GRPO 不能裸跑，见下）
bash scripts/run_sft.sh

# 3) GRPO + AR3PO 主训练
bash scripts/run_grpo.sh

# 4) 用合并后的 GRPO 模型采样真实偏好对 → DPO
python3 tools/export_dpo_pairs.py --gen \
  --model outputs-grpo-qwen3-8b-ar3po-merged \
  --data data/grpo/zh/math_zh_train.jsonl --limit 2000
bash scripts/run_dpo.sh
```

**开发期不需要 GPU 也能验证**（本地从零构造极小模型）：

```bash
python3 tools/trl_contract_probe.py     # 探测 TRL 内部数据契约
python3 tools/ar3po_smoke_test.py       # 验证 AR3PO 补齐/停梯度机制
python3 tools/validate_run_config.py    # 用 run_grpo.sh 的真实配置跑端到端
```

训练后用同一评测集分别测 base 与最终 merged 模型，只有生成式符号准确率提升才算实现目标：

```bash
bash scripts/run_eval_quantize.sh Qwen/Qwen3-8B
bash scripts/run_eval_quantize.sh outputs-dpo-qwen3-8b-math-merged
```

---

## 三、数据

| 文件 | 行数 | 用途 |
|---|---|---|
| `data/grpo/en/math_dapo_train.jsonl` | 12,000 | **阶段一**：DAPO-Math-17k 去重后（真实 17,186 道唯一题），**英文**，纯数值答案 |
| `data/grpo/zh/math_zh_train.jsonl` | 20,000 | **阶段二**：MathInstruct-Chinese（英译中），中文迁移 |
| `data/grpo/proof/numina_proof.jsonl` | 5,000 | 证明题，标了 `verifiable=false`，**默认不进 GRPO** |
| `data/sft/math_sft_demo.jsonl` | 2,000 | NuminaMath 真实完整解题过程，用于 SFT 冷启动 |
| `data/eval/*.jsonl` | 500/30/351/500 | MATH-500 / AIME2024 / AGIEval高考 / DAPO holdout |
| `data/reward/math_dpo_pairs.jsonl` | 动态 | DPO 偏好对（自产） |

### ⚠️ 关于中文数据的重要发现

我实测验证过：**`AI-MO/NuminaMath-1.5` 的 `cn_k12`/`cn_contest` 子集（298,763 行）虽然是中文来源，但题面已被翻译成英文**——抽样 300 条，汉字占比 >10% 的比例为 **0.0%**。

所以中文训练数据只能用 `MathInstruct-Chinese`（也是英译中）。**想要原生中文高中/考研题 + 证明题的高质量训练集，公开渠道基本没有，必须自建**（OCR + LLM 抽取流水线）。

`tools/build_math_dataset.py` 里内置了这个检测，每次构建都会打印中文占比警告。

### 评测数据的判分器（容易踩坑）

| 数据集 | 判分器 | 说明 |
|---|---|---|
| MATH-500 / AIME24 / DAPO holdout | `math_verify` 符号等价 | gold 包进 `\boxed{}` 后比较 |
| **AGIEval 高考** | **`mcq_set` 字母集合比较** | ⚠️ 它的 gold 是**选项下标**（0-based，如 `[0 1 3]`），模型自然输出**选项字母**。用 math_verify 拿字母比下标 → **假阴性 100%**（实测 0/351）|

`evaluate_math.py` 按 `question_type == "mcq"` 自动分派；构建时已把下标转成字母并把
原始下标存在 `answer_indices` 里。回归测试：`python3 tools/test_math_rewards.py` 的 4b 节
会打印「旧路径 0/351 vs 新路径 351/351」。

---

## 四、奖励逻辑（核心设计）

```
总分 R = 1.0 · accuracy  +  0.3 · format  +  1.0 · length
```

| 奖励 | 粒度 | 说明 |
|---|---|---|
| `accuracy_reward` | **整体给分** | `math_verify` 符号等价判定 → 0/1，**不可 hack** |
| `format_reward` | **分层给分** | 0~1 连续值（不是二值！） |
| `length_reward` | 标量 | DAPO Eq.13 软长度惩罚 |

### 为什么不做"步骤给分（PRM）"

三条论文级证据：

1. [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)：过程监督确实更强，但代价是 **PRM800K（80 万条人工步级标注）**。
2. [Qwen 团队 PRM 复盘](https://arxiv.org/abs/2501.07301)：**MC 自动合成的步级数据比 LLM-as-judge/人工都差**；且发现现有 PRM 的最低分**大量集中在最终答案步** → PRM 在优化中漂移回了结果评估。
3. [PRIME](https://arxiv.org/abs/2502.01456)：明确列出过程奖励在 online RL 的三个死结（难定义 / 在线更新不可扩展 / 显存成本），并指出 DeepSeek-R1 **因同样原因没把 PRM 用进大规模 RL**。

加上 48GB 显存约束（PRM 要 +16GB，或 10× MC 采样），**步骤给分不可行**。

### 三条设计铁律

| # | 铁律 | 原因 |
|---|---|---|
| 1 | `format_reward` **必须分档** | 二值格式奖励在模型学会格式后恒为 1 → 退化成常数，不再提供梯度；分档能持续提供组内方差 |
| 2 | `accuracy` 必须**独立**且放 `reward_funcs[0]` | `AR3POTrainer` 用 `ar3po_correctness_reward_idx=0` 单独靠它判定「全错」 |
| 3 | `w_format ≤ 0.3×w_accuracy` | 难题上 accuracy 恒为 0 时，格式权重过高会让模型**转去刷格式分**（reward hacking 入口） |

### 两个实测踩出来的坑（已修）

**坑 1：gold 也必须包进 `\boxed{}` 再 parse。**
`math_verify` 的 `LatexExtractionConfig` 默认需要 anchor（`\boxed{}` 或 "answer is"）。直接 `parse("\sqrt{51}")` 会抽出空值或只抓一个数字 → **模型答对了却被判 0 分**。
实测（MATH-500 + DAPO 共 800 条）自洽率：直接 parse gold **86.5%** → 包进 boxed **100.0%**。

**坑 2：不能用正则检查 ` thinking` 标签。**
TRL 用 `skip_special_tokens=True` 解码，Qwen3 的 ` thinking`/`<｜end▁of▁thinking｜>` 是 special token，**文本里根本不出现**。必须用 `completion_ids` 里的 token id 判断（见 `make_format_reward(think_end_token_id=...)`）。

---

## 五、AR3PO 实现

论文的 **adaptive rollout**（自适应 rollout）：分 S 阶段生成，有正确答案的题提前退出 → 难题拿更多预算，简单题省算力。

**实现要点**（`training/ar3po_trainer.py`）：

```
Round 1：每道题只生成 k = G/S 条
Round 2：只对「仍无正确答案」的题再生成 k 条
组装：  每组拼到恰好 G 条；不足的用第 1 轮的 k 条整体复制补齐，
        复制件的 tool_mask 置 0（停梯度）
```

**为什么"整体复制"能兼容 TRL 的固定形状约束**（已数值验证）：

- **mean 精确不变** ✓
- **std 会变**：`std_2k / std_k = sqrt(2(k-1)/(2k-1))`（只依赖 k 的常数）
- ⇒ `scale_rewards="none"` 时 advantage **完全精确**；用默认的 `group` 时幅度会被统一放大 `sqrt((2k-1)/(2(k-1)))`（k=2 → ×1.22，方向不变）

**所以 `scale_rewards` 必须设 `none`**（正好也是 Dr. GRPO 的主张）。`AR3POTrainer` 检测到 `group` 会自动打印警告。

**response reuse（论文的另一半）当前没有启用。** TRL 0.29.1 的实验 replay trainer 会丢掉 AR3PO 用于复制样本停梯度的 `tool_mask`，而且空 buffer 有崩溃边界；默认 `replay_buffer_size=0`。等上游提供兼容实现后再恢复，避免静默训练错误目标。

---

## 六、48GB 显存要点

| 项 | 占用 |
|---|---|
| Qwen3-8B bf16 权重 | 16.4 GB |
| vLLM colocate 权重 | 16.4 GB |
| KV cache（8×2048 tok） | 2.4 GB |
| 剩余给激活 + LoRA | ~12.8 GB |

**必须开的 4 个开关**（`references/run_grpo.sh` 已配好）：

| 参数 | 值 | 原因 |
|---|---|---|
| `gradient_checkpointing` | `True` | ⚠️ 原脚本在用 LoRA 时**强制关掉了它**，是 8B 必 OOM 的元凶，已修正 |
| `beta` | `0.0` | RLVR 标准做法（DAPO/AR3PO/PRIME 全为 0），省掉 reference 前向 |
| `use_vllm` | `True` + `colocate` | 吞吐差数倍 |
| `max_completion_length` | `2048` | 数学长 CoT 需要（原值 512 远远不够） |

**OOM 降级阶梯**：`num_generations 8→4` → `max_completion_length 2048→1024` → `vllm_gpu_memory_utilization 0.35→0.25` → 换 Qwen3-4B 先验证流程 → 最后才上第 2 张卡。

> 💡 强烈建议先用 **Qwen3-4B** 把"数据→AR3PO→评估"整条流水线跑通，再切 8B。

---

## 七、Qwen3 格式适配

Qwen3 的原生思考标记是 ` thinking ...  response`（**不是** DeepSeek-R1 的 ` thinking`，更不是本仓库原先误用的 `<think>`），且内置 thinking 开关（`/think`、`/no_think`）。

**因此本项目不注入自定义 SYSTEM_PROMPT**，让 tokenizer 的官方 chat template 自行决定格式——不要要求模型放弃它预训练好的格式。

（原 `grpo_training.py` 的 SYSTEM_PROMPT 要求 `<think></think><answer></answer>`，与 Qwen3 原生行为直接冲突，已删除。）

---

## 八、目录结构

```
training/
  grpo_training.py        # GRPO + AR3PO 主训练
  ar3po_trainer.py        # ⭐ AR3PO 自适应 rollout 实现
  math_rewards.py         # ⭐ 奖励函数（整体 + 分层规范 + 长度）
  dpo_training.py         # DPO
  supervised_finetuning.py# SFT
  template.py, tool_utils.py
scripts/
  run_sft.sh / run_grpo.sh / run_dpo.sh
  run_quant.sh / run_eval_quantize.sh / vllm_deployment.sh
tools/
  download_math_datasets.py   # 走 hf-mirror 下载数据集
  build_math_dataset.py       # 转换成仓库 schema
  test_math_rewards.py        # ⭐ 奖励单元测试（真实数据）
  ar3po_smoke_test.py         # ⭐ AR3PO 机制验证（离线极小模型）
  validate_run_config.py      # ⭐ run_grpo.sh 配置端到端验证
  export_dpo_pairs.py         # 从 rollout 自产偏好对
  trl_contract_probe.py       # TRL 数据契约探测
  evaluate_math.py            # ⭐ 评测：avg@k + 截断告警
  validate_jsonl.py           # ⭐ 三种 schema 的校验器
  merge_peft_adapter.py       # 合并 LoRA（进下一步前必做）
  convert_dataset.py, model_quant.py, eval_quantize.py
tests/                    # pytest 用例（pip install -r requirements-dev.txt）
  test_validate_jsonl.py, test_dpo_data_quality.py
docs/                     # 论文精读与分析文档（见下）
data/reward/README.md     # DPO 偏好对怎么生成（该目录初始为空是正常的）
_backup/                  # 重构前的完整备份（可回滚）
```

### 文档索引

| 文档 | 内容 |
|---|---|
| [GRPO_AR3PO_选型与实现方案.md](docs/GRPO_AR3PO_选型与实现方案.md) | 算法选型对比、论文精讲、TRL 映射、落地路线 |
| [Secrets_RLHF_PPO_精读.md](docs/Secrets_RLHF_PPO_精读.md) | Secrets of RLHF Part I (PPO) 全文精讲 |
| [PPO_论文精读.md](docs/PPO_论文精读.md) | PPO 原版论文精读 |
| [ppo_grpo_papers.md](docs/ppo_grpo_papers.md) | PPO / GRPO 论文清单 |
| [LLM_PPO_论文清单.md](docs/LLM_PPO_论文清单.md) | LLM 方向 PPO 论文谱系 |

---

## 九、已知限制

1. **中文数据是英译中**，不是原生中文；原生中文高中/考研题需要自建流水线。
2. **证明题不做 RL**（占比较低时收益不划算）。`numina_proof.jsonl` 只作评测集；若要训练需引入 Lean 形式化验证或离线 judge。
3. **`format_reward` 的分档设计是工程判断**，文献里 DAPO/R1-Zero 用的是二值。建议做一次 A/B，看 TRL 自动记录的 `frac_reward_zero_std` 与最终分数。
4. **AR3PO 的精度收益尚未在真实 7B/8B 上验证**——机制（形状契约、停梯度、分阶段生成）已用离线极小模型验证，但"追平/超过 DAPO"必须在你的真实数据上复现。
5. **response reuse 尚未落地**（`--replay_buffer_size 0`，默认关闭）。已核实 TRL 0.29.1 的实验 replay trainer 有两个问题：
   - **不切分 `tool_mask`**：`slice_group_data` / `update_replay_buffer` 里完全没有 tool_mask，会破坏 AR3PO「复制样本停梯度」的约束；
   - **更根本的是它在标准 GRPO 下也不可用**：父类产出的是 **1D** `grouped_std_rewards`，而它用 `max(dim=0).values > 0` 判断方差，对 1D 张量返回**标量**，导致「有任何方差组 → `num_groups_to_replace=0` → 静默 no-op」；「整批全零方差 → 采样空 buffer 得 None → 崩溃」。
   等 TRL 提供稳定实现后再启用。**宁可少一个优化，也不能静默训练错目标。**

---

## 十、致谢与许可

- 训练框架改造自 [shibing624/MedicalGPT](https://github.com/shibing624/MedicalGPT)（Apache-2.0）
- 核心算法：[AR3PO](https://arxiv.org/abs/2509.25808)、[DAPO](https://arxiv.org/abs/2503.14476)、[GRPO/DeepSeekMath](https://arxiv.org/abs/2402.03300)
- 原项目医疗相关内容已全部移除，其文档归档于 `docs/README_original_medical.md`
