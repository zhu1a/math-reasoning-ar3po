# PPO / GRPO 论文清单

> 面向本仓库 `training/ppo_training.py`（TRL `PPOTrainer`）与 `training/grpo_training.py`（TRL `GRPOTrainer`）的论文索引。
> 所有 arXiv 编号均已核对（arXiv Atom API + 摘要页）。

## 一、PPO 主线（必读）

| 论文 | 年份 | arXiv | 为什么读 |
|---|---|---|---|
| **Trust Region Policy Optimization (TRPO)**<br>Schulman et al. | 2015 / ICML | [1502.05477](https://arxiv.org/abs/1502.05477) | PPO 的前身。理解"信赖域"约束为什么会带来单调改进，是看懂 PPO 裁剪目标的前提。 |
| **High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)**<br>Schulman et al. | 2015 | [1506.02438](https://arxiv.org/abs/1506.02438) | 优势函数估计的标准做法，PPO/GRPO 里 advantage 的计算都用它。 |
| **Proximal Policy Optimization Algorithms (PPO)**<br>Schulman, Wolski, Dhariwal, Radford, Klimov | 2017 | [1707.06347](https://arxiv.org/abs/1707.06347) | **PPO 原始论文**。核心是 clipped surrogate objective，允许同一批数据做多轮 minibatch 更新，比 TRPO 简单且样本效率更好。 |

## 二、PPO 在 LLM / RLHF 上的落地

| 论文 | 年份 | arXiv | 要点 |
|---|---|---|---|
| **Fine-Tuning Language Models from Human Preferences**<br>Ziegler et al. (OpenAI) | 2019 | [1909.08593](https://arxiv.org/abs/1909.08593) | 最早把"奖励模型 + PPO"用到语言模型上，RLHF 范式的起点。 |
| **Learning to summarize from human feedback**<br>Stiennon et al. | 2020 / NeurIPS | [2009.01325](https://arxiv.org/abs/2009.01325) | 大规模人类反馈 + PPO 做摘要，KL 惩罚、奖励模型泛化等工程细节。 |
| **Training language models to follow instructions with human feedback (InstructGPT)**<br>Ouyang et al. | 2022 / NeurIPS | [2203.02155](https://arxiv.org/abs/2203.02155) | 现代 RLHF 三段式（SFT → RM → PPO）的标准流程，本仓库 `reward_modeling.py` 的 pairwise logloss 即出自此文。 |

## 三、GRPO 源头与大规模应用

| 论文 | 年份 | arXiv | 要点 |
|---|---|---|---|
| **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**<br>DeepSeek-AI | 2024 | [2402.03300](https://arxiv.org/abs/2402.03300) | **GRPO 的提出论文**。GRPO 是 PPO 的变体：用同一 prompt 的一组采样输出的组内相对奖励作为 baseline，**去掉 value/critic 网络**，显著省显存。本仓库 README 引用的就是这篇。 |
| **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**<br>DeepSeek-AI | 2025 / **Nature** 645:633-638 | [2501.12948](https://arxiv.org/abs/2501.12948) | 用 GRPO + 规则奖励做纯 RL 训练推理模型，R1-Zero 的 `aha moment`、长度增长等现象都在这里；工程细节（模板、奖励设计、失败模式）非常值得读。 |

## 四、GRPO 的分析与改进（进阶）

| 论文 | 年份 | arXiv | 要点 |
|---|---|---|---|
| **Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs (RLOO)**<br>Ahmadian et al. | 2024 / ACL | [2402.14740](https://arxiv.org/abs/2402.14740) | 论证 RLHF 场景下 PPO 的多数组件并不必要，REINFORCE 式（RLOO）更省算力且效果不差。本仓库 PPO 脚本标注了 "PPO/RLOO"。 |
| **DAPO: An Open-Source LLM Reinforcement Learning System at Scale**<br>ByteDance Seed et al. | 2025 | [2503.14476](https://arxiv.org/abs/2503.14476) | 在 GRPO 基础上做 clip-higher、动态采样、token-level loss、overlong 惩罚；开源了 verl 上的完整大规模 RL 系统。 |
| **Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)**<br>Liu et al. | 2025 | [2503.20783](https://arxiv.org/abs/2503.20783) | 指出 GRPO 目标里的优化偏置会人为拉长回复（尤其错答案），提出无偏的 Dr. GRPO，并给出极简 R1-Zero 配方。 |
| **Group Sequence Policy Optimization (GSPO)**<br>Qwen Team, Zheng et al. | 2025 | [2507.18071](https://arxiv.org/abs/2507.18071) | 把重要性比值与裁剪从 token 级改成**序列级**，训练更稳，尤其对 MoE RL 有效，是 Qwen3 的关键组件之一。 |

## 五、仓库中相邻方法的论文（对照阅读）

| 论文 | arXiv | 与仓库的对应 |
|---|---|---|
| **Direct Preference Optimization (DPO)** — Rafailov et al., 2023 | [2305.18290](https://arxiv.org/abs/2305.18290) | `training/dpo_training.py` |
| **ORPO: Monolithic Preference Optimization without Reference Model** — Hong et al., 2024 | [2403.07691](https://arxiv.org/abs/2403.07691) | `training/orpo_training.py` |

## 六、建议阅读顺序

1. **入门**：`1707.06347`(PPO) → `2203.02155`(InstructGPT)，先建立"SFT → RM → PPO"的完整心智模型。
2. **理解 PPO 细节**：`1502.05477`(TRPO) → `1506.02438`(GAE)。
3. **转 GRPO**：`2402.03300`(GRPO 提出) → `2501.12948`(R1，看它怎么用 GRPO 训推理)。
4. **动手调试**：`2503.20783`(Dr. GRPO，常见坑) → `2503.14476`(DAPO，超参与系统) → `2507.18071`(GSPO，稳定性)。
5. **对照**：`2402.14740`(RLOO) 与 DPO 论文，判断"到底要不要上 PPO/GRPO"。

## 七、与本仓库脚本的对应关系

> ⚠️ 本仓库已从医疗项目改造为**数学推理解题**项目，PPO 相关脚本已删除
> （见 [README.md](../README.md)）。本节保留作为历史对照。

| 论文 | 当前仓库文件 | 状态 |
|---|---|---|
| GRPO（[2402.03300](https://arxiv.org/abs/2402.03300)） | `training/grpo_training.py` | ✅ 在用（规则奖励，无 critic） |
| AR3PO（[2509.25808](https://arxiv.org/abs/2509.25808)） | `training/ar3po_trainer.py` | ✅ 在用（自适应 rollout） |
| 奖励函数设计 | `training/math_rewards.py` | ✅ 在用（整体 + 分层规范 + 长度） |
| DPO（[2305.18290](https://arxiv.org/abs/2305.18290)） | `training/dpo_training.py` | ✅ 在用（偏好对自产） |
| PPO（本篇 1707.06347） | ~~`training/ppo_training.py`~~ | ❌ 已删除（本项目不用 PPO） |
| ORPO（[2403.07691](https://arxiv.org/abs/2403.07691)） | ~~`training/orpo_training.py`~~ | ❌ 已删除 |
| 奖励模型（InstructGPT pairwise logloss） | ~~`training/reward_modeling.py`~~ | ❌ 已删除（改用规则奖励，省显存） |

本仓库当前的完整方案见 [GRPO_AR3PO_选型与实现方案.md](GRPO_AR3PO_选型与实现方案.md)。

## 八、其他检索入口

- arXiv 检索：<https://arxiv.org/list/cs.LG/recent>
- Papers with Code「RLHF / GRPO」主题页与 TRL 文档中的算法说明，可用于跟踪 2025 年后的新变体。
