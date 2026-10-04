# LLM 方向的 PPO 论文清单

> 回答"有没有 LLM 的 PPO 论文"：**有，而且 PPO 今天这么有名主要就是因为 LLM。**
> 原版 PPO（[1707.06347](https://arxiv.org/abs/1707.06347)）是 MuJoCo/Atari 论文，LLM 上的 PPO 是下面这条线逐步做出来的。
> 本地已下载的 PDF 参见本文末尾"本地文件"一节。

---

## 一、方法源头：把 PPO 搬到语言模型上（必读）

| # | 论文 | arXiv | 本地 | 为什么读 |
|---|---|---|---|---|
| 1 | **Fine-Tuning Language Models from Human Preferences**<br>Ziegler et al. (OpenAI), 2019 | [1909.08593](https://arxiv.org/abs/1909.08593) | ✅ | **最早把"奖励模型 + PPO"用到语言模型**的工作。情感续写、TL;DR 摘要。RLHF 范式的起点，能看到很多后来标准做法的雏形。 |
| 2 | **Learning to summarize from human feedback**<br>Stiennon et al., 2020 (NeurIPS) | [2009.01325](https://arxiv.org/abs/2009.01325) | — | 把 1 号工作规模化：大规模人类偏好数据 + 奖励模型 + PPO。**KL 惩罚、奖励模型泛化、奖励 hacking** 的经典分析都在这里。 |
| 3 | **Training language models to follow instructions with human feedback (InstructGPT)**<br>Ouyang et al. (OpenAI), 2022 (NeurIPS) | [2203.02155](https://arxiv.org/abs/2203.02155) | ✅ | **LLM 版 PPO 的"标准答案"**。SFT → RM → PPO 三段式、per-token KL 减进 reward、1.3B 打赢 175B。你仓库 `training/reward_modeling.py:228` 的 pairwise logloss 就出自这篇。 |
| 4 | **Training a Helpful and Harmless Assistant with RLHF**<br>Bai et al. (Anthropic), 2022 | [2204.05862](https://arxiv.org/abs/2204.05862) | — | helpful/harmless 双目标 + 迭代式在线 RLHF。**一个非常重要的实证发现：RL reward 与"策略偏离初始模型 KL 的平方根"大致成线性关系** —— 理解 KL 系数该设多大必看。 |

---

## 二、专门讲"LLM 上怎么把 PPO 调稳/调好"（最实用，强烈推荐）

| # | 论文 | arXiv | 本地 | 为什么读 |
|---|---|---|---|---|
| 5 | **Secrets of RLHF in Large Language Models Part I: PPO**<br>Zheng et al., 2023 | [2307.04964](https://arxiv.org/abs/2307.04964) | ✅ | ⭐ **这是最"就事论事讲 LLM 的 PPO"的一篇。** 逐组件拆解 PPO（advantage、clip、KL、value 初始化…）对策略训练的影响，结论是 **policy constraint（策略约束）是 PPO 能work的关键**，并提出 **PPO-max** 提升稳定性。还开源了 reward model 和 PPO 代码。 |
| 6 | **The N+ Implementation Details of RLHF with PPO: A Case Study on TL;DR Summarization**<br>Huang et al., 2024 | [2403.17031](https://arxiv.org/abs/2403.17031) | ✅ | ⭐ **复现踩坑清单。** 列出 20+ 个关键实现细节（adv 归一化、clip 范围、KL 系数、reward 白化、value loss 系数、学习率…），首次公开复现了 OpenAI TL;DR 的 scaling 行为。**要动手跑 PPO 就先读这篇。** |
| 7 | **Secrets of RLHF in LLMs Part II: Reward Modeling**<br>Wang et al., 2024 | [2401.06080](https://arxiv.org/abs/2401.06080) | — | PPO 的上游：奖励模型的数据噪声、偏好强度、对比学习、元学习。RM 不好，PPO 白跑。 |

---

## 三、PPO 在 LLM 上的定位与横向对比

| # | 论文 | arXiv | 要点 |
|---|---|---|---|
| 8 | **Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study**<br>Xu et al., ICML 2024 | [2404.10719](https://arxiv.org/abs/2404.10719) | 理论上指出 DPO 的固有局限，**实证上 PPO 在对话和代码生成上全面超过 DPO**，并总结 PPO 成功的关键因素。回答"到底要不要上 PPO"。 |
| 9 | **Llama 2: Open Foundation and Fine-Tuned Chat Models**<br>Touvron et al., 2023 | [2307.09288](https://arxiv.org/abs/2307.09288) | 工业级配方：**两轮 RLHF（先 rejection sampling，再 PPO）** + 迭代式奖励模型更新。附录里对 PPO 的稳定性讨论很实在。 |
| 10 | **WebGPT: Browser-assisted question-answering with human feedback**<br>Nakano et al., 2021 | [2112.09332](https://arxiv.org/abs/2112.09332) | ⚠️ 诚实标注：**它的最佳模型用的是 rejection sampling，不是 PPO。** 收录在此是因为它常被误当作"PPO 论文"，读时注意区分。 |

---

## 四、前身（不是 LLM，但决定了思路）

| 论文 | arXiv | 要点 |
|---|---|---|
| **Deep reinforcement learning from human preferences** — Christiano et al., 2017 | [1706.03741](https://arxiv.org/abs/1706.03741) | 用"人类对轨迹段的偏好"训练奖励模型，再用 RL 优化。RLHF 的思想源头，比 Ziegler 早两年。 |

---

## 五、医疗领域（和你这个仓库直接相关）

| 论文 | arXiv | 要点 |
|---|---|---|
| **HuatuoGPT, towards Taming Language Model to Be a Doctor**<br>Zhang et al., 2023 | [2305.15075](https://arxiv.org/abs/2305.15075) | 医疗问诊 LLM：ChatGPT 蒸馏数据 + 真实医生数据，**训练奖励模型做 RLAIF 式对齐**。可作为"MedicalGPT 同类工作"的参照。（其 RL 部分的具体算法以原文为准。） |
| **HuatuoGPT-II: One-stage Training for Medical Adaption of LLMs**<br>Chen et al., 2023 | [2311.09774](https://arxiv.org/abs/2311.09774) | 把预训练与监督阶段统一成 input-output 对，简化域适应流程；中文医学基准 SOTA。 |

---

## 六、从原版 PPO 到"LLM 版 PPO"的 6 个工程差异

读 LLM 版论文时，把这 6 条挂在心里，就不会被公式对不上困扰：

| # | 差异 | 说明 |
|---|---|---|
| 1 | **奖励稀疏、后置** | 原版 PPO 每步都有环境奖励 r_t；LLM 里奖励模型**只对整段 response 给一个标量**，再广播到每个 token。 |
| 2 | **KL 的位置变了** | 原版 PPO 用 KL 作为**目标函数的惩罚项/约束**（公式 5/8），约束的是**单步更新幅度**；LLM RLHF 把 **per-token KL 减进 reward**，且锚定的是**固定的 π_ref（SFT 模型）**，不是 θ_old。**这是最容易搞混的一点。** |
| 3 | **动作空间巨大且长度可变** | 动作 = 词表（几万维），轨迹 = 一整段 response，长度不一 → 需要 padding/packing、loss mask。 |
| 4 | **advantage 的用法变了** | 原版用 GAE(γ=0.99, λ=0.95)；LLM 上常见 γ=1、并对 advantage 做**白化/归一化**，因为回报就是单个标量分。 |
| 5 | **模型数量翻倍** | 原版只需要 policy + value（可共享）；LLM 版通常同时驻留 **policy + value + reward model + reference model** 四个模型 → 显存是主要瓶颈。 |
| 6 | **prompt 部分要 mask** | 只在 response token 上算策略损失和 KL，prompt token 不参与。 |

> 💡 第 5 条就是 GRPO（[2402.03300](https://arxiv.org/abs/2402.03300)）和 RLOO（[2402.14740](https://arxiv.org/abs/2402.14740)）的出发点：**能不能把 value 网络干掉？** GRPO 用"同一 prompt 的一组采样的平均奖励"当 baseline，RLOO 论证 RLHF 场景下 PPO 的多数组件本就不必要。

---

## 七、建议阅读顺序

```
① 2203.02155 InstructGPT      —— 先建立完整心智模型（SFT→RM→PPO）
        ↓
② 2307.04964 Secrets Pt.I: PPO —— 知道 PPO 各组件在 LLM 上到底起什么作用
        ↓
③ 2403.17031 N+ Details        —— 动手前必读的实现细节清单
        ↓
④ 2204.05862 Anthropic HH      —— 理解 KL 系数的经验规律（reward ~ √KL）
        ↓
⑤ 2404.10719 DPO vs PPO        —— 判断到底该不该上 PPO
（1 号 1909.08593 和 2 号 2009.01325 作为历史脉络随时翻阅）
```

**只读一篇的话**：读 [2307.04964](https://arxiv.org/abs/2307.04964)（Secrets of RLHF Part I: PPO）——它是唯一一篇把标题直接写成"LLM 的 PPO"的论文。

---

## 八、本地文件

`docs/papers/` 下已下载：

| 文件 | 对应论文 |
|---|---|
| `LLM_PPO_2307.04964.pdf` | Secrets of RLHF in LLMs Part I: PPO（32 页）⭐ |
| `LLM_PPO_2403.17031.pdf` | N+ Implementation Details of RLHF with PPO（42 页）⭐ |
| `LLM_PPO_2203.02155.pdf` | InstructGPT（68 页，含全部附录） |
| `LLM_PPO_1909.08593.pdf` | Fine-Tuning Language Models from Human Preferences（26 页） |
| `PPO_1707.06347.pdf` | 原版 PPO（12 页），精读见 `docs/PPO_论文精读.md` |

## 九、和你仓库的对应

| 论文 | 仓库文件 | 状态 |
|---|---|---|
| PPO 原版 + LLM 版 PPO | `training/ppo_training.py` | ⚠️ 文件名沿用，但**实现已换成 RLOO**（`RLOOTrainer`，见 `ppo_training.py:21-26, 285-287`）；本环境 TRL 0.29.1 已无 `PPOConfig`/`PPOTrainer` |
| InstructGPT 的 pairwise logloss | `training/reward_modeling.py:228` | 奖励模型训练（PPO 的前置阶段） |
| GRPO | `training/grpo_training.py` | `GRPOTrainer`，无需 critic |
| 兼容性限制 | `docs/FAQ.md` | chatglm/baichuan 无 value head，做不了 PPO/RM |
