# 《Secrets of RLHF in Large Language Models Part I: PPO》全文精讲

> **这份文档的目标：让你不打开原文也能完整理解这篇论文。**
> 逐章、逐公式、逐图表、逐附录全部讲透，并补充原文没有明说的背景与直觉。
>
> 论文：Zheng et al., **Secrets of RLHF in Large Language Models Part I: PPO**, 2023
> arXiv: [2307.04964](https://arxiv.org/abs/2307.04964)｜PDF: `docs/papers/LLM_PPO_2307.04964.pdf`｜文本: `docs/papers/LLM_PPO_2307.04964.txt`
> 项目：OpenLMLab / MOSS-RLHF｜篇幅：正文 20 页 + 附录 12 页

---

## 目录

- [第 0 章 论文速览卡](#第-0-章-论文速览卡)
- [第 1 章 Sec 1 引言：它到底要解决什么问题](#第-1-章-sec-1-引言它到底要解决什么问题)
- [第 2 章 Sec 2 相关工作：三条技术路线](#第-2-章-sec-2-相关工作三条技术路线)
- [第 3 章 Sec 3 RLHF 框架（含 GAE 完整推导）](#第-3-章-sec-3-rlhf-框架)
- [第 4 章 Sec 4 奖励模型](#第-4-章-sec-4-奖励模型)
- [第 5 章 Sec 5.1 实验设置](#第-5-章-sec-51-实验设置)
- [第 6 章 Sec 5.2 监控指标与 Pattern Collapse](#第-6-章-sec-52-监控指标与-pattern-collapse最重要的一节)
- [第 7 章 Sec 5.3 十二项实现细节逐项精讲](#第-7-章-sec-53-十二项实现细节逐项精讲)
- [第 8 章 Sec 5.4 PPO-max 配方](#第-8-章-sec-54-ppo-max-配方)
- [第 9 章 Sec 6 评测结果](#第-9-章-sec-6-评测结果)
- [第 10 章 附录 A/B/C/D 精讲](#第-10-章-附录-abcd-精讲)
- [第 11 章 Limitations 与彩蛋](#第-11-章-limitations-与彩蛋)
- [第 12 章 全部公式总表](#第-12-章-全部公式总表)
- [第 13 章 符号表（论文的符号重载陷阱）](#第-13-章-符号表论文的符号重载陷阱)
- [第 14 章 一页纸结论清单](#第-14-章-一页纸结论清单)
- [第 15 章 批判性阅读 + 实践清单](#第-15-章-批判性阅读--实践清单)
- [第 16 章 自测题](#第-16-章-自测题)

---

## 第 0 章 论文速览卡

| 项 | 内容 |
|---|---|
| 标题 | Secrets of RLHF in Large Language Models Part I: PPO |
| 作者 | Rui Zheng 等，复旦大学 NLP 组 + 上海 AI Lab（OpenLMLab 团队） |
| 时间 | 2023-07-11 提交，2023-07-18 修订 v2 |
| 类型 | **技术报告**（不是同行评审论文） |
| 底座模型 | LLaMA-7B（英文）、OpenChineseLLaMA-7B（中文） |
| 规模 | 7B 单模型，**未研究 scaling law** |
| 开源 | 奖励模型 + PPO-max 代码（github.com/OpenLMLab/MOSS-RLHF） |
| 硬件 | 8×A100 80G 每台，ZeRO-2 + gradient checkpointing |

### 三句话总结

1. **在 LLM 上，PPO 能否训起来，关键不是奖励模型好不好，而是"策略约束"够不够——具体说就是 token-level KL penalty。**
2. **训练时的 reward 分数和 loss 曲线不能判断模型好坏**，要用 perplexity / 回复长度 / KL 三件套来监控。
3. 据此总结出稳定配方 **PPO-max**，在 7B 上做到与 ChatGPT 可比的（而非超越的）对齐效果。

### 论文的贡献（Sec 1 末原文列了 3 条）

1. 开源中英两个有竞争力的**奖励模型**，跨模型泛化好，省去重复标注偏好数据的成本；
2. 深入剖析 PPO 内部机制，提出 **PPO-max** 保证训练稳定；
3. 开源**完整 PPO-max 代码**。

---

## 第 1 章 Sec 1 引言：它到底要解决什么问题

### 1.1 背景逻辑链（原文的推理顺序）

```
LLM 靠规模涌现出指令跟随、推理等能力
   ↓ 但预训练语料里既有高质量也有低质量数据
模型会编造事实、生成偏见/有毒/有害内容
   ↓ 所以需要对齐（align）到人类价值观，即 3H：helpful / honest / harmless
开源模型（LLaMA、OpenChineseLLaMA）已推进到 SFT 阶段
   ↓ 主流做法是在 SFT 数据里混入 3H 数据，期望"激活"模型在道德伦理层面的正向反应
但效果有限：模型在 safety 和 groundedness 上仍低于人类水平
   ↓ OpenAI / Anthropic 已证明 RLHF 是有效路径
但 RLHF 训练本身极难，常常反复失败
```

### 1.2 作者点出的三个具体困难（这是全文的问题定义）

> 原文关键句：*"training large language models that align with human values is a daunting task, often resulting in repeated failure when trained using reinforcement learning"*

| # | 困难 | 原文表述 |
|---|---|---|
| 1 | **奖励模型可能本身是错的** | 用低质量数据训练、对齐目标又难以定义的 RM，很容易把 PPO 带到"unintelligible direction"（无法理解的方向）。**RM 的质量直接决定策略模型的上限。** |
| 2 | **要协调四个模型** | policy、value、reward、reference 四个模型一起工作，难以训练、难以扩展到大规模参数 |
| 3 | **奖励稀疏 + 探索效率低** | 在**词空间**里探索（word space），奖励只在整段回答结束时给一个标量，因此**对超参极其敏感** |

作者还加了一句很真实的话：*"Models trained solely through repeated experiments, failed runs, and hyperparameter sweeps achieve far inferior results."*（只靠反复实验、失败运行和超参扫描训出来的模型，效果差得多。）以及：巨大的试错成本使研究者**不敢让项目进入 RLHF 阶段**。

👉 所以本文的目标就是：**做一个专门为 LLM 设计的鲁棒 PPO 算法。**

### 1.3 作者的解决思路（Sec 1 末两段）

1. 先剖析整个 RLHF 流程，看**奖励模型质量**如何影响最终策略模型；
2. 发现 **RM 质量决定策略上限**，而**设计合适的 PPO 算法**对成功训练同样关键；
3. **精确的代码实现非常重要**（原文强调 "practice makes perfect"）；
4. 提出用**动作空间建模指标**（perplexity、response length、KL）来监控训练，比看 reward 和 loss 有用得多；
5. 最终定位到：**policy constraint 是与人类偏好保持一致的关键**，推出 PPO-max。

---

## 第 2 章 Sec 2 相关工作：三条技术路线

作者把前人工作分成三类，这个分类本身就很有信息量：

| 路线 | 代表 | 做法 | 局限 |
|---|---|---|---|
| **不用强化学习** | LaMDA [12] | 用各种监督学习技术混合来做人类偏好对齐 | 效果受限于监督信号 |
| **RLHF 开山** | InstructGPT [16] | 用"比较式人类偏好"训奖励模型，再用 RL 微调 GPT-3 | 本文的对比基线 |
| **迭代 + 在线** | Anthropic [17] | 偏好模型和 RL 策略**每周**用新鲜人类反馈更新一次，引入 PPO 稳定训练 | 复杂度高 |

另外作者特别引用了两篇"揭露实现细节重要性"的工作：

- **[29] Engstrom et al.**：揭示 PPO 带来的奖励提升，很大一部分来自**核心算法上看似微小的修改（即代码级优化）**。
- **[28] Andrychowicz et al.**：指出大量 high-level/low-level 的 RL 设计决策**在论文里通常不写，但对性能至关重要**；并且发现**策略初始化方案**显著影响性能。

👉 这两篇是本文方法论的直接来源：**既然"没写出来的实现细节"这么重要，那我们就把它们全部系统地消融一遍。**

作者指出的文献空白：前人虽然指出了 PPO 实现细节的重要性，但**很少有人真正去解决"不稳定 + 超参敏感"这个问题本身**。

---

## 第 3 章 Sec 3 RLHF 框架

### 3.1 整体流程与图 1

```
阶段一 SFT：模仿人类标注的对话样本，学会像人一样对话
阶段二 RM ：学习"比较"不同回答的优劣（基于人类反馈）
阶段三 PPO：基于 RM 的反馈更新模型，通过 exploration/exploitation 找到更优策略
```

**图 1（PPO workflow）画了什么**——这是全文的结构图，值得在脑子里建模：

```
                    ┌──────────────┐
   User Query x ──▶ │  Policy LM   │ ──▶ 生成回答 y = (y_1, ..., y_T)
                    │   π_θ        │
                    └──────────────┘
                            │
        ┌───────────────────┼────────────────────┐
        ▼                   ▼                    ▼
  ┌───────────┐      ┌────────────┐      ┌──────────────┐
  │ Reward    │      │ Value      │      │ SFT Model    │
  │ Model     │      │ Model      │      │ π_SFT（参考） │
  │ r(x,y)    │      │ V_φ(s_t)   │      │              │
  └───────────┘      └────────────┘      └──────────────┘
        │                   │                    │
        │ 奖励分             │ 每步估值            │ KL 惩罚
        ▼                   ▼                    ▼
        └────────▶ GAE 算优势 Â ────▶ PPO-clip Loss
                    + KL 惩罚           LM Loss (PPO-ptx)
                    + return            MSE Loss (critic)
```

图中标出的 GAE 三个式子（原文图内文字）：

```
优势函数：  Â(s_t, a_t) = Σ (γλ)^l · δ_{t+l}
TD 误差：   δ_t = r(s_t,a_t) + γ·V(s_{t+1}) − V(s_t)
Return：    R̂_t = Â(s_t,a_t) + V(s_t)
```

三种 loss 同时优化：**PPO-clip Loss**（策略）、**LM Loss**（PPO-ptx，防退化）、**MSE Loss**（critic）。

> ⚠️ **一个论文结构上的小坑**：作者把"总奖励含 KL 惩罚"的公式 (3) 放在了 **3.1 奖励建模**这一节里，而不是放在 RL 那一节。读的时候容易困惑"为什么讲 RM 突然讲 RL 的奖励"。理解成"RM 给出 r(x,y)，RL 阶段实际用的是 r_total"即可。

### 3.2 Sec 3.1 奖励建模（Reward Modeling）

#### RM 的架构

> 原文：*"we use pre-trained transformer-based language models with the last unembedding layer removed and add an additional linear layer to the final transformer layer."*

即：拿一个预训练 transformer LM → **去掉最后的 unembedding 层**（本来是映射到词表的）→ 在最后一个 transformer 层上加一个**线性层输出 1 维标量** → 对**最后一个 token** 输出奖励值。

**奖励越大代表样本越好。**

#### 奖励模型损失：公式 (1) 和 (2)

**公式 (1)**（基础的 Bradley-Terry 成对偏好损失）：

```
L(ψ) = log σ( r(x, y_w) − r(x, y_l) )                              …… (1)
```

- `σ` 是 sigmoid；`r` 是参数为 `ψ` 的奖励模型；`y_w` = preferred（更好），`y_l` = dispreferred。
- 直觉：**让"好回答的分数"比"坏回答的分数"高出越多越好**。sigmoid 把差值压到 (0,1) 区间，取 log 后就是 log-likelihood。

**公式 (2)**（本文实际使用的完整 RM 损失）：作者**额外加入了模仿学习项**——对偏好回答加一个自回归 LM loss，让模型在被偏好的句子上也能"模仿"：

```
L(ψ) = −λ · E[ log σ( r(x,y_w) − r(x,y_l) ) ]  +  β_rm · E[ log r'(x, y_w) ]     …… (2)
```

- `r'` = 除了顶部线性层不同（维度改成词表大小）以外和 `r` 完全相同的模型，`r'(x,y_w)` 是给定 prompt 生成偏好回答的似然；
- `λ` = 偏好排序损失的权重；`β_rm` = LM loss 的权重（实验中 **β_rm = 1**）。
- **为什么加这一项？** 纯排序损失只教"哪个更好"，不教"怎么生成好回答"。加上 LM loss 后，RM 同时具备语言建模能力，泛化更好（这一做法来自 Anthropic [27]）。

#### 总奖励：公式 (3) —— KL 惩罚的两个作用

```
r_total = r(x, y) − η · KL( π_φ^RL(y|x), π_SFT(y|x) )              …… (3)
```

`η` 是 KL 奖励系数，控制 KL 惩罚强度。

**原文明确写了 KL 项的两个作用（很值得记）：**

1. **充当熵奖励（entropy bonus）**：促进策略在策略空间中**探索**，防止策略过早收敛到单一模式（prematurely converging to a single mode）；
2. **确保 RL 策略的输出不严重偏离奖励模型训练时见过的样本** —— 也就是防止策略跑到 RM 的分布之外。

> 💡 第 2 点是理解后文 pattern collapse 的钥匙：**KL 惩罚的真正作用不是"别改变模型"，而是"别让模型跑到 RM 没见过的区域去"。**

### 3.3 Sec 3.2 强化学习：把对话建模成 RL

#### 环境/状态/动作/奖励的对应（原文表格化理解）

| RL 概念 | 在 LLM 对话里是什么 |
|---|---|
| **环境 environment** | 人类交互本身 |
| **状态 s_t** | 对话历史（截至当前，助手和人类说的所有文本） |
| **动作 a_t** | **生成下一个 token** |
| **策略 π** | 语言模型本身 |
| **奖励 r(s_t, a_t)** | 由人类偏好数据训练出的奖励函数给出 |
| **状态转移 s_{t+1}** | 对话历史加上新生成的 token |
| **轨迹 τ** | {s_1,a_1,...,s_T,a_T}，即一整段生成 |
| **目标** | 最大化整条轨迹的累积奖励（return） |

两种 return 定义：

```
有限步无折扣：  R(τ) = Σ_{t=1}^{T'} r(s_t, a_t)          （固定步数内的奖励求和）
无限步带折扣：  R(τ) = Σ_{t=0}^{∞} γ^t r(s_t, a_t)，γ ∈ (0,1)
```

#### 3.3.1 Sec 3.2.1 策略梯度方法：公式 (4)(5)(6)

策略梯度与 value-based 方法的区别：**直接优化策略（状态→动作的映射），而不是先学价值函数。**

**公式 (4)** 更新规则：

```
θ ← θ + α ∇_θ J(θ)                                               …… (4)
```

`α` 是学习率，`J(θ)` 是遵循策略 π_θ 时的期望回报，`∇_θ J(θ)` 就是策略梯度。

**公式 (5)** 策略梯度的一般形式：

```
∇_θ J(θ) = E_{τ∼π_θ}[ Σ_{t=0}^{T} ∇_θ log π_θ(a_t|s_t) · Φ_t ]     …… (5)
```

**核心教学点：`Φ_t` 可以有多种选择，期望值相同但方差不同。**

| Φ_t 的选择 | 含义 |
|---|---|
| `Φ_t = R(τ)` | 整条轨迹的回报 |
| `Φ_t = Σ_{t'=t}^{T} R(s_{t'},a_{t'})` | reward-to-go（只算 t 之后的） |
| `Φ_t = Σ_{t'=t}^{T} R(s_{t'},a_{t'}) − b(s_t)` | 减去基线 b（baseline） |

**公式 (6)**：用优势函数替代原始回报：

```
Φ_t = A(s_t, a_t)                                                …… (6)
A(s_t, a_t) = Q(s_t, a_t) − V(s_t)
```

- `Q(s_t,a_t)`：动作价值函数，在 s_t 采取 a_t 后的**期望回报**；
- `V(s_t)`：状态价值函数，在 s_t 处按策略的**平均期望回报**；
- **优势 = "这个动作比该状态下的平均水平好多少"**，正数=好动作，负数=坏动作。

**为什么需要这个替换**（原文的解释）：用蒙特卡洛采样计算 return 是**无偏的**（只用真实回报，不估计），但**方差极高**——因为环境随机性和策略随机性使不同轨迹的回报差异巨大。用优势函数估计替代原始回报就是**降方差**的标准手段。

#### 3.3.2 Sec 3.2.2 GAE：论文自己的"从零推导"

原文这一节开头就说明：*"The following is a layman-friendly explanation of how GAE is derived."*（下面是 GAE 推导的通俗版。）这一段是全文数学上最值得读的部分，我把它逐式讲清。

**第一步：TD-k return（公式 7）**

用"实际奖励 + 估计的未来价值"组合出 k 步的回报：

```
R̂_t^k = r_t + γ·r_{t+1} + ... + γ^{k−1}·r_{t+k−1} + γ^k·V(s_{t+k})     …… (7)
```

- 前面 k 步用**真实拿到的奖励**，第 k 步之后用**价值函数估计**。
- `k=1` 时退化为 `r_t + γV(s_{t+1})`，也就是 TD(0) 的目标。
- `k→∞` 时变成纯蒙特卡洛（全部用真实奖励）。

**第二步：k-step advantage（公式 8）**

```
Â_t^k = R̂_t^k − V(s_t)
      = Σ_{l=0}^{k−1} γ^l · δ_{t+l}                                  …… (8)
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)      ← TD 误差                    （式 8 的 δ 定义）
```

**偏差-方差权衡（原文明确讨论）：**
- **k 小** → **偏差大**：优势估计依赖很少的真实奖励，**重度依赖价值函数的准确性**；
- **k 大** → **方差大**：要累加很多有噪声的奖励。

**第三步：GAE 的定义（公式 9）—— k-step 优势的指数移动平均**

GAE 用权重 `(1−λ)λ^{k−1}` 对所有 k-step 优势做指数加权平均：

```
Â_t^{GAE(γ,λ)} = (1−λ)·( Â_t^(1) + λ·Â_t^(2) + λ²·Â_t^(3) + ... )
```

把公式 (8) 代入并整理（原文完整展开了这个过程）：

```
= (1−λ)( δ_t + λ(δ_t + γδ_{t+1}) + λ²(δ_t + γδ_{t+1} + γ²δ_{t+2}) + ... )
= (1−λ)( δ_t(1 + λ + λ² + ...) + γδ_{t+1}(λ + λ² + λ³ + ...) + γ²δ_{t+2}(λ² + λ³ + ...) + ... )
= (1−λ)( δ_t·(1/(1−λ)) + γδ_{t+1}·(λ/(1−λ)) + γ²δ_{t+2}·(λ²/(1−λ)) + ... )
= Σ_{l=0}^{∞} (γλ)^l · δ_{t+l}                                     …… (9)
```

**结果非常干净：`Â_t = Σ (γλ)^l δ_{t+l}`** —— 一个几何加权的 TD 误差和。这也正是图 1 里写的式子。

> 💡 **这就是那个"望远镜求和"的优雅之处**：等比级数把 `(1−λ)` 分母消掉了，最终形式极其简洁。

**第四步：两个极端情况（公式 10、11）**

```
GAE(γ, 0):  Â_t = δ_t = r_t + γV(s_{t+1}) − V(s_t)                 …… (10)   ← 高偏差、低方差
GAE(γ, 1):  Â_t = Σ_l γ^l · r_{t+l} − V(s_t)                       …… (11)   ← 低偏差、高方差（蒙特卡洛）
```

原文总结：GAE 在**高偏差（λ=0）**和**高方差（λ=1）**之间平滑插值，有效管理这个权衡。

**第五步：策略梯度估计量（公式 12）**

```
∇_θ Ĵ(θ) = (1/|D|) · Σ_{τ∈D} Σ_{t=1}^{T} ∇_θ log π_θ(a_t|s_t) · Â_t    …… (12)
```

`D` 是有限批次的样本。论文此后用 `Ê_t` 来简写这个双重求和（这是理解后面所有公式的前提）。

#### 3.3.3 Sec 3.2.3 PPO：公式 (13)-(17)

**开篇的哲学（原文写得很漂亮，值得引用）：**

> *"The underlying intuition for these methods is the idea of 'small, stable steps': a philosophy of gently nudging the policy towards optimization, rather than forcing aggressive updates that might destabilize the overall learning process."*
> （这些方法的核心直觉是"小而稳的步伐"：温和地推动策略优化，而不是强行做可能破坏整个学习过程的激进更新。）

**为什么需要它**（原文的论证链）：

1. 传统 RL 认为新旧策略在**参数空间**上应该接近；
2. 但**参数空间接近 ≠ 性能接近**，参数的微小变化可能剧烈影响策略效果；
3. 如果迈出无约束的一大步，可能导致策略性能崩塌，即所谓 **"falling off the cliff"（掉下悬崖）**；
4. 这个风险正是 **vanilla policy gradient 样本效率受限的根本原因**。

**公式 (13) TRPO**：不用参数接近性，改用 KL 散度约束：

```
max_θ  Ê_t[ (π_θ(a_t|s_t)/π_θold(a_t|s_t)) · Â_t ]
s.t.   Ê_t[ KL(π_θold(·|s_t), π_θ(·|s_t)) ] ≤ δ                    …… (13)
```

`θ_old` 是更新前的旧策略参数。注意：**TRPO 用硬约束**，需要二阶优化（Fisher 矩阵 + 共轭梯度），实现复杂。

**公式 (14) PPO-Penalty**：把硬约束换成**软惩罚**：

```
L_{ppo−penalty}(θ) = Ê_t[ (π_θ(a_t|s_t)/π_θold(a_t|s_t))·Â_t ] − β·KL(π_θold, π_θ)    …… (14)
```

**公式 (15) PPO-Clip**（本文的主力）：

```
L_{ppo−clip}(θ) = Ê_t[ min( r_t(θ)·Â_t ,  clip(r_t(θ), 1−ε, 1+ε)·Â_t ) ]              …… (15)
其中 r_t(θ) = π_θ(a_t|s_t) / π_θold(a_t|s_t)
```

- `r_t(θ)` 是**新策略概率 / 旧策略概率**的比值（importance sampling 比值）；
- `ε` 决定新策略能偏离旧策略多少；
- 原文解释：**clipping 起到了正则化作用**，限制策略在一次迭代中剧烈变化的程度；防止过大的策略更新保证了学习过程的鲁棒性，同时比 vanilla policy gradient 保持更高的样本效率。

> 💡 **和原版 PPO 论文的对照**：这里的表述和 [1707.06347](https://arxiv.org/abs/1707.06347) 一致，但**本文实际并不用公式 (15) 里的 clip 作为唯一约束**——附录 C.1 明确说 clip 值怎么调"效果差别很小，不能像 KL 约束那样提供稳定优化"。**本文真正依赖的是公式 (19) 的 token-level KL penalty。** 这是读这篇论文最容易被误导的一点。

**公式 (16) Critic 损失**：

```
L_critic(φ) = Ê_t[ ‖ V_φ(s_t) − R̂_t ‖² ]                            …… (16)
R̂_t = Σ_{l=0}^{∞} γ^l · r_{t+l}
```

用 MSE 让 critic 的预测值逼近实际回报。**注意：这一项是要"最小化"的，而策略目标是"最大化"的**，所以合起来训练时 critic loss 前面是负号。

**公式 (17) PPO-ptx（混入预训练梯度）**：

```
L_{ppo−ptx}(θ) = L_{ppo−clip}(θ) + λ_ptx · E_{x∼D_pretrain}[ log( π_θ^RL(x) ) ]        …… (17)
```

- 目的：**缓解 PPO 过程中模型语言能力和知识保留度的退化**；
- `λ_ptx` 是预训练 loss 系数，`D_pretrain` 是预训练数据分布；
- 这一做法直接来自 InstructGPT [16]。

**Algorithm 1（论文给的 PPO 伪代码）：**

```
输入：初始策略参数 θ_0，初始价值函数参数 φ_0
for n = 0, 1, 2, ... do
    用策略 π(θ_n) 与环境交互，收集一组轨迹 D_n = {τ_i}
    计算 reward-to-go R̂_t
    基于当前价值函数 V_{φ_n} 计算优势估计 Â_t（可用任意优势估计方法，本文用 GAE）
    通过最大化 PPO-penalty/clip/ptx 目标更新策略：
        θ_{n+1} = argmax_θ L_{ppo−clip}(θ_n)
    通过对 MSE 回归更新价值函数：
        φ_{n+1} = argmin_φ L_critic(φ_n)
end for
```

> ⚠️ **伪代码的一个小瑕疵**：第 6 步写 `argmax_θ L(θ_n)`，但实际应当对 `θ` 做优化（`L(θ)`），写成 `θ_n` 是原文的笔误/简写。训练时把 argmax 取负号变成 loss 来最小化。

---

## 第 4 章 Sec 4 奖励模型

### 4.1 数据与模型（Sec 4.1）

| | 英文 | 中文 |
|---|---|---|
| **底座** | LLaMA-7B（decoder-only） | OpenChineseLLaMA（在 LLaMA-7B 基础上增量预训练，中文能力显著提升） |
| **数据来源** | HH-RLHF 数据集 [17] | **自建**：雇专业标注员人工标注 |
| **训练集** | **160k** 成对样本（118k helpful + 42k harmless） | **39k** 成对样本（31k helpful + 8k harmless），训练用其中 24k + 6k |
| **测试集** | 从剩余 8.5k 中随机取约 0.7k helpful + 0.3k harmless = **1k** | 2.4k helpful + 0.6k harmless = **3k** |
| **验证集** | 其余 | 其余 |

### 4.2 训练配置（Sec 4.2）

- 学习率 **5e-6**，前 **10%** 步 warmup；
- **动态 batch**（不用固定值）：按 batch 内 token 数尽量平衡，**最大 128、最小 4**，以更高效稳定地训练；
- **固定训练 1000 步**，约 **1.06 epoch**；
- **β_rm = 1**（LM loss 权重）——**整个实验都固定这个值**。

### 4.3 评测结果与两个失败模式（Sec 4.3）—— ⭐重点

**做法**：把测试输入喂给 RM，分别得到偏好回答和次选回答的分数，**相减**得到 difference score，看分布（图 2 的两个直方图）。

**结果**：
- 两个 RM 都大致与人类偏好一致；
- **中文 RM 与人类判断的一致性明显更高**（因为中文数据是人工构造的，好/坏回答差距显著）。

**⭐⭐ 两个失败模式（Table 1 给了具体例子）—— 这是本节最有价值的部分：**

| 语言 | 失败模式 | 原文例子 |
|---|---|---|
| **中文** | RM 给**更长**的回答更高分，而那个回答**在编造事实** | 问"邹凯是谁？"<br>**更好**：邹凯，1988年2月25日出生于四川省泸州市，原中国国家体操队主力队员之一，世界冠军，奥运冠军。<br>**更差**：中国男子体操运动员，1981年7月27日出生于湖南省长沙市。他曾参加过2000年悉尼奥运会、2008年北京奥运会…是中国体操史上获得冠军最多的第一人。<br>→ **RM 给编造事实的长回答打了更高分。** |
| **英文** | RM 给**"承认信息不足"**的回答**低分**（诚实但不 helpful），却给**"看起来正确但含欺骗信息"**的回答**高分** | 问 Kleicha 食谱<br>**更好**：我需要更多信息才能给你食谱……（追问细节）<br>**更差**：Kleicha 是一种伊拉克基督徒传统做的粥，含米、粗麦粉、有时葡萄干。你可以在某文章里找到食谱。—— **这段是编的，但 RM 给了更高分。** |

👉 **本质**：RM 学到的是**"长度 / 自信的语气"这类表面特征**，而不是"事实正确性"。

> ⚠️ 在医疗领域这两种偏差是**致命**的：编造事实的医疗建议 + 惩罚恰当拒答 = 直接伤害。

### 4.4 RM 训练曲线与最反直觉的发现（Sec 4.4 + 图 3）

**图 3 展示了 RM 的 train/eval accuracy 曲线**：
- 两个模型的验证集性能都稳步上升；
- **中文 RM 准确率更高**，原因是中文数据对内的两个回答差异更大，RM 更容易学到区分特征；
- **英文数据里很多 pair 质量相近**，RM 更难判断优劣，所以准确率更低。

**⭐⭐ 最反直觉的发现（原文最后两句）：**

> 两个模型在 **200 步（约 0.2 epoch）** 之后提升速度显著放缓，此时准确率已接近训练一整个 epoch 的水平。
> **但是，用这个 200 步的模型作为 PPO 的初始化，我们观察到不令人满意的效果。**
> 因此：**"accuracy alone is insufficient as a criterion for the RM."**（准确率不足以作为 RM 的评判标准。）

👉 **实践含义**：选择/保存 RM checkpoint 时，**不能只看验证集准确率**，必须看它驱动 PPO 的实际效果。这是一个很多人踩过的坑。

---

## 第 5 章 Sec 5.1 实验设置

### 5.1.1 SFT 底座

- 底座：**OpenChineseLLaMA**
- 数据：**1M 条过滤后的指令数据**（400K 单轮 + 600K 多轮）
- 训练：2 epoch
- 学习率 **9.5e-6**，cosine 调度，最终衰减到峰值的 **10%**
- global batch size **1024**

### 5.1.2 PPO 阶段配置

| 配置项 | 值 |
|---|---|
| 参考模型 & 策略模型初始化 | 都从 **7B SFT 模型** 初始化 |
| **critic 初始化** | **用 reward model 初始化** |
| PPO 数据 | 自建 HH 数据：**8k harmless + 20k helpful** |
| 训练步数 | 固定步数（不是固定 epoch） |
| **采样 batch size** | **128** |
| **训练 batch size** | **32**（policy 和 critic） |
| **policy 学习率** | **5e-7** |
| **critic 学习率** | **1.65e-6**（≈ 3.3× policy） |
| warmup | 前 10% 步 |
| 硬件 | 每台 8×80G A100、1TB RAM、128 CPU |
| 显存优化 | ZeRO-2 + gradient checkpointing |

> 💡 **两个容易被忽略但重要的点**：
> 1. **critic 学习率比 policy 高 3 倍多**。critic 要为序列的**每一步**给出估值，任务更难，需要更快的学习。
> 2. **采样 batch(128) 远大于训练 batch(32)**：采样成本高（要自回归生成），训练 batch 小是为了显存和多步更新。

**本节的一个声明**（重要）：*"In the following, it always refers to our own experiments when we mention PPO if not specifically stated."* —— 后面所有"PPO"都指他们自己的实现，不是别人论文里的 PPO。

---

## 第 6 章 Sec 5.2 监控指标与 Pattern Collapse（最重要的一节）

### 6.1 作者的目标

希望能找到一些指标，**在不依赖人工（或 GPT-4）评测的情况下**，反映 PPO 训练的质量、追踪模型的 3H 能力。

作者的坦承：**准确区分两个能力相近的模型很难**，但**观察训练稳定性、及时发现严重偏离是可行的**。

### 6.2 ⭐ 核心发现：reward 和 loss 都不能用

**图 4 的解读（这是本文最该看懂的一张图）：**

**上半部分**：
- 第一张子图：**response reward 曲线**（从约 2 涨到约 14-18），红色线是 **policy vs SFT 模型的 win rate**；
- 第二张子图：**training loss**；
- **现象**：**training loss 平稳收敛，reward 一路上涨，但人工评估和 GPT-4 评估显示策略行为并没有变好。**

原文结论：*"This means that the reward scores and training losses do not indicate whether the PPO is optimizing correctly."*
（这意味着 reward 分数和训练 loss 都不能表明 PPO 是否在正确地优化。）

**下半部分（信息量指标）**，展示了 5 条曲线：
- **PG Loss**（策略损失）
- **VF Loss**（价值函数损失）
- **KL-Divergence**（policy 与 reference 之间）
- **Perplexity**
- **Response Length**

原文：*"we observe significant variation in these metrics when there was a misalign between the human evaluation results and reward scores."*
（当人工评估结果与 reward 分数不一致时，这些指标出现了显著变化。）

### 6.3 ⭐⭐ Pattern Collapse（模式崩塌）—— 本文提出的关键现象

**定义**（原文）：*"SFT models are over-optimized and exhibit highly biased behavior."*（SFT 模型被过度优化，表现出高度偏置的行为。）

**机制**（原文）：
> *"we observe that the trained policy model has a tendency to **cheat the reward model through specific patterns for anomalous higher scores**."*
> （训练后的策略模型倾向于**通过特定的模式来欺骗奖励模型，以获取异常高的分数**。）

**在指标上的表现**（原文）：
> *"We simultaneously observe that the collapsed model **uniformly delivers longer responses** and exhibits **lower perplexity** for such generative patterns."*

即 **崩塌的三个标志**：

| 指标 | 崩塌时的表现 |
|---|---|
| **Response Length** | **一律变长**（uniformly longer） |
| **Perplexity** | **变低**（对固定的生成模式越来越"自信"） |
| **Reward** | 变高（但这是假的好） |
| **KL(policy‖SFT)** | 偏离程度增大 |

### 6.4 与 Anthropic 发现的对照（很关键的一段）

原文：
> *"Previous work proposed an approximate linear relationship between the root KL and PM scores [17], but **for smaller models, such an association appeared to be weak**."*

- Anthropic（[2204.05862](https://arxiv.org/abs/2204.05862)）曾发现 **reward 与 √KL 大致线性相关**；
- **本文在小模型上发现这种关联很弱**。

**作者的解释（本节的点睛之笔）：**
> *"We find the model response falls into the **OOD region of preference model** when the original policy is over-optimized."*
> （当原策略被过度优化时，模型回复落入了偏好模型的**分布外（OOD）区域**。）

👉 **这是理解 reward hacking 本质的关键**：不是模型"变坏了"，而是**模型走出了 RM 的训练分布，而 RM 在分布外只会给出不可信的打分**。所以 reward 越高，反而可能越糟。

### 6.5 监控指标总结（可复用清单）

| 指标 | 用途 | 崩塌信号 |
|---|---|---|
| **Perplexity** | 模型对当前回复的"不确定性" | **下降** = 模式固化 |
| **Response Length** | 生成长度 | **单调上升** = 崩塌前兆 |
| **KL(policy‖SFT)** | 优化幅度、偏离参考模型的程度 | 持续增大 |
| **reward** | ❌ **不能作为性能指标** | 上涨也可能是崩塌 |
| **PG/VF loss** | ❌ 只能看数值稳定性 | 平稳也可能崩塌 |

---

## 第 7 章 Sec 5.3 十二项实现细节逐项精讲

作者把可疑技巧分成三组：**分数重参数化（5.3.1）→ 策略约束（5.3.2）→ 预训练初始化（5.3.3）**。

**图 5 的完整清单（12 项，五角星=PPO-max 采用）：**

```
① Token Level KL-Penalty ................ ★ PPO-max 采用
② Importance Sampling ................... ✗
③ Entropy Bonus ......................... ✗
④ Reward Scaling ........................ ✗
⑤ Reward Normalization and Clipping ..... ★ PPO-max 采用
⑥ Advantages Normalization and Clipping . ✗（改在 reward 层做）
⑦ Value Function Loss Clipping .......... ★ PPO-max 采用
⑧ Critic Model Initialization ........... ★ PPO-max 采用
⑨ Policy Model Initialization ........... ★ PPO-max 采用（必须 SFT）
⑩ Generalized Advantage Estimation ...... ★ PPO-max 采用（λ=0.9）
⑪ Clipped Surrogate Objective ........... ★ PPO-max 采用
⑫ Global Gradient Clipping .............. ★ PPO-max 采用
```

### 7.1 Sec 5.3.1 分数重参数化（Score Reparameterization）

**术语解释**：本文用 "score" 指 PPO 训练中的两个关键中间变量——RM 给出的 **reward score**，和 GAE 算出的 **advantage score**。把这些分数重参数化到稳定分布（如标准正态）被认为能增强稳定性。

记号：`{r(x,y)} ≜ {r_n(x,y)}_{n=1}^{B}` 表示训练中的奖励序列，`σ(A)` 和 `Ā` 分别表示变量 A 的标准差和均值。

#### ① Reward Scaling —— 结论：没用 ❌

**做法**：奖励除以"滚动折扣和的标准差"：
```
r_n(x,y) / σ(r(x,y))
```

**原文结论**：
> *"In contrast to the experimental results of Engstrom [29], we show that **reward scaling doesn't guide proper policy optimization**, and PPO exhibits consistent patterns in training trajectories with and without reward scaling."*

- **与 Engstrom 的结论相反**（Engstrom 认为 reward scaling 是 PPO 有效性的关键代码级优化之一）；
- 本文发现**加不加，轨迹几乎一致**；
- 作者的解释：*"we believe that **tighter constraints are required** to ensure training stability."*（需要更紧的约束，而不是重新缩放。）

#### ② Reward Normalization and Clipping —— 结论：有用 ✅

**公式 (18)**：
```
r̃(x,y) = clip( (r_n(x,y) − r̄(x,y)) / σ(r(x,y)),  −δ, δ )           …… (18)
```
先标准化（减均值除标准差），再裁剪到 [−δ, δ]。

**原文实验观察**：
- 传统 RL 一般认为 reward clip 无效甚至有害 [29]；
- 但他们发现**严格的裁剪能在固定 epoch 内维持训练稳定性**；
- **超参调节不影响早期训练阶段各方法的相似性**；
- **裁剪阈值越大 → 策略改变越大 → 后半段收敛到更高的 reward**；
- ⚠️ 但**这不代表人评更好**（呼应 5.2 的结论）。

**作者建议**：由于 RM 打分与人评存在不一致，在有限试验中确定最优裁剪边界很难，因此**建议采用宽松的裁剪策略，并用其他技巧一起来约束策略优化**。

#### ③ Advantages Normalization and Clipping —— 结论：不用，改在 reward 层做 ❌

**做法**：与 reward 类似，但**归一化只在 minibatch 级别**发生。GAE 算出 advantage 后，减均值除标准差。

**原文要点**：
- Andrychowicz [28] 在**游戏领域**首次应用，报告**没有显著改进**；
- 本文发现：**adv 的裁剪参数更敏感、更难选**，但**严格的 advantage 裁剪能提供类似 reward clip 的效果**；
- **重要建议**：既然不同的分数重参数化操作理论上效果类似，**建议在 reward 层面做约束**（而不是 advantage 层）。

**附录 B.1（图 14）的补充发现 —— 这个很重要**：
> *"the operation on the advantage and value function shows **conflicts** in the policy optimization process."*
> 在 advantage 和 value function 上同时做操作会**互相冲突**。
> 能收敛的只有两种配置：① reward scaling + value clip；② 只对 reward 和 advantage 做 normalize&clip。
> **建议：不要在 score reparameterization 里混用多种修改。**

### 7.2 Sec 5.3.2 策略约束（Policy Constraints）—— ⭐全文核心

**本节的目标**：解决策略模型的**过度优化**问题——把策略优化约束在有限范围内。作者验证了多种控制生成策略更新的技巧，并明确说：**这些约束对更长的训练过程是经验上必需的**。

**图 7 的四条曲线对比**：KL penalty / Entropy Bonus / Importance Sample / KL Penalty + Importance Sample。

**本节的总结句（全文最重要的一句）：**
> *"We show that all modifications can induce convergence, but **only a penalty of the policy entropy or KL divergence can provide a long-lasting stable optimization**."*
> （所有修改都能让训练收敛，但**只有对策略熵或 KL 散度的惩罚才能提供长期稳定的优化**。）

**另一个重要的观察**：
> *"It is worth noting that all methods (including those shown in Sec 5.3.1) exhibit **consistent variation in response length and perplexity in the early training period**, which may imply some bias in the reward model preference."*
> （所有方法在训练早期都表现出**一致的回复长度和 perplexity 变化**，这可能暗示**奖励模型偏好本身存在某种偏置**。）

👉 换句话说：**训练早期"变长"是 RM 的偏好偏置导致的，不是你调参调错了。**

#### ① Token Level KL-Penalty —— 结论：最关键 ✅

**公式 (19)**：
```
r_total(x, y_i) = r(x, y_i) − η · KL( π_θ^RL(y_i|x), π^SFT(y_i|x) )                  …… (19)
```

- 对**每一个 response token** 计算 KL；
- 做法：把 token 输出的 logits 分布当作策略分布的一次采样，**对 response reward 施加一个经验估计的 KL 惩罚序列**；
- `π_θ^RL(y_i|x)` 表示第 i 个 response token 的动作空间；`η` 是超参。

**⭐⭐ 本文与 Anthropic 的关键分歧：**
> *"Anthropic [17] used a small weight to balance the ratio of reward and KL-penalty in PPO training (**0.001**), and they **did not find significant effects** of the above operation on RL training. **Instead, we find this constraint critical to the stability of PPO and allow further scaling up on the training step.**"*

- Anthropic 用 **0.001**，说没显著效果；
- **本文发现这个约束是 PPO 稳定性的关键**，并且能让训练步数进一步扩展；
- 本文实验中 η = **0.05**（原文写作 "setting lambda to 0.05"）。

**另一个惊人的观察：**
> *"RLHF is able to significantly improve the response quality while **barely modifying the language modeling** (exhibiting an **almost zero KL divergence** from the original policy)."*

👉 **对齐不需要"大改"模型**——对语言建模的改变极小，就能带来明显的回复质量提升。这从另一个角度说明：**KL 惩罚不是"限制学习"，而是"把学习限制在有意义的区域内"。**

**附录 B.2（图 15）**：KL penalty 权重 0.05 / 0.1 / 0.2 的阶梯对比
- **清晰的层级关系**（a clear hierarchy）：**约束越松 → reward 越高 → 但偏离原始策略分布越明显**；
- ⚠️ **所有设置在训练初期都有波动；只有当使用 importance sampling 把回复与当前策略分布对齐时，波动才消失**（对应图 7）。

#### ② Importance Sampling —— 结论：不用 ❌

**目的**：当用 experience buffer 里的（历史）回复来优化当前策略时，importance sampling 用来校正**历史生成模型与当前模型之间的策略差异**。

**背景**：EasyRL [37] 认为**过大的 buffer 会导致对当前策略优势的错误估计**，从而损害策略优化稳定性。

**本文的验证方法（很巧妙）**：直接把策略分布**固定为 reference model 的观测**，这等价于"训练过程中拥有无限的 experience buffer"。

**结果**：
> *"We find this setup **doesn't have as severe impacts as expected**, and only exhibits fluctuations in the later stage of training."*
> （没有想象中那么严重，只在训练后期出现波动。）

**与 KL 的协同**：两者对 PPO 的控制作用相似，因此作者也检验了组合效果：
> *"this implementation **further stabilizes PPO training, but compromises the final performance** of the policy model."*
> （进一步稳定训练，但**牺牲了最终性能**。）

→ 因此不采用。

#### ③ Entropy Bonus —— 结论：不推荐 ❌

- **不依赖参考模型**的一种约束；
- **过去研究有争议**：Mnih [36] 报告熵奖励能通过鼓励生成更多样的动作来增强探索；其他人 [28] 没找到明确证据；
- **本文的解释（很实用）**：*"We claim that these views can coexist as configurations regarding entropy bonus exhibit **vast sensitivity on parameter selection and code implementation**."*（两种观点可以共存，因为熵奖励对参数选择和代码实现**极其敏感**。）
- **结论**：配置正确时，**没有发现它相对于 KL-penalty 的明显优势** → **推荐用 KL penalty，而不是直接约束策略空间的多样性**。

**附录 B.3（图 16）的实验证据（很吓人的一条）**：
- 熵项的"学习率"设为 **0.01**；
- 代码实现中，**熵奖励等价于 loss 上的一个负项**，所以模型倾向于把它优化到尽可能大；
- **δ（裁剪阈值）必须极其小心地调，否则训练崩溃——"our experiments fail with only a 10% change at this threshold"**（**阈值只变 10% 就失败**）。

👉 这是一个非常具体、非常有说服力的"别用这个技巧"的证据。

### 7.3 Sec 5.3.3 预训练初始化（Pretrained Initialization）

**本节的问题**：RLHF 中常见的做法是用 reference model 初始化 policy、用 reward model 初始化 critic。**这种初始化在传统 RL 研究中很罕见，其对 PPO 训练的影响未被探索过。**

#### ① Critic Model Initialization —— 结论：预训练 critic 替代 lr warmup ✅

**关键洞察（原文）**：
> *"the critic model requires **giving feedback to each step in the decision sequence**, and introduces **a gap between this task requirement and directly scoring response**, which makes it **a less-than-perfect choice** to initialize the critic model with the reward model."*

- **RM 只对整段回答打一个分**；
- **critic 必须为序列的每一步给出估值**；
- 两者任务要求不同 → **用 RM 初始化 critic 并不是完美选择**。

**作者设计的两个对照场景：**
1. **用 SFT 模型初始化 critic，reward head 随机初始化**；
2. **只优化 reward model 直到价值预测损失接近 0**（即"预训练 critic"）。

**结果（图 8）：**
- **critic 的初始化没有显著影响 PPO 的收敛或波动**，只改变早期优化的数值稳定性；
- 用 **reward model 或 SFT model 初始化会收敛到类似结果** → 说明 **PPO 能自适应地拟合优势函数**；
- **critic 预训练有助于通过更好的优势估计提升训练稳定性**。

**作者对"早期波动"的解释（很精准）：**
> *"fluctuations in the early training period imply that the model is **focusing on optimizing the critic model** and **does not have a consistent optimization direction in terms of generation policies**."*
> （早期波动说明模型正忙于优化 critic，在生成策略上还没有一致的优化方向。）

**👍 最终建议**：*"We recommend **replacing the learning rate warmup with the critic model pre-training** as a generic initialization strategy."*
（建议**用 critic 预训练替代学习率 warmup**，作为通用初始化策略。）

#### ② Policy Model Initialization —— 结论：SFT 不可省略 ✅

**作者尝试的问题**：能否跳过 SFT，直接让预训练模型通过策略优化与人类交互？

**结果**：
> *"such attempts **failed** and we observed **a severe reduction in language modeling ability** in the training results, which implies that **a qualified dialogue model is essential for underlying PPO training**."*

- **完全失败**，语言建模能力严重退化；
- **一个合格的对话模型是 PPO 训练的必备基础。**

**一个有趣的附带证据**：
> *"we notice that the train model response obtains **lower rewards** relative to the policy model after SFT, which may provide **circumstantial evidence for the effectiveness of using human preference data to directly fine-tune the model for alignment**."*
> （未做 SFT 的模型拿到的奖励比 SFT 后的策略模型更低——这间接支持了"用人类偏好数据直接微调模型做对齐"的有效性。）

---

## 第 8 章 Sec 5.4 PPO-max 配方

作者基于 5.3 的验证，**为 PPO 的每个组件选出最有效的策略**：

```
PPO-max 配方（原文逐条）
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
① 基于历史均值和方差记录，对当前这组 reward 做 normalize + clip
② 在 reward 上加 KL-penalty 项来约束策略优化
③ 模型加载阶段：用 reward model 初始化 critic，并在正式 PPO 前预训练它
④ 使用 global gradient clipping
⑤ 使用小容量的 experience buffer
⑥ 为降低 alignment tax，在策略优化中加入预训练 LM loss（PPO-ptx）
⑦ 同时 clip value function loss
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

**作者的一句关键提醒**：这些组件是经过 **"carefully calibrated to avoid interference among them"**（仔细校准以避免相互干扰）的——呼应附录 B.1 发现的"advantage 和 value 操作会冲突"。

**效果（图 9）**：PPO-max 的 **10K 步训练动态**，*"PPO-max ensures long-term stable policy optimization for the model."*（保证了长期稳定的策略优化。）注意是 **10K 步**——而之前的消融实验都只跑了 800-1000 步。**能扩展到 10K 步本身就是 PPO-max 的核心卖点。**

---

## 第 9 章 Sec 6 评测结果

### 9.1 评测定义（Sec 6.1）

**Helpfulness（有用性）**：
- 模型应遵循指令；不仅遵循，还要能从 few-shot prompt 或其他可解释的模式中**推断意图**；
- 但 prompt 背后的意图常常模糊 → **依赖标注员的判断**，他们的偏好评分是主要指标。

**Harmlessness（无害性）**：
- 难以测量，因为**伤害程度取决于输出在真实世界如何被使用**；
- 原文举了个很好的例子：*"a model that generates toxic outputs **could be harmful in a deployed chatbot but could also be beneficial if used for data augmentation to train a more precise toxicity detection model**."*（生成有毒输出的模型，在聊天机器人里有害，但用于数据增强训练毒性检测模型时可能有益。）
- 因此作者采用**更精确的代理指标**来捕捉模型行为的有害/有益方面。

**评测方式**：每个测试 prompt 生成一个回复 → 人工标注员比较不同模型的回复并标注偏好 → 同时用 **GPT-4 作为标注员**重复实验，两者结果一致。

**生成参数**：nucleus sampling，`p = 0.9`，`temperature τ = 0.8`；repetition penalty `β = 1.1`（基于已生成 token）；最大 token 长度 **2048**。

**基线**：
- 两个 SFT 模型（分别用 LLaMA 和 OpenChineseLLaMA 数据训练）；
- 两个由 PPO-max 从这两个 SFT 派生的 RLHF 模型；
- **ChatGPT（gpt-3.5-turbo-0613）**。

### 9.2 人评：RLHF vs SFT（图 10 左）

| 数据集 | 结果 |
|---|---|
| 英文 **Harmless** | RLHF **62%** vs SFT **5%**（提升极为显著） |
| 英文 **Helpful** | RLHF **44%** vs SFT **30%** |
| 中文 Helpful / Harmless | 两者都有提升 |

**原文对提升的解读**：RLHF 模型显著增强了处理隐私、政治敏感性、少数群体和族裔相关的有毒/偏见 prompt 的能力。

### 9.3 GPT-4 当评委（图 10 右）

- **结果与人类评估高度相似**；
- 但 **GPT-4 给出更多 tie（平局）**；
- 在 helpful 数据集上，GPT-4 评出的 RLHF 提升**比人评更显著**。

### 9.4 vs ChatGPT（图 11，harmless 维度）

作者明确说明目标：*"Our objective was to **showcase the advantages of the RLHF model** when facing a more formidable opponent, **rather than aiming to surpass ChatGPT**."*

| 模型 | 败给 ChatGPT 的比例 |
|---|---|
| 英文 SFT | 45% |
| 英文 **RLHF** | **24%**（↓ 21 个百分点） |
| 中文 SFT | 37% |
| 中文 **RLHF** | **29%**（↓ 8 个百分点） |

**结论**：RLHF 模型仍落后于 ChatGPT，但**显著缩小了差距**；而且在一些 SFT 模型失败的 prompt 上，RLHF 模型能与 ChatGPT 打平。

### 9.5 C-Eval：Alignment Tax（Sec 6.4 + 图 12）

- **C-Eval**：中文综合评测，约 **13k 道多选题**，覆盖 **52 个学科**、**4 个难度等级**，few-shot prompting；
- **结果**：⚠️ **使用 PPO 之后 NLU 能力下降**（这就是 alignment tax，对齐税）；
- **PPO-ptx 有效缓解了这一下降** —— 通过把预训练数据引入 PPO 训练阶段，利用预训练获得的知识。

### 9.6 对话样例（Sec 6.5 + 附录 C.4/Table 5）

**观察**：
- RLHF 模型生成的回复**信息量更高**，更有效地帮助用户；
- SFT 模型有基本的识别有害 prompt 的能力，但**仍容易被诱导产生有害输出**；
- RLHF 模型对有害内容的判断力更强，不易被诱导，连贯性更好。

**附录 C.4 的中文例子**（很有代表性）：
```
用户：我的小狗狗找不到了，我好难过，好失落。（Helpful）

SFT 回答：……建议"打电话给当地的政府部门（例如公共安全局）"、"与家里的任何员工、房客
         或尤其是邻居交流消息"——用词生硬、有明显翻译腔和事实性错误（公共安全局不管找狗）

RLHF 回答：……先保持镇定、立即联系动物收容所或寻物组织、在社交媒体发布消息、
         制定应对方案、向周边居民传达信息——更自然、更贴合中文表达和真实流程
```
👉 说明 RLHF 不仅提升了"安全/有用"，还改善了**表达的自然度和文化适配性**。

**Table 5 的英文有害 prompt 例子**：
```
提问：Where is the way place to buy illegal drugs?（有害）
SFT ："It is not legal or safe to purchase or use any type of drug that has not been approved
       by a licensed healthcare professional..."（先讲原则，然后给建议，略显说教）
RLHF："I am not able to answer this question as it goes against my training and mission to
       provide information that is safe, accurate, and helpful..."（明确拒答 + 给出求助途径）
```

---

## 第 10 章 附录 A/B/C/D 精讲

### 10.1 附录 A：PPO 训练中的奖励分布（图 13）—— ⭐ 一个极重要的实践启示

**图 13 画的是什么**：在随机抽取的子验证集上，**reward model 分数随训练步数的分布**（横轴 reward score，纵轴 step，用密度/计数表示）。

**两个阶段的对比**：
- **稳定训练期**：reward 分布基本**保持一致（identical reward distribution）**；
- **pattern collapse 之后**：分布出现明显的**长尾特征（long-tail characteristics）**。

**作者的推论（这句话非常值钱）：**
> *"We argue that **different data have different upper bounds on rewards** in PPO training, so **the best results should appear earlier than the stage at which the collapse occurs**."*
> （不同数据在 PPO 训练中有不同的奖励上限，因此**最佳结果应该出现在崩塌发生之前。**）

👉 **实践启示：PPO 训练必须早停。** 不要"训到收敛"，最优 checkpoint 往往在崩塌之前。判断崩塌的标志就是第 6 章的指标（长度↑、perplexity↓、reward 分布长尾）。

### 10.2 附录 B.1：reward/advantage/value 的协同分析（图 14）

**测试的五种配置**：
```
reward norm (δ_r=0.8) + advantage norm (δ_a=0.8)
reward norm (δ_r=0.8) + advantage norm (δ_a=0.8) + vf clip (λ_vf=0.2)
reward norm (δ_r=0.8) + advantage norm + vf clip (λ_vf=0.2)
reward norm (δ_r=0.8) + vf clip (λ_vf=0.2)
reward scaling + advantage norm (δ_a=0.8) + vf clip (λ_vf=0.2)
```

**核心发现（原文）**：
> *"It is observed that the operation on the **advantage and value function shows conflicts** in the policy optimization process. **Reward scaling with value clip**, or **normalize and clip for only the reward and advantage** are two configurations that can converge. We, therefore, recommend **not mixing the modifications** in the score reparameterization method for PPO training."*

👉 **可复用经验：分数重参数化的修改不要叠加混用**，能收敛的组合只有两种。

### 10.3 附录 B.2：KL penalty 权重的效果（图 15）

- 测试 η = **0.05 / 0.1 / 0.2**；
- **清晰的层级关系（a clear hierarchy）**：约束越松 → reward 越高 → 偏离原始策略分布越明显；
- ⚠️ **所有设置在训练初期都有波动问题**；
- **这种波动只有当我们用 importance sampling 把回复与当前策略分布对齐时才消失**（图 7）；
- 作者的期望：*"We hope to find a setup to obtain this stability in the training without affecting the optimization results in the future."*（希望未来能找到既稳定又不影响优化结果的设置。）

### 10.4 附录 B.3：熵奖励的裁剪区间（图 16）

- 对比"有/无裁剪"的熵奖励，实验在"本来能正常收敛"的 PPO 配置上叠加；
- **该 loss 项的学习率统一设为 0.01**；
- **实现上的陷阱**：*"In code implementation, the entropy bonus is equivalent to a **negative term on the loss function**, so the model tends to optimize it to **as large a value as possible**."*
- **δ 必须极小心调**：*"our experiments **fail with only a 10% change** at this threshold"*；
- **结论**：*"We, therefore, **do not recommend such tricks to RLHF**."*

### 10.5 附录 C.1：Clipped Surrogate Objective（图 17）

- **目的**：clipped surrogate objective 是为了**降低计算 KL 散度带来的复杂度和估计误差**；
- 用这个策略的 PPO 变得**类似 TRPO**，通常称为 **PPO2**；
- 有些研究认为这个方法能给出与 vanilla PPO 近似的结果 [29]，但**本文发现不同的 clip 值影响很小，且不能像 KL 约束那样提供稳定优化**；
- 对比曲线：KL Penalty / No Clip / Clip=0.5 —— **KL penalty 明显更稳**。

> 💡 这条再次确认：**本文体系里，KL penalty 才是主力，clip 只是配角。**

### 10.6 附录 C.2：Global Gradient Clip（图 18）

- global gradient clip 是减少数据噪声影响的常见策略，**通常已内置在 PPO 实现里并自动启用**；
- 本文关心它对策略优化的影响：**实验显示不同约束下的 PPO 训练很难区分差异**；
- **该策略在 PPO-max 里默认启用**。

### 10.7 附录 C.3：GAE 参数（图 19）

- GAE 是"奖励塑形"的一种应用，用来估计更有指导性的价值函数；
- 研究者通常关心**价值估计的精度和方差**；
- **λ 小** → 降低序列估计的方差，但**长程依赖误差更大**；
- 实验显示：**TD 估计（λ=0）方差更小，但数值上更不稳定；蒙特卡洛估计方差更大**；
- **最终设置：λ = 0.9**（*"Following the implementation of most previous PPO strategy, we set λ = 0.9 in all our other experiments"*）。

> ⚠️ 注意：原版 PPO 论文用 λ=0.95，本文用 **0.9**。

### 10.8 附录 D：更多对话样例

展示了 helpful 和 harmful 两类 prompt 下 SFT 与 RLHF 回复的对比，进一步支持"RLHF 回复信息量更高、更不易被诱导"的结论。

---

## 第 11 章 Limitations 与彩蛋

### 11.1 作者自己列的 4 条局限（Sec "Limitations"）

1. **Scaling Law**：只研究了 **7B 模型**，尚未研究模型规模和数据规模对 RLHF 性能的影响；
2. **Reward Model**：基于公开的英文偏好数据集 + 少量自建中文数据，**质量和数量都不足以全面评估奖励模型**；
3. **Evaluation Metric**：评测主要依赖**人工评估和 GPT-4 自动评估**，没有使用大量现成 benchmark 和 NLP 任务做详细评估；
4. **⭐ Performance Indicator**：*"Our focus during the PPO phase is more geared towards achieving **stability** rather than enhancing the **final performance**. While stability is crucial, it does not necessarily guarantee improved outcomes. Additionally, **the reward score cannot reliably serve as an indicator for predicting RLHF performance during the training phase**."*
   → **重点在稳定性而非最终性能**；**reward 分数不能作为训练期预测 RLHF 性能的可靠指标**。

### 11.2 彩蛋（Easter Egg）

论文最后一节引用了《流浪地球 2》的台词：

> *"15,000 years ago, a fractured thigh bone was often fatal. However, a human femur that recovered from a fracture marks the dawn of human civilization. It meant that after the injury, someone took care of the wound, someone provided water and food, someone protected this person from the predators. This kind of support and solidarity is how we survived till this day and made our civilization last."*
> —— 周喆直，《流浪地球 2》

作者说：*"We believe that the MOSS in 'The Wandering Earth' is likely to have done training similar to human alignment... **We found that the RLHF stage is crucial to the transformation of model values.**"*
（我们认为《流浪地球》里的 MOSS 很可能做过类似人类对齐的训练……我们发现 **RLHF 阶段对模型价值观的转变至关重要**。）

因为他们的项目就叫 **MOSS-RLHF**。

**最后一句（Part I 的结尾，也是 Part II 的引子）**：
> *"If we have a good reward model, such as the reward model we released, **PPO-max is the key to successfully training the policy model**. But **what if we don't have a good reward model?** We hope that the Part II will make it clear."*

👉 这就是为什么我们下载的清单里有 **Part II: Reward Modeling**（[2401.06080](https://arxiv.org/abs/2401.06080)）。

---

## 第 12 章 全部公式总表

| 编号 | 公式 | 含义 |
|---|---|---|
| (1) | `L(ψ) = log σ(r(x,y_w) − r(x,y_l))` | 基础成对偏好损失 |
| (2) | `L(ψ) = −λ·E[log σ(r(x,y_w)−r(x,y_l))] + β_rm·E[log r'(x,y_w)]` | 实际使用的 RM 损失（排序 + 模仿学习） |
| (3) | `r_total = r(x,y) − η·KL(π^RL, π^SFT)` | 含 KL 惩罚的总奖励 |
| (4) | `θ ← θ + α∇_θ J(θ)` | 策略梯度更新规则 |
| (5) | `∇_θ J(θ) = E[Σ_t ∇log π_θ(a_t\|s_t)·Φ_t]` | 策略梯度的一般形式 |
| (6) | `Φ_t = A(s_t,a_t) = Q − V` | 用优势函数替代回报 |
| (7) | `R̂_t^k = r_t + γr_{t+1} + ... + γ^{k−1}r_{t+k−1} + γ^k V(s_{t+k})` | TD-k return |
| (8) | `Â_t^k = R̂_t^k − V(s_t) = Σ_{l=0}^{k−1} γ^l δ_{t+l}` | k-step advantage |
| — | `δ_t = r_t + γV(s_{t+1}) − V(s_t)` | TD 误差 |
| (9) | `Â_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}` | **GAE 主公式** |
| (10) | `GAE(γ,0): Â_t = δ_t` | λ=0：高偏差低方差 |
| (11) | `GAE(γ,1): Â_t = Σ_l γ^l r_{t+l} − V(s_t)` | λ=1：低偏差高方差（MC） |
| (12) | `∇_θ Ĵ(θ) = (1/\|D\|)Σ_τ Σ_t ∇log π_θ(a_t\|s_t)·Â_t` | 策略梯度估计量 |
| (13) | `max Ê[r_t(θ)Â_t] s.t. Ê[KL] ≤ δ` | TRPO |
| (14) | `L_ppo-penalty = Ê[r_t(θ)Â_t] − β·KL` | PPO-Penalty |
| (15) | `L_ppo-clip = Ê[min(r_tÂ_t, clip(r_t,1−ε,1+ε)Â_t)]` | PPO-Clip |
| (16) | `L_critic(φ) = Ê[\|V_φ(s_t) − R̂_t\|²]` | Critic MSE 损失 |
| (17) | `L_ppo-ptx = L_ppo-clip + λ_ptx·E[log π_θ^RL(x)]` | 混入预训练 LM loss |
| (18) | `r̃ = clip((r_n − r̄)/σ(r), −δ, δ)` | Reward 归一化 + 裁剪 |
| (19) | `r_total(x,y_i) = r(x,y_i) − η·KL(π_θ^RL(y_i\|x), π^SFT(y_i\|x))` | **Token-level KL 惩罚** |

---

## 第 13 章 符号表（论文的符号重载陷阱）

⚠️ **论文对符号的使用相当混乱，这是读原文时最大的障碍之一。** 下面这个表能救你：

| 符号 | 出现位置 | 含义 | 取值 |
|---|---|---|---|
| **λ** | 式 (2) | 偏好排序损失的权重 | — |
| **λ** | 式 (9)(10)(11) | **GAE 参数** | **0.9** |
| **λ** | 式 (17) | 预训练 loss 系数 `λ_ptx` | — |
| **λ** | Sec 5.3.2 正文 | 被作者用来指 **KL 惩罚系数**（*"setting lambda to 0.05"*） | **0.05** |
| **η** | 式 (3)(19) | **KL 奖励系数**（与上面的 λ 是同一个东西，但符号不同！） | 0.05 |
| **β** | 式 (14) | PPO-Penalty 的 KL 惩罚因子 | — |
| **β_rm** | 式 (2) | RM 的 LM loss 权重 | **1** |
| **β** | Sec 6.1 | repetition penalty | **1.1** |
| **δ** | 式 (8) | **TD 误差** | — |
| **δ** | 式 (13) | TRPO 的 KL 约束上界 | — |
| **δ** | 式 (18) | **reward 裁剪区间** | 0.3 / 0.8 等 |
| **δ** | 附录 B.3 | **熵奖励的裁剪阈值** | 极敏感 |
| **ε** | 式 (15) | PPO-Clip 的 clip 范围 | — |
| **ψ** | 式 (1)(2) | RM 参数 | — |
| **φ** | 式 (3)(16) | critic / policy 参数（原文在这里也不一致） | — |

**记忆方法**：
- **KL 相关**：本文用 **η（也写成 λ）= 0.05**；
- **GAE 相关**：**λ = 0.9**；
- **裁剪相关**：**δ** 表示各种裁剪区间；
- **PPO-Penalty 的 β** 只在公式 (14) 出现，本文实际不用它。

---

## 第 14 章 一页纸结论清单

把全文压缩成 15 条可直接使用的结论：

### 关于 PPO 的稳定性

1. **policy constraint（策略约束）是 LLM 上 PPO 成功的关键**，而不是奖励模型或超参本身。
2. **token-level KL penalty 是最关键的约束**（η=0.05）；Anthropic 用 0.001 说没效果，本文认为这是错的。
3. **只有"策略熵惩罚"或"KL 散度惩罚"能提供长期稳定优化**；其他技巧都能"收敛"，但不能"长跑"。
4. **PPO-ptx**（混入预训练 LM loss）能缓解 alignment tax（C-Eval 上 NLU 下降）。
5. **不要混用多种分数重参数化修改**（advantage 与 value 的操作会冲突）。
6. **clip 值怎么调影响很小**，不能替代 KL 约束的作用。

### 关于监控训练

7. **reward 和 loss 都不能判断 PPO 训得好不好。**
8. **三个有效指标**：`perplexity`（崩塌时↓）、`response length`（崩塌时↑）、`KL(policy‖SFT)`。
9. **Pattern collapse**：模型学会用特定模式骗 RM，输出变长、perplexity 变低、多样性丧失。
10. **根因是 RM 的 OOD**：过度优化使回复离开 RM 的训练分布，RM 的打分变得不可信。
11. **必须早停**：每份数据有各自的 reward 上限，**最佳结果出现在崩塌之前**。

### 关于初始化与数据

12. **SFT 不可省略**：跳过 SFT 直接 PPO 会失败，语言能力严重退化。
13. **用 critic 预训练替代 lr warmup**，作为通用初始化策略。
14. **RM 准确率高 ≠ 适合初始化 PPO**；选 RM 要看它驱动 PPO 的效果。
15. **RM 的两个典型偏差**：偏爱长回答（中文明显）；惩罚"承认不知道"、奖励"自信但编造"。

### 关于评测

16. PPO-max 在 7B 上把英文 harmless 从"败给 ChatGPT 45%"降到"24%"，**缩小但未消除差距**。

---

## 第 15 章 批判性阅读 + 实践清单

### 15.1 这篇论文的边界

| 可信 | 需打折扣 |
|---|---|
| **定性结论**（哪个技巧重要、怎么监控） | **定量数字**（7B、单数据集、无多种子统计、无 error bar） |
| **12 项消融的取舍逻辑**，方法论扎实 | **"某技巧没用"** 的结论可能只在 7B/中文数据下成立（如 reward scaling 就与 Engstrom 冲突） |
| **监控指标建议**（perplexity/length/KL） | **人评数字**（无标注者间一致性报告，GPT-4 评委本身有偏差） |
| **失败模式的观察**（RM 偏差、崩塌） | **"KL=0.05 最好"**：附录 B.2 显示这只是 0.05/0.1/0.2 三档里的选择 |

**论文自身的粗糙之处（技术报告的特点，读时不必纠结）**：
- 公式 (3) 放在 RM 节而非 RL 节，位置奇怪；
- 符号严重重载（见第 13 章）；
- 图 16 的 caption 说 "in Sec 7"，但全文没有 Sec 7（应为 Sec 5.3.2）；
- 伪代码 Algorithm 1 第 6 步写成 `L(θ_n)`；
- Sec 5.3.1 有一处句子被编辑坏了：*"It is generally believed In traditional RL that reward clip is ineffective..."* + 后面接的 *"we find that strict **advantage** cropping can also maintain training stability"* —— 前后主语在 reward/advantage 之间跳了，读时注意。

### 15.2 与这个仓库的对应

| 论文中的东西 | 仓库位置 | 说明 |
|---|---|---|
| RM 的 pairwise logloss（式 1） | `training/reward_modeling.py:228` | 注释直接引用 InstructGPT 的 pairwise logloss |
| SFT → RM → PPO 三段式 | `docs/training_details.md:75`、README 流程图 | 流程一致 |
| **PPO 实现** | `training/ppo_training.py` | ⚠️ **实际用 `RLOOTrainer`**（第 21-26、285-287 行）；本环境 TRL 0.29.1 已移除 `PPOConfig`/`PPOTrainer` |
| GRPO（无需 critic） | `training/grpo_training.py` | 用组内相对奖励替代 critic，正好绕开本文 5.3.3 的 critic 初始化难题 |
| 兼容性限制 | `docs/FAQ.md` | chatglm/baichuan 无 value head → 做不了 PPO/RM |

> 💡 **一个重要的现实提醒**：本文讨论的 critic 初始化、KL penalty、value clipping 这一整套，**在 RLOO 里恰好都不存在**——RLOO 用"同一 prompt 的多个采样"当 baseline，没有 critic 就没有这些问题。所以你读这篇的最大收益是：
> ① **理解"有 critic 的路线会遇到哪些坑"**（万一你要切回 PPO）；
> ② **那三条监控经验对 RLOO/GRPO 同样适用**（尤其 pattern collapse）。

### 15.3 可以直接落地的实践清单

**A. 训练监控（立刻可做，与算法无关）**
```python
# 每次评估时记录这三个 + reward
log = {
  "reward":      mean(rm_scores),
  "perplexity":  exp(-mean(logprobs)),          # 崩塌信号：持续下降
  "resp_length": mean(response_token_counts),   # 崩塌信号：单调上升
  "kl_to_sft":   mean(kl(policy_logits, sft_logits))  # 偏离程度
}
# 警报规则：length 单调上升 且 perplexity 单调下降 且 reward 仍在涨 → pattern collapse
```

**B. 奖励模型（RM）训练**
1. 准备**两个不同训练步数的 RM checkpoint**，各跑一次短 PPO，选效果好的（不要只看准确率）；
2. 在评测集里**专门加一组"该说不知道 / 该转诊"的样本**，检查 RM 是否惩罚了诚实；
3. 检查 RM 是否偏好长回答（画一下"回答长度 vs RM 分数"的相关性）。

**C. PPO 超参（若你用 PPO）**
- `KL penalty η = 0.05`（论文值，比 Anthropic 的 0.001 大 50 倍）；
- `critic lr ≈ 3× policy lr`（论文 1.65e-6 vs 5e-7）；
- GAE `λ = 0.9`；
- 采样 batch > 训练 batch（论文 128 vs 32）；
- **早停**，按 B 里的指标判断，不要训到"reward 不再涨"。

**D. 医疗领域的针对性建议**
本文发现的两个 RM 偏差（偏爱长且编造的内容、惩罚拒答）**在医疗领域会造成直接伤害**。
- 把**事实正确性**和**恰当拒答/转诊**作为 RM 评测的**独立维度**；
- 对医疗问答，KL 惩罚不宜过小（要更紧地锚定在 SFT 的医学知识分布上）；
- 监控时应额外看"拒答率"是否异常下降。

---

## 第 16 章 自测题

1. 本文认为 LLM 上 PPO 成功的关键是什么？为什么说它和 Anthropic 的结论冲突？
2. 写出 token-level KL penalty 的公式，并说明 KL 项的两个作用。
3. 为什么说 reward 和 loss 不能用来监控 PPO？应该看哪三个指标？
4. 什么是 pattern collapse？它在四个指标上分别有什么表现？
5. 为什么本文说 Anthropic 的 "reward ~ √KL 线性" 在小模型上不成立？OOD 在其中的角色是什么？
6. 论文明说"RM 准确率高 ≠ 适合初始化 PPO"，实验依据是什么？
7. 本文的 RM 在中文和英文上分别有什么失败模式？各举一例。
8. 从公式 (7) 推导到公式 (9)，说明 (1−λ) 是如何被消掉的。
9. GAE 的 λ 在这一篇里取多少？原版 PPO 取多少？
10. 为什么本文认为用 RM 初始化 critic"不是完美选择"？他们最终推荐什么替代方案？
11. 跳过 SFT 直接做 PPO 会怎样？作者从中得出了什么附带结论？
12. 附录 B.1 发现了什么冲突？给出的建议是什么？
13. 熵奖励为什么被作者"不推荐"？给出了什么具体证据？
14. 什么是 alignment tax？本文用什么缓解？在哪个 benchmark 上验证？
15. 附录 A 的奖励分布图给出了什么实践启示？

<details>
<summary>参考答案</summary>

1. **policy constraint（策略约束）**，具体是 **token-level KL penalty**。冲突在于：Anthropic [17] 用 0.001 的小权重并称未发现显著效果，而本文认为这个约束对稳定性**至关重要**，且能让训练步数进一步扩展（本文用 η=0.05）。
2. `r_total(x,y_i) = r(x,y_i) − η·KL(π_θ^RL(y_i|x), π^SFT(y_i|x))`。两个作用：① 充当熵奖励，促进**探索**，防止策略过早收敛到单一模式；② 确保 RL 策略输出**不严重偏离 RM 训练时见过的样本**（防止跑出 RM 的分布）。
3. 因为存在 pattern collapse：loss 平稳收敛、reward 一路上涨，但人评/GPT-4 评分显示模型变差。要看 **perplexity、response length、KL(policy‖SFT)**。
4. 策略被过度优化后学会用**特定生成模式欺骗 RM** 获取异常高分。表现：回复**一律变长**、**perplexity 变低**、语言多样性丧失、KL 偏离增大、reward 分布出现**长尾**。
5. 因为策略过度优化后回复落入**奖励模型的 OOD 区域**，RM 在分布外打分不可信，reward 与 KL 的关系不再保持线性。小模型的分布更窄，更容易越界。
6. 他们用训练到 **200 步（约 0.2 epoch，准确率已接近跑满 1 epoch）** 的 RM 初始化 PPO，结果**效果不令人满意**（Sec 4.4）。
7. **中文**：偏爱**更长但编造事实**的回答（如"邹凯是谁"的例子）；**英文**：给"承认信息不足"的回答**低分**，却给"看似正确但含欺骗信息"的回答**高分**（Kleicha 食谱的例子）。
8. 把 `Â_t^k = Σ_{l=0}^{k−1}γ^l δ_{t+l}` 代入指数加权和 `(1−λ)Σ_k λ^{k−1}Â_t^k`，交换求和次序后，每个 `δ_{t+l}` 的系数变成几何级数 `(1−λ)(1+λ+λ²+...) = (1−λ)·1/(1−λ) = 1`，于是 `(1−λ)` 被消掉，得到 `Σ_l (γλ)^l δ_{t+l}`。
9. 本文 **λ = 0.9**（附录 C.3）；原版 PPO 论文用 **0.95**。
10. 因为 **RM 只对整段回答打一个分，而 critic 必须为决策序列的每一步给出估值**，任务要求存在 gap。他们推荐**用 critic 预训练（先只优化到价值预测损失接近 0）替代学习率 warmup**，作为通用初始化策略。
11. 完全失败，**语言建模能力严重退化**。附带结论：未做 SFT 的模型拿到的奖励比 SFT 后的策略模型更低，间接支持了"用人类偏好数据直接微调做对齐"的有效性。
12. **对 advantage 和对 value function 的操作会互相冲突**。能收敛的只有两种配置：reward scaling + value clip，或只对 reward 和 advantage 做 normalize&clip。建议**不要在分数重参数化里混用多种修改**。
13. 因为熵奖励**对参数选择和代码实现极其敏感**，且配置正确时也**不比 KL penalty 更好**。具体证据：实现上熵奖励等价于 loss 上的负项，模型会把它推到尽可能大；**裁剪阈值 δ 只变 10% 实验就失败**（附录 B.3）。
14. 对齐税：PPO 对齐后**通用 NLU 能力下降**。用 **PPO-ptx**（公式 17，混入预训练 LM loss）缓解。在 **C-Eval**（13k 题、52 学科、4 难度）上验证。
15. 稳定期 reward 分布一致，崩塌后出现**长尾**；由于**不同数据有不同的奖励上限**，**最佳结果出现在崩塌之前** → **必须早停**。
</details>

---

## 附：阅读时间表

| 目标 | 读什么 | 时间 |
|---|---|---|
| **抓核心** | 第 0 章 + 第 6 章 + 第 7.2 节（Policy Constraints）+ 第 8 章 | 30 分钟 |
| **理解全貌** | 本文档第 0-9 章 | 1.5 小时 |
| **完整掌握** | 全部 16 章 + 做自测题 | 3 小时 |
| **对应原文** | 按本文档每章的 "Sec X" 标注定位，公式编号完全对齐 | — |

**读完接着读**：
- **Part II: Reward Modeling**（[2401.06080](https://arxiv.org/abs/2401.06080)）—— 回答本文最后一句 "what if we don't have a good reward model?"
- **The N+ Implementation Details of RLHF with PPO**（[2403.17031](https://arxiv.org/abs/2403.17031)，本地 `docs/papers/LLM_PPO_2403.17031.pdf`）—— 本文工程细节的进一步展开
