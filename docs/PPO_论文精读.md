# PPO 论文精读导读

> 对象论文：**Proximal Policy Optimization Algorithms**（Schulman, Wolski, Dhariwal, Radford, Klimov, OpenAI, 2017）
> arXiv: [1707.06347](https://arxiv.org/abs/1707.06347)｜PDF 本地副本：`docs/papers/PPO_1707.06347.pdf`｜纯文本：`docs/papers/PPO_1707.06347.txt`
> 全文只有 12 页，正文 8 页 + 附录超参表。**建议先读正文第 1~5 节，第 6 节实验可以后补。**

---

## 0. 一句话核心

**PPO 把 TRPO 的"信赖域约束"换成"裁剪概率比"的替代目标，于是只用一阶优化（SGD/Adam）就能对同一批数据做多轮更新，同时保住 TRPO 的稳定性。**

记住这条主线，全文所有公式都是它的展开：

```
vanilla PG：一批数据只能更新一次（多做几步就崩）
   ↓ 为什么崩？参数走远后数据分布与策略不再匹配，替代目标失效
TRPO：加信赖域约束（二阶、要算 Fisher 矩阵、共轭梯度），稳但复杂
   ↓ 能不能只用一阶？
PPO：clip(概率比) —— 一旦新旧策略偏离太多，梯度直接为 0，等于自动"刹车"
```

---

## 1. 论文身份证

| 项 | 内容 |
|---|---|
| 标题 | Proximal Policy Optimization Algorithms |
| 作者 | John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (OpenAI) |
| 时间 | 2017-07-20 提交，2017-08-28 修订（v2） |
| 领域 | cs.LG，强化学习 / 策略梯度 |
| 影响力 | 现代 RLHF 的默认算法；到 2024 年 GRPO 出现前，PPO 几乎是所有对齐流程的 RL 部分 |
| 实验场景 | MuJoCo 连续控制（7 个任务）、Roboschool 3D 人形、Atari 49 个游戏 |

注意一个学习上的坑：**论文里的"环境"是机器人和游戏，不是语言模型。** 语言模型上的 PPO（RLHF）是后来 [1909.08593](https://arxiv.org/abs/1909.08593)、[2203.02155](https://arxiv.org/abs/2203.02155) 做的适配，第 8 节专门讲这个落差。

---

## 2. 读前必备的 5 个概念（10 分钟补齐）

| 概念 | 一句话解释 | 符号 |
|---|---|---|
| **on-policy** | 只能用"当前策略自己采样出来"的数据做更新；策略一变，旧数据就得扔 | π_θ |
| **概率比 ratio** | 同一动作，新策略相对旧策略的偏好变化倍数 | r_t(θ) = π_θ(a_t\|s_t) / π_θold(a_t\|s_t) |
| **优势 Advantage** | 这个动作比"平均水平"好多少，正数=好动作，负数=坏动作 | Â_t |
| **替代目标 surrogate** | 因为真实期望回报难以估计，改用一批样本上的近似目标来优化 | L(θ) |
| **信赖域 trust region** | 限制单次策略更新的幅度，防止一步走太远把训练搞崩 | KL ≤ δ |

一个直觉比喻（贯穿全文）：

> 你在调一个旋钮（策略参数），每批数据只能告诉你"当前旋钮附近哪个方向更好"。
> vanilla PG：拧一小步就换新数据；拧多了就脱离数据的可信范围，越拧越错。
> TRPO：规定"每次最多拧 δ 度"，但需要精确测量旋钮的几何结构（二阶信息），贵。
> PPO：直接做一个"拧过头就不再给梯度"的目标函数——**过头的方向奖励被削平成常数**，你就自发地不会拧过头了。

---

## 3. 逐节精读

### 3.1 Abstract + Sec 1 Introduction（第 1 页）

要读懂三句关键话：

1. *"standard policy gradient methods perform one gradient update per data sample"* —— 传统策略梯度一批数据只更新一次，数据效率低。
2. *"a novel objective function that enables multiple epochs of minibatch updates"* —— 本文贡献：**一批数据可以反复用 K 个 epoch**（这是"数据效率"的来源）。
3. 三条对比路线各自的毛病（原文列得很清楚）：
   - **Q-learning**：带函数逼近时在很多简单问题上失效，理论不清楚；
   - **vanilla policy gradient**：数据效率差、鲁棒性差；
   - **TRPO**：相对复杂，且**不兼容含噪声的结构（如 dropout）或参数共享**（policy 和 value 共用网络、或多任务共享）。

> 💡 第 3 点是理解 PPO 动机的关键：TRPO 要用 Fisher 向量积和共轭梯度，实现上要求目标是"干净"的、可二阶展开的；一旦网络里有 dropout 的随机性或 policy/value 共享参数，就很别扭。PPO 的目标函数是**普通的可导 loss**，塞进任意架构都行。

### 3.2 Sec 2.1 Policy Gradient Methods（第 2 页）

公式 (1) 是策略梯度估计量：

```
ĝ = Ê_t[ ∇_θ log π_θ(a_t|s_t) · Â_t ]        …… (1)
```

公式 (2) 是它的"loss 形式"（为了能用 autograd 求导，把梯度写成某个目标的梯度）：

```
L^PG(θ) = Ê_t[ log π_θ(a_t|s_t) · Â_t ]      …… (2)
```

**本节最重要的一句**（原文原话）：

> *"doing so is not well-justified, and empirically it often leads to destructively large policy updates"*

即：用同一批轨迹对 L^PG 连做多步优化，**没有理论依据，实验上也常常导致破坏性的巨大更新**。作者说这种设置的效果"与完全不裁剪/不惩罚差不多甚至更差"。

👉 这就是全文的"问题定义"：**我想多轮复用数据，但直接多轮会崩，所以要改造目标函数。**

### 3.3 Sec 2.2 Trust Region Methods（第 2 页）

TRPO 的做法，公式 (3)(4) 是一个**带约束**的优化问题：

```
maximize_θ   Ê_t[ (π_θ(a_t|s_t) / π_θold(a_t|s_t)) · Â_t ]              …… (3)
subject to   Ê_t[ KL[ π_θold(·|s_t), π_θ(·|s_t) ] ] ≤ δ                 …… (4)
```

- θ_old = 更新前的参数（所以 r(θ_old)=1，是起点）。
- 约束的意思是：**平均 KL 散度不能超过 δ**，超了就说明走太远。
- 求解方式：对目标做线性近似、对约束做二次近似，然后用**共轭梯度**解 —— 这就是"二阶、复杂"的来源。

理论其实支持用**惩罚项**代替硬约束，即公式 (5)：

```
maximize_θ  Ê_t[ r_t(θ)·Â_t − β·KL[π_θold(·|s_t), π_θ(·|s_t)] ]         …… (5)
```

但作者强调：**β 极难选**（不同问题、甚至同一问题不同阶段都不一样），所以他们试过固定 β 直接用 SGD 优化公式 (5)，**效果不够好，需要额外改造** —— 这个"改造"就是第 3 节。

### 3.4 Sec 3 Clipped Surrogate Objective（第 3 页）⭐ 全文核心

先引入 **CPI 目标**（Conservative Policy Iteration，公式 6）：

```
L^CPI(θ) = Ê_t[ r_t(θ) · Â_t ]        …… (6)
```

问题：**没有约束时最大化 L^CPI 会导致过大的策略更新**。所以作者要"惩罚让 r_t 偏离 1 的变化"。

**主目标（公式 7）：**

```
L^CLIP(θ) = Ê_t[ min( r_t(θ)·Â_t ,  clip(r_t(θ), 1−ε, 1+ε)·Â_t ) ]      …… (7)
```

其中 ε 是超参，论文推荐 **ε = 0.2**。

逐项拆解公式 (7)：

| 部分 | 含义 |
|---|---|
| `r_t(θ)·Â_t`（min 的第一项） | 就是 L^CPI，未裁剪的版本 |
| `clip(r_t(θ), 1−ε, 1+ε)`（第二项内） | 把概率比硬夹到 [0.8, 1.2]，**切断了"把 r 推出这个区间"的动机** |
| `min(...)` | 取两者的**较小值** → 最终目标是未裁剪目标的一个**悲观下界（lower bound）** |

原文对 `min` 的解释是全文最关键的一句话，务必读懂：

> *"With this scheme, we only ignore the change in probability ratio when it would make the objective improve, and we include it when it makes the objective worse."*
> （我们只在概率比变化**会让目标变好时忽略它**，在它**会让目标变差时保留它**。）

还有一句技术细节：`L^CLIP(θ) = L^CPI(θ)` 在 θ_old 附近的**一阶近似**下成立（因为 r=1），但 θ 走远后两者分道扬镳——这正是我们想要的"近处正常走、远处自动刹车"。

**图 1 是全文最该盯着看的一张图**（左 A>0，右 A<0）。用数字算一遍就懂了，取 ε=0.2：

| r | 0.5 | 0.9 | 1.0 | 1.1 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| **A = +1** 时 L^CLIP | 0.5 | 0.9 | 1.0 | 1.1 | **1.2** | **1.2** |
| **A = −1** 时 L^CLIP | **−0.8** | **−0.8** | −1.0 | −1.1 | −1.5 | −2.0 |

读表结论（对应图 1 的形状）：

- **A>0（好动作）**：想增大 r，收益一路涨到 r=1.2 就**封顶**（图上折线先升后平）。→ 好动作最多被提升 20%。
- **A<0（坏动作）**：想减小 r，代价一路降到 r=0.8 就**封顶**（图上先平后降）。→ 坏动作最多被压制 20%。
- 反方向是**正常给梯度的**：A>0 而 r 很小（新策略快把这个好动作忘了）时梯度照给；A<0 而 r 很大时梯度照给。**只削"过头"的那一侧。**

这就是 PPO 的精髓：**不是把 r 截断就完事，而是用 min 让"过头方向的奖励变常数"，从而梯度自然为 0。**

图 2 是另一个佐证：沿着一次 PPO 更新的方向插值，`L^CLIP` 始终在 `L^CPI` 下方（是下界），且 `L^CLIP` 的最大值出现在 KL≈0.02 处 —— 也就是**目标函数自己把更新幅度限制住了**，不需要显式约束。

### 3.5 Sec 4 Adaptive KL Penalty Coefficient（第 4 页）

这是公式 (7) 的**替代方案**（也可以叠加使用），即给 β 做自适应：目标 KL 设为 d_targ，每轮更新后看实际 KL 并缩放 β。

```
L^KLPEN(θ) = Ê_t[ r_t(θ)·Â_t − β·KL[π_θold(·|s_t), π_θ(·|s_t)] ]        …… (8)

每轮更新后计算 d = Ê_t[ KL[π_θold, π_θ] ]
  若 d < d_targ / 1.5  →  β ← β / 2      （KL 太小，惩罚放松，允许走大点）
  若 d > d_targ × 1.5  →  β ← β × 2      （KL 太大，惩罚收紧）
```

细节：1.5 和 2 是启发式选的，算法对这个不敏感；β 的初值也不重要，因为会快速自调节。

**重要实验结论（作者自己说的）**：

> *"we found that the KL penalty performed worse than the clipped surrogate objective"*
> —— KL 惩罚**不如**裁剪目标，论文把它作为 baseline 保留。

### 3.6 Sec 5 Algorithm（第 4~5 页）

**完整目标函数（公式 9）** —— 当 policy 和 value 共享参数时，要把三项合起来：

```
L^{CLIP+VF+S}_t(θ) = Ê_t[ L^CLIP_t(θ) − c1 · L^VF_t(θ) + c2 · S[π_θ](s_t) ]     …… (9)
```

| 项 | 含义 | 系数 |
|---|---|---|
| `L^CLIP` | 策略目标（公式 7），要最大化 | — |
| `L^VF = (V_θ(s_t) − V_t^targ)²` | value 网络的平方误差（注意前面是**减号**，因为它是最小化的 loss） | c1 |
| `S[π_θ](s_t)` | 策略熵，鼓励探索 | c2 |

**优势估计（公式 10~12）** —— GAE 的截断版本：

```
Â_t = −V(s_t) + r_t + γ·r_{t+1} + ... + γ^{T−t−1}·r_{T−1} + γ^{T−t}·V(s_T)      …… (10)
Â_t = δ_t + (γλ)·δ_{t+1} + ... + (γλ)^{T−t−1}·δ_{T−1}                            …… (11)
δ_t = r_t + γ·V(s_{t+1}) − V(s_t)                                                …… (12)
```

- 公式 (10) 是 `[Mni+16]`（A2C）用的估计量，只看 T 步之内，**λ=1 时公式 (11) 退化成公式 (10)**。
- 公式 (12) 的 δ_t 就是 **TD 误差**：实际收益 + 下一状态估值 − 当前估值。
- λ 控制 bias/variance 权衡：λ→1 方差大偏差小，λ→0 偏差大方差小；论文用 **λ = 0.95**。

**算法 1（PPO, Actor-Critic Style）原文伪代码：**

```
for iteration = 1, 2, ... do
    for actor = 1, 2, ..., N do
        用 π_θold 在环境中跑 T 个 timestep
        计算优势估计 Â_1, ..., Â_T
    end for
    用 K 个 epoch、minibatch size M ≤ NT 优化替代目标 L（对 θ 求导）
    θ_old ← θ
end for
```

逐行解读与实现要点：

| 伪代码 | 含义 / 坑 |
|---|---|
| `N 个 actor 各跑 T 步` | 并行采样，一轮共 **NT 条数据**（MuJoCo 设置 N=1、T=2048） |
| `计算 Â_1..Â_T` | 用公式 (11)(12)，需要 value 网络 `V`，**这一轮内 V 也被更新**（见公式 9） |
| `K 个 epoch、M ≤ NT` | **同一批 NT 数据反复用 K 次**（MuJoCo：K=10、M=64）——这就是"多轮更新"的落地 |
| `θ_old ← θ` | 一轮结束后才刷新 θ_old，**这批数据随即丢弃**（on-policy 的本质） |

⚠️ 一个极易混淆的点：整个算法里 **θ_old 是"上一轮策略"，不是"固定的参考模型"**。它只保证"单步更新别太猛"。语言模型 RLHF 里那个"不许跑偏"的 **π_ref（SFT 模型）是论文里没有的东西**，见第 8 节。

### 3.7 Sec 6 Experiments（第 5~8 页）—— 可以略读，但要记住两个结论

**6.1 哪个替代目标最好**（7 个 MuJoCo 任务 × 3 seed = 21 次运行，归一化后随机策略=0、最好=1）：

| 目标函数 | 平均归一化得分 |
|---|---|
| 不裁剪、不惩罚 | **−0.39**（比随机策略还差！HalfCheetah 上彻底崩） |
| **裁剪，ε = 0.1** | 0.76 |
| **裁剪，ε = 0.2** | **0.82 ← 最佳** |
| 裁剪，ε = 0.3 | 0.70 |
| 自适应 KL，d_targ = 0.003 / 0.01 / 0.03 | 0.68 / 0.74 / 0.71 |
| 固定 KL，β = 0.3 / 1 / 3 / 10 | 0.62 / 0.71 / 0.72 / 0.69 |

三个必记结论：
1. **不裁剪会灾难性失败**（−0.39）—— 印证 Sec 2.1 的"destructively large updates"。
2. **ε=0.2 是甜点**，太小（0.1）限制过度、太大（0.3）不够稳。
3. **裁剪 > 自适应 KL > 固定 KL**；且裁剪对 ε 不敏感（0.1~0.3 都在 0.70 以上），KL 惩罚对 β/d_targ 明显更敏感 —— 这正是作者偏爱裁剪的理由。

**6.2** PPO(Clip) 在绝大多数连续控制任务上超过 TRPO、CEM、vanilla PG(adaptive)、A2C、A2C+TR。**6.3** 3D 人形跑/转向/被方块砸（Roboschool）展示高维能力。**6.4** Atari 49 游戏（Table 2）：

| 评分指标 | A2C | ACER | **PPO** | Tie |
|---|---|---|---|---|
| 训练全程平均回报（偏好学得快） | 1 | 18 | **30** | 0 |
| 最后 100 回合平均回报（偏好最终性能） | 1 | 28 | 19 | 1 |

读法：**PPO 学得更快、更早拿到分（30 胜），但最终上限略逊于 ACER（19 胜）**，而 PPO 简单得多。作者在 Sec 1 也是这么说的："similarly to ACER though it is much simpler"。

### 3.8 Sec 7 Conclusion + 附录 A 超参

结论就三句：PPO 有信赖域方法的稳定可靠，但**只需几行代码改动**（相对 vanilla PG），适用于更一般的场景（如 policy/value 联合架构），整体表现更好。

**附录 A 超参（MuJoCo，Table 3）—— 这是复现时的黄金参考：**

| 超参 | 值 |
|---|---|
| Horizon (T) | 2048 |
| Adam 学习率 | 3e−4 |
| Num. epochs (K) | 10 |
| Minibatch size (M) | 64 |
| 折扣 γ | 0.99 |
| GAE λ | 0.95 |

其他：Roboschool（Table 4）T=512、K=15、M=4096、32/128 actors；Atari（Table 5）T=128、K=3、M=32×8、c1=1、c2=0.01、ε=0.1×α，学习率随 α 从 1 线性退火到 0。

---

## 4. 八个常见困惑（读的时候一定会卡）

**Q1：为什么公式 (7) 要取 min，而不是直接用 clip？**
因为 clip 单独使用会**在错误的方向上给梯度**。举例：A>0 但 r=0.3（比 1−ε 还小），`clip` 会把它顶到 0.8，目标变成 0.8·A，梯度却在鼓励"继续减小 r"——方向反了。取 min 后，这种区间的目标退化为未裁剪的 r·A，梯度方向正确。**min 保证它永远是未裁剪目标的悲观下界。**

**Q2：为什么 A>0 和 A<0 的图（图 1）形状镜像？**
因为"过头"的方向不同：A>0 时怕把 r 推得太大（>1+ε），A<0 时怕把 r 压得太小（<1−ε）。裁剪点因此分别落在 1+ε 和 1−ε。见第 3.4 节的数字表。

**Q3：r 被 clip 了，梯度不就一直是 0 了吗？**
不是。`clip()` 本身在区间外梯度为 0，但 min 会把那一支**选掉或选上**：
- A>0 且 r>1+ε：min 选中第 2 项（常数）→ 梯度 0（刹车）。
- A>0 且 r<1+ε：min 选中第 1 项 r·A → 梯度 A>0 正常上升。
所以是**单侧刹车**，不是双侧锁死。

**Q4：既然能多轮更新，为什么不能更新 100 个 epoch？**
K 越大，θ 离 θ_old 越远，替代目标与真实回报的偏差越大（只有一阶近似保证）。ε 的裁剪是"每步的软约束"，K 是"总步数"，两者相乘才是实际偏离量。实践中 K 常取 3~10。

**Q5：θ_old 为什么不每步都刷新？**
因为 on-policy 的假设：数据由 θ_old 采样，替代目标在 θ_old 附近才可信。每步刷新 θ_old 就退化成"每批数据只更新一次"（回到 vanilla PG 的数据效率）。**刷新 θ_old = 丢弃数据。**

**Q6：公式 (4) 的 KL 为什么写 KL[π_θold, π_θ] 而不是反过来？**
在 θ_old 处两个方向都近似为 0，但作为"约束/惩罚"时，KL[π_old ‖ π_new] 关于 θ_new 是**无界**的（new 在某动作上趋于 0 而 old 不为 0 时罚不到），而 KL[π_new ‖ π_old] 是有界的。论文用的写法在实现中（含 LLM 的 RLHF）也常用"new ‖ old"的近似；TRL 就提供多种 `kl_penalty` 变体（kl / abs / mse / full）。**这点实现细节不同库不一致，以你用的库为准。**

**Q7：公式 (9) 里 c1 前面为什么是减号？**
因为整体是**最大化** L^{CLIP+VF+S}，而 value error 是要**最小化**的 loss，所以减去 c1·L^VF。同理熵项是加号（鼓励熵大）。论文 6.1 的 MuJoCo 实验 policy 和 value 不共享参数，所以 c1 无关；Atari 用 c1=1、c2=0.01。

**Q8：为什么公式 (11) 在 λ=1 时等于公式 (10)？**
把 δ_t = r_t + γV(s_{t+1}) − V(s_t) 代入并按 γ 展开，中间项是望远镜求和（telescoping）：γV(s_{t+1}) 与其后一项的 −γV(s_{t+1}) 抵消，最后只剩 −V(s_t) + Σγr + γ^{T−t}V(s_T)。这正好是公式 (10)。**这个推导建议自己动手写一遍，是理解 GAE 的最快路径。**

---

## 5. 从论文到 LLM / RLHF（重要落差）

论文写于 2017，环境是机器人和游戏。把 PPO 用到语言模型上，需要做这些"翻译"：

| 论文里的概念 | LLM / RLHF 里的对应 |
|---|---|
| 环境状态 s_t | prompt + 已生成的 token 序列 |
| 动作 a_t | 下一个 token（动作空间 = 词表，几万维） |
| 一条轨迹 τ | 一整段 response |
| 轨迹回报 R(τ) | 奖励模型对整段 response 打的**一个标量分** |
| V(s_t) | value head 对"前缀"的估值（需要 `AutoModelForCausalLMWithValueHead`） |
| θ_old | 本轮更新前的策略（同上） |
| **（论文没有）** | **π_ref = SFT 模型，用作"别跑出人类语言分布"的锚点** |

**最关键的差异：KL 的使用方式变了。**

- 论文里 KL 是公式 (5)/(8) 的**惩罚项**，或公式 (4) 的**约束**，用来限制**单步更新幅度**，对标的是 θ_old。
- LLM RLHF 里通常**同时**做两件事：
  1. 用裁剪目标公式 (7)（TRL 的 PPO 实现方式）；
  2. 把 per-token KL 惩罚**直接减进 reward**：`r_t = r_θ(s_t,a_t) − β·KL[π_θ(·|s_t) ‖ π_ref(·|s_t)]`，其中 π_ref 是**固定的 SFT 模型**。

第 2 条在 PPO 原文里**没有**，它来自 InstructGPT（[2203.02155](https://arxiv.org/abs/2203.02155)）一系的工作，目的是防止策略为了骗高 reward 而输出乱码、复读（reward hacking）。所以读 RLHF 代码时，**别拿论文公式 (8) 去套 `init_kl_coef`，它们是两个层面的东西。**

**工程代价（这也解释了 GRPO 为什么会出现）：** PPO 在 LLM 上要同时驻留 **4 个模型** —— policy、value/critic、reward model、reference model。GRPO（[2402.03300](https://arxiv.org/abs/2402.03300)）去掉 value 网络（用组内相对奖励代替 baseline），RLOO（[2402.14740](https://arxiv.org/abs/2402.14740)）进一步论证 RLHF 场景下 PPO 的多数组件并不必要。

---

## 6. 和你这个仓库的对应（读完顺便看代码）

⚠️ **先纠正一个容易踩的坑**：`training/ppo_training.py` **当前用的不是 PPO**。

- 该文件 `import` 的是 `RLOOConfig, RLOOTrainer`（`training/ppo_training.py:21-26`），文件里也写着：
  > *"We then build the RLOOTrainer... RLOO does not need a separate value model or ref model (unlike PPO)"*（`ppo_training.py:285-287`）
- 本环境安装的 **TRL 0.29.1 里已经没有 `PPOConfig` / `PPOTrainer`**（我验证过：`from trl import PPOConfig` 直接 ImportError，只剩 `GRPOConfig/GRPOTrainer`、`RLOOConfig/RLOOTrainer`），而 `requirements.txt:10` 钉的是 `trl>=0.29.0`。

也就是说：**论文里的 PPO 在这套代码栈里已经被"去掉 critic"的 RLOO 取代了**，而 RLOO 正是论文 Sec 2.1 提到的 vanilla policy gradient 谱系（重要性采样 + 基线）的现代化版本。所以读 PPO 论文的价值在于**理解"为什么要裁剪/限制更新幅度"这个问题的来源**，而仓库目前的实现是这个问题的一种更省资源的答案。

对照表：

| 论文 | 仓库文件 | 状态 |
|---|---|---|
| PPO（本篇 1707.06347） | `training/ppo_training.py` | 文件名沿用，实现已换成 RLOO |
| RLOO / Back to Basics（2402.14740） | `training/ppo_training.py`（`RLOOTrainer`） | 当前实际使用 |
| GRPO（2402.03300） | `training/grpo_training.py`（`GRPOTrainer`） | 无需 critic |
| 奖励模型（pairwise logloss，来自 InstructGPT） | `training/reward_modeling.py:228` | PPO/RLOO 的前置阶段 |
| 兼容性限制 | `docs/FAQ.md` | chatglm/baichuan 不支持 value head，做不了 PPO/RM |

**动手验证建议（读懂后做一遍，收益很大）：**

1. 写个 5 行脚本复现第 3.4 节那张数字表，并画图对比图 1 的两个子图 —— 比看论文快 10 倍理解 clip 的作用。
2. 用 `torch` 手写 `L^CLIP` 的负号版本（`loss = -min(r*A, clamp(r,1-eps,1+eps)*A).mean()`），对 `r` 做一次反传，检查 r>1+ε 且 A>0 时梯度确实是 0。
3. 自己推导 Q8（λ=1 → 公式 10），再和 `grpo_training.py` 里不需要 value 网络的 advantage 计算做对比。

---

## 7. 自测题（能答上来就算读懂了）

1. PPO 相对 vanilla policy gradient 多做了什么，才允许"多轮复用同一批数据"？
2. 写出 L^CLIP，并解释 min 为什么让它成为 lower bound。
3. A>0 时裁剪点是 1+ε 还是 1−ε？A<0 呢？为什么？
4. 论文里选出来的最佳 ε 是多少？不裁剪时 MuJoCo 的得分是多少？
5. 裁剪 vs 自适应 KL，哪个更好？依据是哪个表？
6. Algorithm 1 里 θ_old 什么时候刷新？为什么不能每步刷新？
7. 论文用的是 KL[π_old, π_new] 还是 KL[π_new, π_old]？为什么实现里常见另一种写法？
8. PPO 论文里有没有 reference model？LLM RLHF 为什么要额外加 π_ref + KL 惩罚？

<details>
<summary>参考答案（点开前先自己想）</summary>

1. 用带裁剪的概率比替代目标 L^CLIP 限制了单步更新幅度，使替代目标在 θ_old 附近仍与真实回报对齐，因此可在 NT 条数据上做 K 个 epoch 的小批量更新。
2. `L^CLIP = Ê[min(rÂ, clip(r,1−ε,1+ε)Â)]`；min 保证结果不超过未裁剪值，即对 L^CPI 的悲观下界。
3. A>0 → 1+ε（怕好动作的概率被推得太高）；A<0 → 1−ε（怕坏动作的概率被压得太低）。
4. ε=0.2（得分 0.82）；不裁剪为 −0.39，比随机策略还差。
5. 裁剪更好（0.76/0.82/0.70 vs 自适应 KL 0.68/0.74/0.71），依据 Table 1。
6. 每轮迭代结束时（`θ_old ← θ`）；因为替代目标只在 θ_old 附近可信，刷新即意味着旧数据必须丢弃。
7. 论文写 KL[π_θold, π_θ]；实现（含 TRL 的多种 `kl_penalty`）常用近似或反向，因为两个方向在 θ_old 处一阶等价，但数值/有界性不同。
8. 没有。π_ref 是 RLHF 适配引入的（InstructGPT 谱系），用来把策略锚定在 SFT 的语言分布上、抑制 reward hacking。
</details>

---

## 8. 阅读时间表

**30 分钟速通**（只想抓住思想）
Sec 1（3 分钟）→ Sec 2.1 末两段（3 分钟）→ **Sec 3 + 图 1 + 公式 (7)**（15 分钟）→ Sec 5 的 Algorithm 1（5 分钟）→ Sec 6.1 的 Table 1（4 分钟）。**跳过 Sec 4、Sec 6.2-6.5、附录 B。**

**2 小时精读**（要能复现/调参）
按第 3 节顺序通读全部正文（60 分钟）→ 手算第 3.4 节数字表 + 自己推导公式 (11)→(10)（30 分钟）→ 读附录 A 三张超参表并对照 Sec 6 的实验设置（15 分钟）→ 做第 7 节自测题（15 分钟）。

**读完之后**：按第 6 节第 3 条的建议，把 `docs/ppo_grpo_papers.md` 里的 GRPO 论文（[2402.03300](https://arxiv.org/abs/2402.03300)）接着读 —— GRPO 的第一步就是把 PPO 的 value 网络换成"组内平均奖励"，你会立刻看出它在省什么。
