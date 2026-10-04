# GRPO 优化论文选型与落地实现方案

> 面向本仓库 `training/grpo_training.py`（TRL 0.29.1 `GRPOTrainer`）
> 你的规格：数学解答能力 · 规则奖励（规范性+准确性）· 无 critic · 痛点=显存/吞吐 + 效果不明显 · 要求=可读性高、稳定性高、关联性强

---

## 0. 结论

**推荐论文：AR3PO — Improving Sampling Efficiency in RLVR through Adaptive Rollout and Response Reuse**
Yuheng Zhang et al.（UIUC + Amazon），2025-09-30，arXiv: [2509.25808](https://arxiv.org/abs/2509.25808)
本地 PDF：`docs/papers/GRPO_2509.25808.pdf`

**一句话理由**：它是候选里**唯一同时命中你两个痛点**的论文——效果上稳定超过 GRPO、追平/略超 DAPO；成本上比 DAPO 少 **4.2×** 生成量。而且它解决的是 GRPO 在数学任务上**最要命的结构性缺陷**（组内奖励全同 → advantage 全 0 → 梯度为 0）。

**额外好消息**：TRL 0.29.1 已经内置了 AR3PO"复用"那一半的实现（`GRPOWithReplayBufferTrainer`，我实测可导入），所以落地成本比预期低得多。

**备选叠加**：GRESO（arXiv: [2506.02177](https://arxiv.org/abs/2506.02177)），纯 pre-rollout 过滤，总训练时间 2.0× 加速、精度不降，实现更简单但**不提升效果**。

---

## 1. 候选对比（关键一列：TRL 0.29.1 是否已内置）

| 论文 | 时间 | 核心贡献 | 对你痛点的作用 | TRL 0.29.1 状态 |
|---|---|---|---|---|
| **DAPO** [2503.14476](https://arxiv.org/abs/2503.14476) | 2025-03 | clip-higher、dynamic sampling、token-level loss、overlong 惩罚 | 效果↑；但 dynamic sampling **反而 +3× 生成成本** | ✅ `loss_type='dapo'`、`epsilon_high` 已内置；**dynamic sampling 未内置** |
| **Dr. GRPO** [2503.20783](https://arxiv.org/abs/2503.20783) | 2025-03 | 去掉 std 归一化偏置、消除长度偏置 | 效果↑ | ✅ `scale_rewards` 可配（`'none'` 即 Dr.GRPO 风格） |
| **GSPO** [2507.18071](https://arxiv.org/abs/2507.18071) | 2025-07 | 序列级重要性比值与裁剪 | 稳定性↑（MoE 场景显著） | ✅ `importance_sampling_level='sequence'` 已内置 |
| **VAPO** [2504.05118](https://arxiv.org/abs/2504.05118) | 2025-04 | value-based 增强，AIME 60.4 | 效果↑↑ | ❌ 需要 critic → **直接违背你"省显存"的目标** |
| **⭐ AR3PO** [2509.25808](https://arxiv.org/abs/2509.25808) | **2025-09** | **vanishing advantage 的高效解法：自适应 rollout + 响应复用** | **效果↑ + 生成成本↓4.2×** | ✅ 已实现 adaptive rollout；❌ replay buffer 那半在 TRL 0.29.1 上不可用（见 §4） |
| **GRESO** [2506.02177](https://arxiv.org/abs/2506.02177) | 2025-06 | pre-rollout 概率过滤无效 prompt | 成本↓2.0×，精度不降 | ❌ 未内置（但只需 override `_get_train_sampler`） |

**选型逻辑**：

> DAPO / Dr. GRPO / GSPO 的贡献**你已经"免费拥有"**（都是 config 开关），再花时间"实现"它们没有增量价值。
> 真正值得动手的只有两个：**AR3PO**（唯一同时提升效果+降低成本）和 **GRESO**（只降成本）。
> 你说"效果提升不明显"是痛点之一，所以 **AR3PO 胜出**。

---

## 2. AR3PO 论文精讲

### 2.1 它要解决的问题

GRPO 的 advantage 是**组内归一化**：

```
A_i = (R_i − mean({R_i}_{i=1}^G)) / std({R_i}_{i=1}^G)
```

当一组 response **全对或全错**时，`std = 0`，**所有 A_i 都变成 0 → 该组不产生任何梯度**。

论文称之为 **vanishing advantage（优势消失）**，这类 prompt 叫 **zero-variance prompt**。

DAPO 的解法是 **dynamic sampling**：不断采新 prompt 直到组内有非零方差。**代价是生成量至少增加 3 倍**（论文原话），对大模型很快成为瓶颈。

论文的两个关键观察：

1. **GRPO 对每个 prompt 生成固定数量，不管难度。** 难题需要更多次采样才能碰到一个正确答案；简单题生成太多是浪费，且容易"全对"。
2. **GRPO 只用 on-policy 样本，丢弃历史生成的回答。** 难题当前这步全错就没有训练信号——**哪怕之前某一步曾经生成过正确答案却被丢掉了**。

### 2.2 两个技术

#### 技术一：Adaptive Rollout（自适应 rollout）

把 rollout 分成 **S 个阶段**，维护一个 prompt 池 `U`：

```
每一阶段：对 U 中每个 prompt 生成 k 个 response
         → 有至少一个正确的 prompt 移出 U
         → 重复，直到 U 为空或达到最大阶段数
```

- **简单题只生成 k 个**（k < G）→ 省下预算
- **难题生成 S×k 个**（= G）→ 提高至少碰到一个正确的概率

论文还有个很好的解读：这等价于**对 prompt 做自适应加权**——难题拿到更多 response，在优化中权重更高。

#### 技术二：Response Reuse（响应复用）

维护一个 **replay buffer `B`**，存所有历史生成的**正确** response。
对自适应 rollout 后仍然全错的 prompt，**从 `B` 里随机取一个正确 response 替换组内一个错误 response**，再重算 advantage。

难点：复用样本来自旧策略 `π_old`，重要性比值 `r_c,t = π_θ'(o_c,t) / π_θold(o_c,t)` 可能**过大（高方差）或过小（梯度消失）**。论文给了两个解法：

| 方案 | 做法 | 论文结果（Qwen2.5-7B 平均分） |
|---|---|---|
| 直接 rollout replay | 用旧策略概率当分母 | **41.8**（最差） |
| **Option I** | 用**当前策略**重算 token 概率当分母（引入偏差，降低方差） | 42.8 |
| **⭐ Option II** | **对复用 response 停梯度**，只在 advantage 计算里用它的 reward 信息；模型只用 on-policy 样本更新 | **43.1**（最好，且更省算力） |

**Option II 的直觉（论文的解释很漂亮）**：既然旧策略当年能生成正确答案，说明当前模型**本该已经具备解这道题的知识**。所以复用它的**奖励信息**、给当前错误回答**负 advantage**，本质是一种"负样本训练"——**明确告诉模型别往错误方向探索**。

> 💡 **Option II 是实现的关键**：不需要重算概率、不需要为复用样本算梯度，**只需在 advantage 构造时把它放进去，再用 mask 把它的 loss 置 0**。这让实现变得非常干净。

### 2.3 实验结果（论文 Table 1）

| 模型 | 方法 | Math500 | Minerva | Olympiad | AIME24 | 平均 | 采样数 | 相对 DAPO 加速 |
|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B | GRPO | 77.5 | 37.4 | 38.8 | 15.2 | 42.2 | 512×8.0 | 3.0 |
| | DAPO | 77.2 | 36.4 | 41.1 | 16.9 | 42.9 | **1536×8.0** | 1.0 |
| | **AR3PO** | **78.8** | 36.0 | 39.6 | **18.0** | **43.1** | **512×5.7** | **4.2** |
| Llama-3.1-8B | GRPO | 52.6 | 26.6 | 19.9 | 6.7 | 26.5 | 512×8.0 | 3.0 |
| | DAPO | 53.6 | 28.1 | 20.3 | 9.2 | 27.8 | **1536×8.0** | 1.0 |
| | **AR3PO** | 53.7 | 26.6 | 21.2 | **9.5** | **27.8** | **512×6.7** | **3.6** |

**三个必记结论**：
1. AR3PO **稳定超过 GRPO**，追平或略超 DAPO；
2. **生成成本只有 DAPO 的 1/4.2**（DAPO 要 1536 个 prompt 才凑够 512 个有效组，AR3PO 只要 512）；
3. 随训练推进，AR3PO 平均只需 **5.7 个 response/prompt**（从 8 降下来）——**越训越省**。

**训练动态（论文 Figure 2）**：
- "无正确答案 prompt 比例"从 **~0.3 降到 <0.2**（response reuse 的功劳）
- 最难的一档 prompt（成功率 0~0.2）平均拿到 **6.95** 个 response；最简单一档（0.8~1.0）只用约 **4** 个

### 2.4 论文超参（可直接照抄）

| 项 | 值 |
|---|---|
| 阶段数 S / 每阶段 k | **S=2, k=4**（最多 8，与 GRPO 的 G=8 对齐） |
| training prompt batch | 512（DAPO 用 1536） |
| mini-batch | 128 |
| 学习率 | Qwen2.5-7B **1e-6**；Llama-3.1-8B **1e-7**（已 RLHF 过，故更小） |
| max prompt / response | 1024 / 3072（Qwen）；1024 / 2048（Llama） |
| warmup | **不用** |
| verifier | **Math-Verify**（github.com/huggingface/Math-Verify） |
| 框架 | verl + vLLM |

---

## 3. 逐条对照你的 4 个要求

| 你的要求 | AR3PO 的匹配度 |
|---|---|
| **数学解答能力** | ✅ 论文就是纯数学推理 RLVR，训练集 DAPO-Math 14K，评测 MATH500/Minerva/Olympiad/AIME24 |
| **规则奖励、不要 critic** | ✅ RLVR 范式，二值规则奖励，**完全无 value network**，显存需求与 GRPO 同级 |
| **痛点①显存/速度/吞吐** | ✅ **降低生成成本 4.2×**。生成是 GRPO 的主要瓶颈，这项收益直接 |
| **痛点②效果不明显** | ✅ 稳定超 GRPO，追平/略超 DAPO；且把全错组从 30% 降到 20% |
| **可读性高** | ✅ Option II 只需在 advantage 构造 + mask 上做手脚，**不碰 loss 公式** |
| **稳定性高** | ✅ 只改了采样层、不碰损失；不引入新超参；任何不安全情形都自动回退到标准 GRPO |
| **与现有代码关联强** | ✅ 纯 `GRPOTrainer` 子类化，`training/grpo_training.py` 只需换 trainer 类和加 2 个参数 |

---

## 4. ⭐ 落地设计（TRL 0.29.1）

### 4.1 TRL 内置的 replay buffer —— 看起来像"已内置一半"，实际不可用

TRL 0.29.1 里有 **`trl/experimental/grpo_with_replay_buffer/`**，其 config 的 docstring 原文：

> *"A cache that stores the rollouts with the highest advantage scores and variance per group. **If a new group has 0 variance, it is replaced with a group sampled from the replay buffer.**"*

新增参数只有一个：`replay_buffer_size`（默认 64）。导入是成功的：

```python
from trl.experimental.grpo_with_replay_buffer import (
    GRPOWithReplayBufferTrainer, GRPOWithReplayBufferConfig,
)
# 继承链: GRPOWithReplayBufferTrainer -> GRPOTrainer -> BaseTrainer -> Trainer
```

**但"能导入"≠"能用"。** 我最初据此判断"AR3PO 的 response reuse 已经免费拥有"，
**这是错的**。核实源码后发现两个独立缺陷，导致它在 AR3PO 下失效、在标准 GRPO 下也不可靠
（详见 4.2）。已在 `--replay_buffer_size 0` 关闭。

⚠️ 另外即便它能工作，TRL 的做法也与论文不同：TRL 是**整组替换**（更接近 AR3PO 论文
Table 2 里最差的 *"direct rollout replay"*，41.8），而论文最优的 Option II 是
「把历史正确 response 放进当前组 → 给它停梯度 → 让当前错误回答拿负 advantage」（43.1）。

**所以 response reuse 这一半目前是缺失的**，见 4.2。

### 4.2 落地阶段（已按实测结果修正）

> ⚠️ **本节曾经写错，2026-09 已更正。** 原来把「换用 TRL 内置 replay buffer」列为
> 「阶段 A：零代码风险，先拿收益」。**实测证明该内置实现不可用**，详见下。

#### ~~阶段 A（零代码风险，先拿收益）~~ → ❌ 已作废：TRL 0.29.1 的 replay trainer 是坏的

原计划：把 `GRPOTrainer` 换成 `trl.experimental.grpo_with_replay_buffer.GRPOWithReplayBufferTrainer`
并设 `--replay_buffer_size 64`。

**核实到 TRL 源码后，该实现有两个独立缺陷**：

| # | 缺陷 | 证据 | 后果 |
|---|---|---|---|
| 1 | **不切分 `tool_mask`** | `slice_group_data` / `update_replay_buffer` / `sample_from_replay_buffer` 里完全没有 `tool_mask`（全文只在 102/124/125/300/373 行出现，都是读取与输出） | 与 AR3PO 混用时，被替换的位置恰是 `tool_mask=0` 的补齐样本 → 替换进来的组**拿不到梯度**，replay 静默失效 |
| 2 | **对 1D 张量用 `max(dim=0)`** | 父类产出 `grouped_std_rewards = rewards.view(-1, G).std(dim=1).repeat_interleave(G, dim=0)` → **1D**；而 replay trainer 用 `group_std_rewards.max(dim=0).values > 0` 判断 | 对 1D 张量该表达式返回**标量**，于是：· 批次里存在任何有方差组 → `num_groups_to_replace = 0` → **静默 no-op**；· 整批全零方差 → 采样空 buffer 得 `None` → `for item in None` → **`TypeError` 崩溃** |

**结论**：缺陷 2 说明它在**标准 GRPO 下也是坏的**（要么不生效要么崩），
不只是「与 AR3PO 不兼容」。

**因此当前方案：`--replay_buffer_size 0`（默认关闭）。**
`training/grpo_training.py` 里加了双重保险：
- `ar3po_stages > 1` 时强制置 0 并打警告
- `ar3po_stages == 1` 时若启用，会用子类覆写 `update_with_replay_buffer`
  拦住「空 buffer + 全零方差」的崩溃路径（返回 `None`，上游会正常回退）

**这意味着 AR3PO 论文的 response reuse 目前未落地**，只有 adaptive rollout 生效。
详见 `training/ar3po_trainer.py` 顶部说明与 README「已知限制」第 5 条。

#### 阶段 B（论文独有部分：adaptive rollout）— ✅ 已实现并验证

TRL 的固定形状约束是最大障碍：`GRPOTrainer` 把 completion 按 `[num_prompts, num_generations]` reshape，**每组的 response 数必须相同**。

**破解思路**：论文用 **S=2, k=4**，最多 8 个 —— 那就**让形状恒定为 8**：

```
Round 1：对所有 prompt 生成 4 个
        → 算 reward
        → 筛出「全错」的难题集合 H
Round 2：只对 H 再生成 4 个 → H 中的 prompt 凑满 8 个
        → 对 Round 1 就成功/有效的 prompt（非 H），
          把它们的 4 个 completion 复制一份凑满 8 个，
          复制件的 tool_mask 置 0（不参与梯度）

复制补齐的数学性质（已数值验证）：
  · mean 精确不变 ✓
  · std 会变：std_2k / std_k = sqrt(2(k-1)/(2k-1))  ← 只依赖 k 的常数
    → 组内 advantage 的符号与相对大小完全保持，仅幅度被统一缩放
    → 配 scale_rewards="none" 时 advantage 与"只有 k 条"时**完全相同**
```

⚠️ **实测修正（重要）**：我原先假设"复制后 mean 和 std 都不变"是**错的**。正确结论是：
- `scale_rewards="none"` → **数学上完全精确**，推荐用这个
- `scale_rewards="group"`（默认）→ advantage 幅度被放大 `sqrt((2k-1)/(2(k-1)))`
  （k=2 → ×1.2247，k=3 → ×1.118，k=4 → ×1.080），方向不受影响
- 实现里已加**自动警告**：检测到 `scale_rewards="group"` 时会打印缩放倍数

顺带的好处：`scale_rewards="none"` 正好也是 Dr. GRPO 主张的"不要用组内 std 归一化"，
一举两得。

**效果**：
- 难题拿 8 个 response（预算翻倍，提高碰到正确答案的概率）
- 简单/有效题只花 4 个（**省 50% 生成**）
- 形状恒为 8 → **完全兼容 TRL 的 reshape 逻辑**

**需要 override 的扩展点**（我已在本地确认全部存在）：

```python
GRPOTrainer._generate(prompts) -> (
    prompt_ids_list, completion_ids_list, tool_mask_list,
    completions, num_items_in_batch, sampling_per_token_logps_list, extra_fields,
)   # ← 7 元组（不是 5 元组）
GRPOTrainer._generate_and_score_completions(inputs) -> dict   # 主扩展点
GRPOTrainer._calculate_rewards(inputs, prompts, completions, completion_ids_list)
```

**⭐ 实测出的 TRL 真实数据契约**（`tools/trl_contract_probe.py`，已用离线极小模型跑通）：

| 事实 | 说明 |
|---|---|
| `_generate` 收到的 prompts **已经重复过 G 次** | 训练 sampler 用 `RepeatSampler(mini_repeat_count=num_generations)`，故 `len(prompts) == generation_batch_size == G × 唯一问题数`，且同一问题的 G 个副本**连续排列** |
| `_generate` 只对每个 prompt 生成 **1 条** | `len(completion_ids) == len(prompts)`（HF 与 vLLM 路径一致） |
| 上游用 `rewards.view(-1, G)` 归一化 | 返回的扁平列表长度**必须恰好等于 len(prompts)** |
| **`tool_mask` 就是"停梯度"开关** | `_compute_loss`: `mask = completion_mask * tool_mask`，置 0 即该序列不产生梯度 |
| `num_items_in_batch` **必须是 0 维 tensor** | 传 Python int 会在 `shuffle_sequence_dict` 抛 `'int' object is not subscriptable`（已踩坑并修复） |

**✅ 已完成的实现**：`training/ar3po_trainer.py` —— 这是**可运行代码**，不是骨架。

核心思路比原设计更简单：因为 prompts 已经重复过 G 次，只需**按组抽取 k 个副本**去生成，再按组拼回。

```python
class AR3POTrainer(GRPOTrainer):
    def _generate_and_score_completions(self, inputs):
        self._ar3po_inputs = inputs      # 挂上答案列，供判对错
        try:     return super()._generate_and_score_completions(inputs)
        finally: self._ar3po_inputs = None

    def _generate(self, prompts):
        G, S = self.num_generations, self.args.ar3po_stages
        groups  = self._split_groups(prompts, G)   # 连续 G 个相同 → 一组
        pending = list(range(len(groups)))
        for _ in range(S):
            if not pending: break
            # 每道待解问题取第 1 个副本作 prompt，重复 k 次
            cur = [prompts[groups[g][0]] for g in pending for _ in range(k)]
            rollout = super()._generate(cur)       # 只生成 k×|pending| 条
            ...
            pending = [g for g in pending if 该组仍全错]   # 有正确即退出
        return self._assemble(...)  # 每组拼到恰好 G 条，不足的复制补齐 + tool_mask=0
```

**验收标准**（1~4 已用 `tools/ar3po_smoke_test.py` 实测验证）：

| # | 验收项 | 结果 |
|---|---|---|
| 1 | 能在 TRL 上跑通完整训练步 | ✅ 基线与 AR3PO 均跑通 |
| 2 | 组装后 `len(out) == len(in)`（满足 `view(-1,G)` 约束） | ✅ 实测 16 → 16 |
| 3 | 补齐样本 `tool_mask` 全 0（真正停梯度） | ✅ 实测 4 行全 0 |
| 4 | 生成量下降 | ✅ 实测 16 → 12（**25%**） |
| 5 | advantage 仍在完整 G 条上计算 | ✅ 实测 16 行（补齐样本参与组内归一化） |
| 6 | 真实 7B/8B 上的分数与吞吐收益 | ⏳ 待你在真实数据上复现 |

**运行验证的方式**（不需要 GPU、不需要 HF Hub，本地从零构造极小模型）：

```bash
python3 tools/trl_contract_probe.py    # 探测 TRL 契约
python3 tools/ar3po_smoke_test.py      # 端到端对比基线 vs AR3PO
```

#### 阶段 C（可选进阶：Option II 精确版）

阶段 A 的 replay buffer 是"整组替换"，阶段 C 可改成论文原版：
- 只把**一个历史正确 response** 塞进当前组（而非替换整组）
- 在 `_calculate_rewards` 之后、advantage 计算之前构造组内 reward 向量
- 对该 slot 的 `completion_mask` 置 0（Option II 停梯度）

**收益按论文是 41.8 → 43.1**；但实现复杂度上升，**建议阶段 A/B 稳定后再评估**。

---

## 5. 前置修复清单（P0：不做这些，任何算法都无效）

我在你的仓库里查到 3 个必须先解决的问题：

### 5.1 ⚠️ vLLM 未安装 —— 吞吐的第一瓶颈

```
vllm: MISSING
```
`GRPOConfig.use_vllm` 默认 `False`，且环境里没有 vllm → **当前 rollout 用的是 HF `model.generate`，吞吐与 vLLM 差数倍**。

**这比换算法重要得多**。AR3PO 论文也是用 vLLM 做 rollout 的。
**动作**：装 vllm（`pip install vllm`）并设 `--use_vllm True --vllm_mode colocate`（单机多卡）或 `server`（分离部署）。

### 5.2 ⚠️ reward 函数与数据必须配套

现状（`training/grpo_training.py:395-398` 与 `data/grpo/sample.jsonl`）：

| 项 | 现状 | 问题 |
|---|---|---|
| 数据 | `data/grpo/sample.jsonl` 是**医疗** QA，answer 是短文本（如"食管克罗恩病"） | 你要做数学 → **需换数学数据**（如 DAPO-Math / MATH / GSM8K） |
| 奖励 | `accuracy_reward` + `format_reward`，用 `math_verify` 的 `parse`/`verify` + `\boxed{}` | ⚠️ 医疗短答案**永远不会有 `\boxed{}`**，`format_reward` 大概率恒 0，`accuracy_reward` 也几乎恒 0 |

**好消息**：换成数学数据后，这套奖励**本来就是对的**（它就是从 TRL 数学示例搬来的）。你原仓库的错配会被自动修好。

`math_verify` **已安装** ✓，可直接用。

**建议的奖励设计**（对应你说的"规范性 + 准确性"）：

```python
from math_verify import parse, verify, LatexExtractionConfig, ExprExtractionConfig

def accuracy_reward(completions, answer, **kwargs):
    """准确性：用 math_verify 做符号等价判定（比字符串匹配鲁棒得多）。"""
    rewards = []
    for (completion,), gold in zip(completions, answer):
        content = completion["content"]
        try:
            gold_parsed = parse(gold, extraction_config=[LatexExtractionConfig(), ExprExtractionConfig()])
            pred_parsed = parse(content, extraction_config=[LatexExtractionConfig(boxed_match_priority=0),
                                                            ExprExtractionConfig()])
            rewards.append(float(verify(gold_parsed, pred_parsed)))
        except Exception:
            rewards.append(0.0)
    return rewards

def format_reward(completions, **kwargs):
    """规范性：分档给分，避免二值奖励让组内奖励全同。"""
    import re
    rewards = []
    for (completion,) in zip(completions):
        c = completion["content"]
        score = 0.0
        if "####" in c or "boxed" in c:
            score += 0.5                      # 有最终答案标记
        if re.search(r"(?i)(step|步骤|首先|then|therefore)", c):
            score += 0.2                      # 有推理过程
        if 20 <= len(c.split()) <= 800:
            score += 0.3                      # 长度合理（抑制长度爆炸）
        rewards.append(score)
    return rewards
```

> 💡 **两条设计要点**：
> 1. **避免纯二值奖励**。GRPO 靠组内方差，二值奖励在难题上必然全 0 → 正好是 AR3PO 要解决的问题。细粒度 format reward 能天然提供梯度。
> 2. **format reward 里加长度项**，可抑制 GRPO 已知的长度爆炸倾向。

### 5.3 超参核对

你在 `training/grpo_training.py` 里没有显式设置这些，用的是 TRL 默认值：

| 参数 | TRL 默认 | 建议 | 理由 |
|---|---|---|---|
| `num_generations` | 8 | 8（配合 AR3PO 的 S=2,k=4） | 组内样本数，显存的主要驱动 |
| `max_completion_length` | **256** | **2048~3072** | ⚠️ 256 对数学推理**远远不够**，长链会全部被截断 |
| `beta` | 0.0 | 0.0 | RLVR 不建议 KL 惩罚（论文亦然） |
| `loss_type` | `dapo` | `dapo` | 已是最优默认 |
| `scale_rewards` | `group` | **`none`**（用 AR3PO 时） | ⚠️ 见下方说明：AR3PO 的补齐机制要求用 `none` 才数学精确；也正好是 Dr.GRPO 的主张 |
| `epsilon_high` | None | 0.28 | DAPO 的 clip-higher |
| `mask_truncated_completions` | False | **True** | 截断的样本不该产生梯度 |
| `use_vllm` | False | **True** | 见 5.1 |

⚠️ **`max_completion_length=256` 是最容易被忽略但影响最大的一个**——数学题的长 CoT 动辄上千 token。

⚠️ **`scale_rewards` 必须配 `none` 才能用 AR3PO**：
用 adaptive rollout 时，简单题会被"复制补齐"，而复制只保持均值、会改变标准差
（`std_2k/std_k = sqrt(2(k-1)/(2k-1))`）。若用默认的 `group`（除以 std），
补齐组的 advantage 幅度会被统一放大 `sqrt((2k-1)/(2(k-1)))`（k=2 时 ×1.22），
方向虽不变但幅度失真。用 `none`（advantage = reward − mean）时**完全精确**。
`AR3POTrainer` 会在检测到 `group` 时自动打印警告。


---

## 6. 分阶段实施路线

```
阶段 0  基线建立（1天）
  · 换数学数据（DAPO-Math / MATH，建议 7K~14K 条）
  · 重写 reward 函数（accuracy + 分档 format）
  · 修超参（max_completion_length 2048+、mask_truncated_completions、epsilon_high）
  · 装 vllm 并开 --use_vllm True
  · 跑 200 步，记录 baseline 分数 + 三个监控指标（reward / resp_length / 零方差组比例）
      ↓  验收：零方差组比例 < 60%，reward 有上升趋势
阶段 1  换 replay buffer trainer（0.5天，零算法风险）
  · GRPOWithReplayBufferTrainer + --replay_buffer_size 64
  · 同样 200 步，对比零方差组比例与分数
      ↓  验收：零方差组比例下降 ≥ 10%，分数不低于基线
阶段 2  实现 adaptive rollout（2~3天，本方案第 4.2 节阶段 B）
  · AR3POTrainer(GRPOWithReplayBufferTrainer)
  · 用「复制+mask」保持形状恒为 8
      ↓  验收：达标所需步数减少，或同预算下分数提升；生成 token 量下降
阶段 3  接入 DPO（见第 8 节）
```

**每一步都保持 `training/grpo_training.py` 可运行**，用命令行开关切换，方便回滚。

---

## 7. 风险与降级方案

| 风险 | 概率 | 降级方案 |
|---|---|---|
| `trl.experimental` API 变动 | 中 | **pin `trl==0.29.1`**；或用 `GRPOWithReplayBufferTrainer` 的源码自建一份到 `training/` 下 |
| 复制补齐导致显存上升（8 个 slot 但 4 个有效） | 中 | completion 部分显存与 token 数线性相关；可把 `num_generations` 降到 4、k=2 降低峰值 |
| adaptive rollout 与 TRL 内部 reshape/`num_items_in_batch` 冲突 | 中 | 退回阶段 1（replay buffer 已解决主要问题） |
| AR3PO 在 8B 上的收益不如论文的 7B | 中 | 论文 32B 实验也显示"追平 DAPO 而非超越"，但**成本优势始终存在** |
| vLLM 与 LoRA/量化不兼容 | 低 | vLLM 支持 LoRA adapter；若冲突则先全参小模型验证 |

---

## 8. DPO 那一半怎么接

你要求"GRPO 和 DPO 分别实现综合给分，强化输出答案的规范性和准确性"。两者其实可以**串成一条生产线**，而不是各自独立：

```
GRPO（本方案）→ 用规则奖励直接优化准确性 + 规范性
      ↓ 顺带产出大量「正确 vs 错误」的 response 对
      ↓ 按 format reward 分档，构造 chosen / rejected 偏好对
DPO（training/dpo_training.py 已存在）
      ↓ 用偏好对做离线蒸馏，进一步固化格式与解题风格
```

**优势**：
- **数据免费**：GRPO 的 rollout 本身就是天然的偏好对来源（正确答案 vs 错误答案、格式好 vs 格式差），不用额外人工标注
- **互补**：GRPO 负责"探索 + 提升上限"，DPO 负责"稳定 + 压低方差"，且 DPO 不需要 reward 模型也没有采样开销
- **与你现有代码最贴合**：`training/grpo_training.py` 和 `training/dpo_training.py` 已经是并列的两个脚本，只需加一个"从 GRPO rollout 导出偏好对"的中间脚本

**建议顺序**：先把阶段 0-2 跑通（GRPO 有稳定提升），再做 DPO 蒸馏。**不要在 GRPO 还没跑通时并行搞 DPO**——否则两个都是变量，无法定位问题。

---

## 9. 备选方案：GRESO

arXiv: [2506.02177](https://arxiv.org/abs/2506.02177)（CMU + LLNL + UIUC），本地 `docs/papers/GRPO_2506.02177.pdf`

**核心**：zero-variance prompt 有**强时间一致性**（之前零方差的，之后大概率仍零方差），但约 **20% 会重新变有效** → 所以用**概率过滤**而非硬剪枝：

```
p_f(x_i) = 1 − p_e^{z_i}
z_i = 最近连续零方差 的次数
```
再配自适应探索概率（easy/hard 分开调，目标零方差比例 25%，1:2 分配）+ 自适应 rollout batch size。

**效果**：rollout 最高 **2.4×**、总训练时间最高 **2.0×** 加速，**精度不降**。

**为什么是备选**：它**只降成本、不提升效果**，解决不了你"效果不明显"的痛点。
**但如果阶段 B 的 adaptive rollout 实现受阻**，GRESO 是最佳降级——它只需 override `_get_train_sampler`（在 rollout 之前跳过 prompt），是三种方案里**侵入性最小的**。

---

## 10. 本地文件

| 文件 | 内容 |
|---|---|
| `docs/papers/GRPO_2509.25808.pdf` | ⭐ AR3PO 论文（12 页） |
| `docs/papers/GRPO_2506.02177.pdf` | GRESO 论文（备选） |
| `training/ar3po_trainer.py` | ⭐ **可运行的 AR3PO 实现**（`AR3POConfig` + `AR3POTrainer`） |
| `tools/trl_contract_probe.py` | 探测 TRL `_generate` 数据契约的探针 |
| `tools/ar3po_smoke_test.py` | AR3PO 端到端验证（基线 vs AR3PO 对比） |
| `docs/GRPO_AR3PO_选型与实现方案.md` | 本文档 |

