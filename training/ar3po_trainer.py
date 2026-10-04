"""AR3PO: Adaptive Rollout and Response Reuse Policy Optimization —— TRL 实现。

论文：Improving Sampling Efficiency in RLVR through Adaptive Rollout and Response Reuse
      Yuheng Zhang et al., arXiv:2509.25808

本文件只实现论文的 **adaptive rollout**（自适应 rollout）部分。
**response reuse**（响应复用）由 TRL 自带能力负责，见文末说明。

===========================================================================
一、TRL 0.29.1 的实际数据契约（已由 tools/trl_contract_probe.py 实测确认）
===========================================================================
1. 训练 sampler 用 `RepeatSampler(mini_repeat_count=num_generations)`，
   所以进入 `_generate(prompts)` 的 prompts **已经**把每个唯一问题重复了 G 次：
       len(prompts) == generation_batch_size == G * (唯一问题数)
   且同一问题的 G 个副本是**连续**排列的（第 i 组 = prompts[i*G : (i+1)*G]）。

2. `_generate` 对**每个收到的 prompt 生成 1 条** completion（HF 与 vLLM 路径一致），
   因此 len(completion_ids) == len(prompts)。

3. 上游用 `rewards.view(-1, num_generations)` 做组内归一化，
   所以必须返回**长度恰好等于 len(prompts)** 的扁平列表，且按组顺序排列。

4. `tool_mask` 会被乘进 loss 的 mask
   （`GRPOTrainer._compute_loss`: mask = completion_mask * tool_mask），
   因此把某条序列的 tool_mask 置 0 == **对该序列停梯度**
   —— 这正是论文 Option II 需要的机制。

===========================================================================
二、本实现的策略：S 阶段自适应 rollout + "整体复制补齐 + 停梯度"
===========================================================================
· 第 1 轮：每个唯一问题只生成 k = G/S 条
· 判正确：用第 `ar3po_correctness_reward_idx` 个奖励函数判定（默认第 0 个 = 准确性）
· 后续轮：只对「目前仍无正确答案」的问题继续生成 k 条
· 组装：每组拼到恰好 G 条；不足的部分用该组第 1 轮的 k 条**整体复制**补齐，
        复制件的 tool_mask 置 0（不参与梯度）

  为什么"整体复制"是安全的（已数值验证）：
      设一组的 reward 为 r_1..r_k，整体复制后为 2k 个点。
      · **mean 精确不变**（复制不改变均值）
      · **std 会变**：std_{2k} / std_k = sqrt( 2(k-1) / (2k-1) )
        这是一个**只依赖 k、与数据无关的常数**（k=2 时 = 0.8165）
      因此：组内 advantage 的**符号与相对大小完全保持**，但**整体幅度**会被统一缩放。
         · 若 scale_rewards="group"（默认）：advantage 被放大 sqrt((2k-1)/(2(k-1)))
           （k=2 → ×1.2247，k=3 → ×1.118，k=4 → ×1.080）
         · 若 scale_rewards="none"：advantage = reward − mean，
           **真实样本的 advantage 与"只有 k 条"时完全相同** ← 推荐配置

  ⚠️ 因此使用本 trainer 时**强烈建议设 scale_rewards="none"**：
     这样 padding 在数学上是精确的；同时这也正好是 Dr. GRPO 所主张的
     "不要用组内 std 归一化"，一举两得。
     若你坚持用默认的 "group"，本类会打印一条警告（幅度被统一放大，方向不受影响）。

===========================================================================
三、与论文的对应关系
===========================================================================
· 论文 Algorithm 1 第 5-9 行（多阶段 rollout、有正确答案即移出池）→ `_generate` 主循环
· 论文 3.2 节 response reuse → **未落地**，原因有两层（均已核实到 TRL 源码）：
    (a) TRL 0.29.1 的 replay trainer 不切分 `tool_mask`
        （`slice_group_data` / `update_replay_buffer` 里完全没有 tool_mask），
        而本实现正是靠 tool_mask=0 让复制样本停梯度 → 混用会静默破坏约束；
    (b) 更根本的是：父类产出的 `grouped_std_rewards` 是 **1D** 张量
        （`GRPOTrainer`: `rewards.view(-1, G).std(dim=1).repeat_interleave(G)`），
        而 replay trainer 用 `group_std_rewards.max(dim=0).values > 0` 判断哪些组有方差
        —— 对 1D 张量这返回**标量**，于是：
            · 批次里存在任何有方差的组 → `num_groups_to_replace = 0` → **静默 no-op**
            · 整批全零方差 → 采样空 buffer 得到 None → `for item in None` → **崩溃**
        即它在标准 GRPO 下也是坏的（要么不生效要么崩）。
  → 因此 `grpo_training.py` 在 adaptive rollout 开启时强制禁用该 mixin。
· 论文 Option II（对复用样本停梯度）→ 本文件用 tool_mask 实现了同样的机制
"""

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from loguru import logger
from trl import GRPOConfig, GRPOTrainer


@dataclass
class AR3POConfig(GRPOConfig):
    """在 GRPOConfig 之上增加 AR3PO 的开关。"""

    ar3po_stages: int = field(
        default=2,
        metadata={
            "help": "自适应 rollout 的阶段数 S。每个唯一问题最多生成 S*k 条，"
                    "k = num_generations / S。设为 1 即退化为标准 GRPO。论文用 S=2。"
        },
    )
    ar3po_correctness_reward_idx: int = field(
        default=0,
        metadata={
            "help": "用第几个奖励函数判定「答对了」。默认 0，即 reward_funcs 的第一个"
                    "（通常是 accuracy_reward）。切勿用 format_reward 判定正确性。"
        },
    )
    replay_buffer_size: int = field(
        default=0,
        metadata={
            "help": "TRL replay buffer 的容量（0 = 关闭，**默认且推荐保持 0**）。"
                    "TRL 0.29.1 的实验 trainer 有两个问题：(a) 不切分 tool_mask，"
                    "会丢掉 AR3PO 的停梯度约束；(b) 对父类产出的 1D group_std_rewards "
                    "用 max(dim=0) 得到标量，导致有方差时静默 no-op、全零方差时崩溃。"
                    "即它在标准 GRPO 下也不可靠。详见本文件顶部说明。"
        },
    )


class AR3POTrainer(GRPOTrainer):
    """带自适应 rollout 的 GRPO Trainer。

    用法::

        from training.ar3po_trainer import AR3POConfig, AR3POTrainer

        args = AR3POConfig(
            num_generations=8,      # G
            ar3po_stages=2,         # S  → k = 4
            max_completion_length=3072,
            loss_type="dapo",
            scale_rewards="group",
            ...
        )
        trainer = AR3POTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[accuracy_reward, format_reward],
            args=args,
            train_dataset=train_dataset,
        )
        trainer.train()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ar3po_inputs: Optional[list[dict[str, Any]]] = None

        stages = max(1, int(getattr(self.args, "ar3po_stages", 1) or 1))
        if stages > 1 and self.num_generations % stages != 0:
            raise ValueError(
                f"AR3PO: num_generations ({self.num_generations}) 必须能被 "
                f"ar3po_stages ({stages}) 整除。推荐 num_generations=8, ar3po_stages=2。"
            )
        if stages > 1 and getattr(self.args, "scale_rewards", "group") == "group":
            k = self.num_generations // stages
            import math

            factor = math.sqrt((2 * k - 1) / (2 * (k - 1))) if k > 1 else float("inf")
            logger.warning(
                "AR3PO: 检测到 scale_rewards='group'。补齐样本会让组内 std 变为原来的 "
                "sqrt(2(k-1)/(2k-1))，使 advantage 幅度被统一放大 {:.4f} 倍（方向不变）。"
                "建议改用 scale_rewards='none'，此时补齐在数学上完全精确。",
                factor,
            )

    # ------------------------------------------------------------------ #
    # 1. 挂上 inputs，让 _generate 能拿到答案列来判对错
    # ------------------------------------------------------------------ #
    def _generate_and_score_completions(self, inputs):
        self._ar3po_inputs = inputs
        try:
            return super()._generate_and_score_completions(inputs)
        finally:
            self._ar3po_inputs = None

    # ------------------------------------------------------------------ #
    # 2. 自适应 rollout 主体
    # ------------------------------------------------------------------ #
    def _generate(self, prompts: list):
        G = self.num_generations
        S = max(1, int(getattr(self.args, "ar3po_stages", 1) or 1))

        # 各种不安全情形一律回退到原始实现，保证永不炸
        if S <= 1 or not self.model.training or self._ar3po_inputs is None or self.tools:
            return super()._generate(prompts)

        groups = self._split_groups(prompts, G)
        if groups is None:
            return super()._generate(prompts)

        k = G // S
        n_groups = len(groups)

        per_group: list[list[dict]] = [[] for _ in range(n_groups)]  # 每组的多轮结果
        pending = list(range(n_groups))

        for _stage in range(S):
            if not pending:
                break
            # 每个待处理问题取第 1 个副本作 prompt，重复 k 次
            cur_prompts = [prompts[groups[gi][0]] for gi in pending for _ in range(k)]
            with self._quiet_metrics():
                rollout = super()._generate(cur_prompts)

            for j, gi in enumerate(pending):
                per_group[gi].append(self._take_slice(rollout, j * k, (j + 1) * k))

            # 已有正确答案的问题退出（AR3PO：至少一条正确就不再追加生成）
            still_pending = []
            for gi in pending:
                sl = per_group[gi][-1]
                if not self._has_correct(sl, self._ar3po_inputs[groups[gi][0]]):
                    still_pending.append(gi)
            pending = still_pending

        return self._assemble(per_group, n_groups, G)

    # ------------------------------------------------------------------ #
    # 3. 工具函数
    # ------------------------------------------------------------------ #
    @staticmethod
    def _split_groups(prompts: list, G: int):
        """把已重复的 prompts 按「连续 G 个相同」切组。结构不符则返回 None。"""
        if G <= 1 or len(prompts) % G != 0:
            return None
        groups = []
        for start in range(0, len(prompts), G):
            block = prompts[start : start + G]
            if all(p == block[0] for p in block):
                groups.append(list(range(start, start + G)))
            else:
                return None
        return groups

    @contextmanager
    def _quiet_metrics(self):
        """子调用期间屏蔽 metrics，避免同一 step 的日志被重复记录。"""
        saved = self._metrics
        self._metrics = defaultdict(lambda: defaultdict(list))
        try:
            yield
        finally:
            self._metrics = saved

    @staticmethod
    def _take_slice(rollout: tuple, lo: int, hi: int) -> dict:
        """从 _generate 的 7 元组里切出 [lo, hi) 这一段。"""
        prompt_ids, completion_ids, tool_mask, completions, _n, logprobs, _extra = rollout
        return {
            "prompt_ids": list(prompt_ids[lo:hi]),
            "completion_ids": list(completion_ids[lo:hi]),
            "tool_mask": None if tool_mask is None else list(tool_mask[lo:hi]),
            "completions": list(completions[lo:hi]),
            "logprobs": None if logprobs is None else list(logprobs[lo:hi]),
        }

    def _has_correct(self, sl: dict, row: dict) -> bool:
        """用指定奖励函数判断这 k 条里有没有正确答案。"""
        idx = int(getattr(self.args, "ar3po_correctness_reward_idx", 0))
        if idx >= len(self.reward_funcs):
            return False
        fn = self.reward_funcs[idx]
        if not callable(fn):
            return False

        n = len(sl["completions"])
        keys = [key for key in row if key not in ("prompt", "completion", "completion_ids")]
        kwargs = {key: [row[key]] * n for key in keys}
        try:
            out = fn(
                prompts=[row["prompt"]] * n,
                completions=sl["completions"],
                completion_ids=sl["completion_ids"],
                **kwargs,
            )
        except Exception:
            # 奖励函数签名不兼容 → 保守认为"没有正确答案"，即继续为该题生成
            return False
        return any((0.0 if r is None else float(r)) > 0 for r in out)

    def _assemble(self, per_group: list, n_groups: int, G: int) -> tuple:
        """把各组的多轮结果拼成「长度恰好 G*n_groups」的扁平列表。"""
        # 先判断是否真的需要补齐（决定 tool_mask 是否必须存在）
        real_counts = [sum(len(sl["completion_ids"]) for sl in g) for g in per_group]
        any_pad = any(c < G for c in real_counts)
        any_tool_mask = any(
            sl["tool_mask"] is not None for g in per_group for sl in g
        )
        any_logprobs = any(sl["logprobs"] is not None for g in per_group for sl in g)
        need_mask = any_pad or any_tool_mask

        flat = {"prompt_ids": [], "completion_ids": [], "completions": [],
                "tool_mask": [], "logprobs": []}
        total_tokens = 0

        for gi in range(n_groups):
            slices = per_group[gi]
            n_real = real_counts[gi]
            n_pad = G - n_real

            # (a) 真实生成的部分
            for sl in slices:
                flat["prompt_ids"].extend(sl["prompt_ids"])
                flat["completion_ids"].extend(sl["completion_ids"])
                flat["completions"].extend(sl["completions"])
                if need_mask:
                    if sl["tool_mask"] is None:
                        flat["tool_mask"].extend(
                            [[1] * len(ids) for ids in sl["completion_ids"]]
                        )
                    else:
                        flat["tool_mask"].extend(sl["tool_mask"])
                if any_logprobs:
                    if sl["logprobs"] is None:
                        flat["logprobs"].extend(
                            [[0.0] * len(ids) for ids in sl["completion_ids"]]
                        )
                    else:
                        flat["logprobs"].extend(sl["logprobs"])
                total_tokens += sum(len(ids) for ids in sl["completion_ids"])

            # (b) 用第 1 轮的 k 条整体复制补齐（tool_mask=0 → 停梯度）
            if n_pad > 0:
                base = slices[0]
                m = len(base["completion_ids"])
                if m == 0:
                    raise RuntimeError("AR3PO: 第 1 轮未产出样本，无法补齐。")
                for t in range(n_pad):
                    src = t % m
                    flat["prompt_ids"].append(base["prompt_ids"][src] if base["prompt_ids"] else [])
                    flat["completion_ids"].append(base["completion_ids"][src])
                    flat["completions"].append(base["completions"][src])
                    flat["tool_mask"].append([0] * len(base["completion_ids"][src]))
                    if any_logprobs:
                        lp = base["logprobs"][src] if base["logprobs"] else []
                        flat["logprobs"].append(list(lp))

        # 结构校验：这是与 TRL 上游 view(-1, G) 的硬约束
        assert len(flat["completion_ids"]) == n_groups * G, (
            f"AR3PO 组装长度 {len(flat['completion_ids'])} != {n_groups * G}"
        )

        # 注意：num_items_in_batch 必须是 0 维 tensor。
        # TRL 的 shuffle_sequence_dict 只对 tensor 做 0 维特判，传 Python int 会在
        # permute() 里抛 "int object is not subscriptable"。
        return (
            flat["prompt_ids"],
            flat["completion_ids"],
            flat["tool_mask"] if need_mask else None,
            flat["completions"],
            torch.tensor(total_tokens, dtype=torch.long),
            flat["logprobs"] if any_logprobs else None,
            {},
        )
