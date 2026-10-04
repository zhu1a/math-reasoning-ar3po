# -*- coding: utf-8 -*-
"""显存硬地板计算器 —— 只用**精确算术**排除"数学上不可能"的配置。

用法::

    python3 tools/memory_planner.py                      # 对比所有预设
    python3 tools/memory_planner.py --gpu 5090-32G
    python3 tools/memory_planner.py --model Qwen3-8B --gpu A100-40G --no-vllm

===========================================================================
这个工具能回答什么、不能回答什么（重要）
===========================================================================
【能确定地回答】某个配置**是否数学上不可能**。

    硬地板 = 权重 × 权重份数 + KV cache
      · 权重份数：用 vLLM colocate 时是 2（训练模型一份 + vLLM 一份）
                  关掉 vLLM 时是 1
      · 权重和 KV 都是**精确算术**（参数量 × dtype 字节数；层数 × KV头 × head_dim）
      · 硬地板**不含**激活、CUDA 上下文、框架开销、显存碎片

    地板 > 显卡容量 ⇒ **必定 OOM**。不需要知道激活占多少，也不用浪费一天去试。
    这是本工具唯一有确定性的结论，也是最省时间的那一类结论。

【不能回答】地板装得下时，**激活到底够不够**。

    激活内存受 gradient checkpointing、attention 实现（flash/sdpa/eager）、
    logits 计算范围、batch/序列长度、碎片化影响，差异可达数倍 —— 任何静态公式都是猜。
    所以本工具**故意不输出"总计峰值"和通过性判定**：那种数字的精度是假的，
    只会让人在错误前提上做决定。**激活必须实测**（见文末）。

===========================================================================
为什么值得存在（而不是"算一次就完了"）
===========================================================================
它把"试试看能不能跑"（可能浪费一整天装环境+下模型）变成"1 秒判断值不值得试"。
典型例子：Qwen3-8B + vLLM 在 32GB 卡上，硬地板 32.75 GiB > 31 GiB 容量，
**不用装环境、不用下模型，就能确定这条路是死的**。
"""

import argparse
from dataclasses import dataclass

GiB = 1024 ** 3

#: CUDA 上下文是刚性开销，留 1 GiB 参考（实际 0.5~1.5）
CUDA_CONTEXT_GIB = 1.0


# --------------------------------------------------------------------------- #
# 模型预设（架构数据来自 Qwen3 Technical Report, arXiv:2505.09388, Table 1）
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    name: str
    layers: int
    hidden: int
    inter: int
    vocab: int
    n_kv: int
    head_dim: int
    tie: bool

    @property
    def params(self) -> int:
        emb = self.vocab * self.hidden
        per_layer = (
            self.hidden * self.hidden                        # q_proj
            + self.hidden * (self.n_kv * self.head_dim) * 2  # k_proj, v_proj
            + self.hidden * self.hidden                      # o_proj
            + self.hidden * self.inter * 3                   # gate, up, down
        )
        return emb * (1 if self.tie else 2) + per_layer * self.layers

    def weight_gib(self, bytes_per_param: int = 2) -> float:
        return self.params * bytes_per_param / GiB

    def kv_kb_per_token(self, bytes_per_elem: int = 2) -> float:
        return 2 * self.n_kv * self.head_dim * bytes_per_elem * self.layers / 1024

    def kv_gib(self, tokens: int, bytes_per_elem: int = 2) -> float:
        return self.kv_kb_per_token(bytes_per_elem) * tokens * 1024 / GiB


# 参数量核对：Qwen3-8B 反推 8.19B，技术报告标称 8.2B ✓
MODELS = {
    "Qwen3-8B":   ModelSpec("Qwen3-8B",   36, 4096, 12288, 151669, 8, 128, False),
    "Qwen3-4B":   ModelSpec("Qwen3-4B",   36, 2560, 9728,  151669, 8, 128, True),
    "Qwen3-1.7B": ModelSpec("Qwen3-1.7B", 28, 2048, 6144,  151669, 8, 128, True),
    "Qwen3-14B":  ModelSpec("Qwen3-14B",  40, 5120, 13824, 151669, 8, 128, False),
}

#: 保守可用显存（GiB）：已扣掉驱动/显示/保留部分
GPUS = {
    "4090-24G":  (23.5, "Ada sm_89"),
    "5090-32G":  (31.0, "Blackwell sm_120 软件栈风险"),
    "A100-40G":  (39.0, "Ampere sm_80"),
    "L40S-48G":  (47.0, "Ada sm_89"),
    "A100-80G":  (79.0, "Ampere sm_80"),
    "H100-80G":  (79.0, "Hopper sm_90"),
}


def floor_gib(model: str, g: int, seq: int, copies: int) -> dict:
    m = MODELS[model]
    w = m.weight_gib()
    kv = m.kv_gib(g * seq) if copies > 1 else 0.0   # 只有 vLLM 才需要它的 KV
    return {"w": w, "kv": kv, "floor": w * copies + kv}


def main():
    ap = argparse.ArgumentParser(description="显存硬地板计算器（仅做必要条件判断）")
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--gpu", action="append", choices=list(GPUS))
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_completion_length", type=int, default=2048)
    ap.add_argument("--no-vllm", action="store_true",
                    help="关掉 vLLM（HF generate）：权重只有 1 份，但 rollout 慢 3~10x")
    args = ap.parse_args()

    copies = 1 if args.no_vllm else 2
    models = [args.model] if args.model else list(MODELS)
    gpus = args.gpu if args.gpu else list(GPUS)
    mode = "HF generate（1 份权重）" if copies == 1 else "vLLM colocate（2 份权重）"
    tokens = args.num_generations * args.max_completion_length

    print("=" * 96)
    print(f"硬地板 = 权重×{copies} + KV({args.num_generations}×{args.max_completion_length}"
          f" = {tokens} tokens)     模式: {mode}")
    print("（不含激活/框架开销；地板装得下 ≠ 一定跑得起来，激活必须实测）")
    print("=" * 96)

    for gpu in gpus:
        cap, arch = GPUS[gpu]
        print(f"\n### {gpu}  保守可用 {cap} GiB  [{arch}]")
        print(f"{'模型':<12}{'权重':>8}{'×份':>6}{'KV':>8}{'硬地板':>9}"
              f"{'占容量':>8}{'地板后余额':>11}   判定")
        for mo in models:
            r = floor_gib(mo, args.num_generations, args.max_completion_length, copies)
            used = r["floor"] / cap
            left = cap - r["floor"] - CUDA_CONTEXT_GIB
            if r["floor"] > cap:
                verdict = "❌ 确定不可能（地板就超了，无需实测）"
            elif left <= 0:
                verdict = "❌ 几乎不可能（扣掉 CUDA 上下文后没余额）"
            elif left < 4:
                verdict = f"⚠️ 很紧：剩 {left:.1f}GiB 放激活，需实测"
            else:
                verdict = f"🟡 值得一试：剩 {left:.1f}GiB 放激活，需实测"
            print(f"{mo:<12}{r['w']:7.2f}G{f'×{copies}':>6}{r['kv']:7.2f}G"
                  f"{r['floor']:8.2f}G{used:7.1%}{cap - r['floor']:10.2f}G   {verdict}")

    if args.model:
        m = MODELS[args.model]
        print(f"\n注：{args.model}  KV = {m.kv_kb_per_token():.0f} KB/token"
              f"（{m.layers} 层 × {m.n_kv} KV头 × {m.head_dim} 维）。")
        print("    Qwen3-8B 与 Qwen3-4B 的 KV/token **相同** —— 换 4B 只省权重，不省 KV。")

    print("\n" + "-" * 96)
    print("激活怎么实测（跑 2~3 步后打印，这是唯一的可信来源）：")
    print("    import torch")
    print("    print('peak alloc  ', torch.cuda.max_memory_allocated() / 1024**3, 'GiB')")
    print("    print('peak reserve', torch.cuda.max_memory_reserved() / 1024**3, 'GiB')")
    print("  reserved 明显大于 allocated → 有碎片，可试 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")


if __name__ == "__main__":
    main()
