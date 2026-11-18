# Qwen3 Thinking SFT 数据

本目录的数据由 `tools/build_sft_thinking_data.py` 从
`data/sft/math_sft_demo.jsonl` 非破坏性生成，原文件不会被覆盖。

## 文件

- `math_sft_demo_thinking.jsonl`：通过结构和长度检查的训练数据，共 1978 条。
- `math_sft_demo_thinking.rejected.ndjson`：被过滤的 22 条记录及原因，只用于审计，不参与训练。特意不使用 `.jsonl` 后缀，避免训练器递归加载。
- `math_sft_demo_thinking.report.json`：输入哈希、tokenizer、长度限制和过滤统计。

每条 assistant 标签采用以下格式：

```text
<think>
完整解题过程
</think>

Final answer: \boxed{最终答案}
```

## 质量规则

- 只接受一问一答且角色合法的记录。
- 最终答案必须是最后一个完整、括号平衡的 `\boxed{...}`。
- 过滤 `nan`、空答案、缺少 boxed、可疑翻译指令和自相矛盾提示。
- 拒绝包含聊天模板保留 token 或已有 thinking 标签的记录，避免嵌套和模板注入。
- 使用本地 Qwen3-8B tokenizer 验证 thinking token 恰好各出现一次且顺序正确。
- 按当前 SFT 的 2048 token 上限过滤超长记录，保证最终答案不会被预处理截掉。

这些规则保证结构、模板和长度质量；由于原数据没有独立 gold solution，不能仅靠规则证明每一步数学推导都正确。被过滤记录全部保留，便于后续人工复核或使用独立 verifier 再清洗。

## 训练

不要把本文件复制进 `data/sft/`，否则目录加载会同时读入新旧数据。正式训练时将
`scripts/run_sft.sh` 中的参数改为：

```bash
--train_file_dir ./data/sft_thinking
```

并保持：

```bash
--enable_thinking true
```

重新生成时必须指定一个尚不存在的输出路径；构建工具默认拒绝覆盖任何已有产物。
