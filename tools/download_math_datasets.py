"""通过 HF 镜像下载数学数据集（huggingface.co 被 DNS 屏蔽，hf-mirror.com 可达）。

下载内容：
  1. BytedTsinghua-SIA/DAPO-Math-17k   —— AR3PO/DAPO 论文的训练集（阶段1 复现）
  2. HuggingFaceH4/MATH-500            —— 评测
  3. Maxwell-Jia/AIME_2024             —— 评测
  4. AI-MO/NuminaMath-1.5              —— 中文/大规模（阶段2 中文迁移）
"""
import os
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import hf_hub_download  # noqa: E402

RAW = Path("data/_raw")
RAW.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("BytedTsinghua-SIA/DAPO-Math-17k", "data/dapo-math-17k.parquet", "dapo-math-17k.parquet"),
    ("HuggingFaceH4/MATH-500", "test.jsonl", "math500-test.jsonl"),
    ("Maxwell-Jia/AIME_2024", "aime_2024_problems.parquet", "aime2024.parquet"),
    ("AI-MO/NuminaMath-1.5", "data/train-00000-of-00003.parquet", "numina-train-0.parquet"),
    ("AI-MO/NuminaMath-1.5", "data/train-00001-of-00003.parquet", "numina-train-1.parquet"),
    ("AI-MO/NuminaMath-1.5", "data/train-00002-of-00003.parquet", "numina-train-2.parquet"),
]

ok, fail = [], []
for repo, filename, outname in TARGETS:
    dest = RAW / outname
    if dest.exists() and dest.stat().st_size > 0:
        print(f"[skip] {outname} 已存在 ({dest.stat().st_size/1e6:.1f} MB)", flush=True)
        ok.append(outname)
        continue
    print(f"[down] {repo}/{filename} -> {outname}", flush=True)
    try:
        p = hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset",
                            local_dir=str(RAW), endpoint="https://hf-mirror.com")
        src = Path(p)
        if src.resolve() != dest.resolve():
            src.replace(dest)
        print(f"[ ok ] {outname} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)
        ok.append(outname)
    except Exception as e:
        print(f"[FAIL] {outname}: {type(e).__name__}: {e}", flush=True)
        fail.append(outname)

print("\n=== 下载汇总 ===")
print("成功:", ok)
print("失败:", fail)
sys.exit(1 if fail else 0)
