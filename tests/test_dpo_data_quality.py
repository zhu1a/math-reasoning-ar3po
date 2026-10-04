import pytest
from datasets import Dataset

from training.dpo_training import validate_preference_dataset


def test_rejects_placeholder_preference_pairs():
    dataset = Dataset.from_list([
        {
            "chosen": f"逐步推理…\n第一步\n第二步\n\\boxed{{{i}}}",
            "rejected": "随便写点东西",
        }
        for i in range(20)
    ])

    with pytest.raises(ValueError, match="占位/退化数据"):
        validate_preference_dataset(dataset)


def test_accepts_diverse_rollout_pairs():
    dataset = Dataset.from_list([
        {
            "chosen": f"Reasoning path {i}; \\boxed{{{i}}}",
            "rejected": f"Alternative incorrect reasoning {i}; \\boxed{{{i + 1}}}",
        }
        for i in range(20)
    ])

    validate_preference_dataset(dataset)
