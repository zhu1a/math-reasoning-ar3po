# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: Train R1 model with GRPO rl algo.
"""
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import re
from datasets import load_dataset
import torch
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from transformers.integrations import is_deepspeed_zero3_enabled
from trl import GRPOConfig, ModelConfig, TrlParser
from peft import LoraConfig, TaskType, get_peft_model

# Support both ``python -m training.grpo_training`` and the historical
# ``python training/grpo_training.py`` entrypoint.  In the latter form Python
# otherwise only adds ``training/`` (not the repository root) to sys.path, so
# importing the package below fails before argument parsing starts.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from training.ar3po_trainer import AR3POConfig, AR3POTrainer
from training.math_rewards import build_reward_funcs

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with GRPO
    """
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "The tokenizer for weights initialization."}
    )
    # Dataset arguments
    dataset_name: Optional[str] = field(
        default="openai/gsm8k",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(
        default=None, metadata={"help": "Directory containing training files for local datasets."}
    )
    train_samples: Optional[int] = field(default=-1, metadata={"help": "Number of samples to train on, -1 for all"})
    subset_name: Optional[str] = field(default="main",
                                       metadata={"help": "Subset name, e.g., 'default', 'main'. default is 'default'"})
    dataset_splits: Optional[str] = field(default="train", metadata={"help": "Split name"})
    preprocessing_num_workers: Optional[int] = field(default=10,
                                                     metadata={"help": "Number of workers for preprocessing"})
    # 奖励权重（accuracy 固定为 1.0，不暴露）
    format_weight: float = field(
        default=0.3,
        metadata={"help": "规范性奖励的权重。建议 ≤0.3×accuracy：权重过高时，难题上 "
                          "accuracy 恒为 0，模型会转去刷格式分（reward hacking 入口）。"}
    )
    length_weight: float = field(
        default=1.0,
        metadata={"help": "DAPO 软长度惩罚的权重（函数本身返回 0 ~ -1）。"}
    )
    # QLoRA arguments
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})


def get_checkpoint(training_args: GRPOConfig):
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    return last_checkpoint


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def grpo_train(
        model_args: ModelConfig, script_args: ScriptArguments, training_args: GRPOConfig
):
    # Add distributed training initialization
    is_main_process = training_args.local_rank in [-1, 0]

    # Only log on main process
    if is_main_process:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Model parameters {model_args}")
        logger.info(f"Script parameters {script_args}")
        logger.info(f"Training parameters {training_args}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        (
            script_args.tokenizer_name_or_path
            if script_args.tokenizer_name_or_path
            else model_args.model_name_or_path
        ),
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    if script_args.train_file_dir and os.path.exists(script_args.train_file_dir):
        # ⚠️ 预先检查 schema 一致性。
        # load_dataset("json", data_dir=...) 会把目录下【所有】jsonl 合并加载，
        # 若各文件字段不一致会抛出晦涩的 CastError。这里提前拦下来给出清晰提示。
        import glob as _glob
        import json as _json

        files = sorted(_glob.glob(os.path.join(script_args.train_file_dir, "**", "*.jsonl"), recursive=True))
        if not files:
            raise FileNotFoundError(f"{script_args.train_file_dir} 下没有找到任何 .jsonl 文件")
        keysets = {}
        for fp in files:
            with open(fp, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        keysets[fp] = tuple(sorted(_json.loads(line).keys()))
                        break
        uniq = set(keysets.values())
        if len(uniq) > 1:
            detail = "\n".join(f"    {os.path.basename(k)}: {v}" for k, v in keysets.items())
            raise ValueError(
                f"目录 {script_args.train_file_dir} 下的 jsonl 字段不一致，无法合并加载：\n{detail}\n"
                f"  → 请把不同 schema 的数据放到【不同子目录】，然后用 --train_file_dir 精确指定。\n"
                f"  → 本仓库的约定：data/grpo/en（英文）| data/grpo/zh（中文）| data/grpo/proof（证明题）"
            )
        if is_main_process:
            logger.info(f"将加载 {len(files)} 个文件，字段 {list(uniq)[0]}：{[os.path.basename(f) for f in files]}")
        dataset = load_dataset("json", data_dir=script_args.train_file_dir, split="train")
    else:
        # Load from HuggingFace hub
        dataset = load_dataset(script_args.dataset_name, script_args.subset_name, split=script_args.dataset_splits)

    if script_args.train_samples > 0:
        dataset = dataset.shuffle(seed=42).select(range(script_args.train_samples))

    # Prepare dataset
    with training_args.main_process_first(desc="Dataset preparation"):
        # 注意：这里**不注入自定义 SYSTEM_PROMPT**。
        # Qwen3 的原生思考格式是  thinking...<｜end▁of▁thinking｜>，由 tokenizer 的官方 chat template
        # 自动处理（可配合 enable_thinking / /no_think 控制）。强行要求模型改用
        # <think></think> 会让它放弃预训练好的格式，白白浪费容量。
        dataset = dataset.map(
            lambda x: {
                'prompt': [
                    {'role': 'user', 'content': x['question']}
                ],
                'answer': x['answer']
            },
            num_proc=script_args.preprocessing_num_workers,
            desc="Processing dataset" if is_main_process else None,
        )

    # Split dataset
    train_test_split = dataset.train_test_split(test_size=0.1)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    if is_main_process:
        logger.info("*** Initializing model kwargs ***")

    # Model initialization
    torch_dtype = (
        model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    )

    # Set up distributed training config
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ddp = world_size != 1

    # Check for QLoRA compatibility
    if script_args.qlora and is_deepspeed_zero3_enabled():
        logger.warning("ZeRO3 are both currently incompatible with QLoRA.")

    # Check quantization settings
    if model_args.load_in_4bit and model_args.load_in_8bit:
        raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")

    # Set up quantization config
    quantization_config = None
    if script_args.qlora and (model_args.load_in_4bit or model_args.load_in_8bit):
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif model_args.load_in_4bit or model_args.load_in_8bit:
        # Support quantization even without qlora flag
        if is_main_process:
            logger.info(
                f"Quantizing model, load_in_4bit: {model_args.load_in_4bit}, load_in_8bit: {model_args.load_in_8bit}")
        if is_deepspeed_zero3_enabled():
            raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=model_args.load_in_4bit,
            load_in_8bit=model_args.load_in_8bit,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
        )

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        dtype=torch_dtype,
        low_cpu_mem_usage=(not is_deepspeed_zero3_enabled()),
        quantization_config=quantization_config,
    )

    num_gpus = torch.cuda.device_count()
    if ddp:
        model_kwargs["device_map"] = None
    elif num_gpus > 1:
        max_memory = {}
        for i in range(num_gpus):
            gpu_props = torch.cuda.get_device_properties(i)
            total_mem = gpu_props.total_memory
            # 预留20%内存给训练时的梯度、优化器状态等
            usable_mem = int(total_mem * 0.8)
            max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"
        model_kwargs["max_memory"] = max_memory
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = "auto"

    if is_main_process:
        logger.info(f"Using {num_gpus} GPUs")
        logger.info(f"model_kwargs={model_kwargs}")

    config = AutoModelForCausalLM.config_class if hasattr(AutoModelForCausalLM, 'config_class') else None
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=model_args.trust_remote_code,
            revision=model_args.model_revision,
        )
    except Exception:
        config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        **model_kwargs,
    )

    # Patch MoE modules for DeepSpeed ZeRO-3
    model_type = getattr(config, "model_type", None) if config else getattr(model.config, "model_type", None)
    if model_type == "mixtral" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
        set_z3_leaf_modules(model, [MixtralSparseMoeBlock])

    if model_type == "deepseek_v3" and is_deepspeed_zero3_enabled():
        for layer in model.model.layers:
            if 'DeepseekV3MoE' in str(type(layer.mlp)):
                layer.mlp._z3_leaf = True

    if model_type == "qwen3_moe" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])

    if model_type == "qwen3_5_moe" and is_deepspeed_zero3_enabled():
        from deepspeed.utils import set_z3_leaf_modules
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeSparseMoeBlock
        set_z3_leaf_modules(model, [Qwen3_5MoeSparseMoeBlock])

    if is_main_process and hasattr(model, 'hf_device_map'):
        logger.info(f"Model Device Map: {model.hf_device_map.items()}")
    elif is_main_process and num_gpus > 1:
        logger.info("Model Device Map:")
        for name, param in model.named_parameters():
            if hasattr(param, 'device'):
                logger.info(f"  {name}: {param.device}")
                break

    # Configure LoRA if enabled
    if model_args.use_peft:
        if is_main_process:
            logger.info("Fine-tuning method: LoRA(PEFT)")
        # ⚠️ 这里原先是 "LoRA 与 gradient checkpointing 冲突，强制关掉" —— 那是错的。
        # 二者可以共存（PEFT 官方支持），而且在 8B 模型 + 48GB 显存下
        # gradient checkpointing 是**必须开**的，否则必 OOM。
        # 需要注意的只是 LoRA 下要打开 input 的 requires_grad（由 Trainer 处理）。
        if training_args.gradient_checkpointing:
            logger.info("LoRA + gradient checkpointing: 保持开启（8B/48GB 场景必需）")
        target_modules = model_args.lora_target_modules if model_args.lora_target_modules else None
        if target_modules == 'all' or (target_modules and 'all' in target_modules):
            target_modules = find_all_linear_names(model, int4=model_args.load_in_4bit, int8=model_args.load_in_8bit)
        if is_main_process:
            logger.info(f"Peft target_modules: {target_modules}, lora rank: {model_args.lora_r}, ")
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
        )
        model = get_peft_model(model, peft_config)
        # Fixed FP16 ValueError for quantized models
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:
        if is_main_process:
            logger.info("Fine-tuning method: Full parameters training")

    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")

    # 取 Qwen3 的 thinking 结束 token id。
    # 必须用 token id 而不是正则：TRL 用 skip_special_tokens=True 解码，
    # ` thinking` / `response` 是 special token，文本里根本不会出现。
    think_end_token_id = None
    try:
        tid = tokenizer.convert_tokens_to_ids(" response")
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            think_end_token_id = tid
            logger.info(f"检测到 Qwen3 thinking 结束 token: `response` → id={tid}")
    except Exception as e:
        logger.debug(f"未检测到 thinking token（非 Qwen3 模型可忽略）: {e}")

    # 构造奖励函数：整体给分(accuracy) + 分层规范分(format) + DAPO 软长度惩罚(length)
    # 顺序很重要：accuracy 必须在 index 0，AR3POTrainer 靠它判定「全错」的难题。
    reward_funcs, reward_weights = build_reward_funcs(
        max_completion_length=training_args.max_completion_length,
        format_weight=getattr(script_args, "format_weight", 0.3),
        length_weight=getattr(script_args, "length_weight", 1.0),
        think_end_token_id=think_end_token_id,
    )
    # TRL 0.29.x reads reward weights exclusively from GRPOConfig.  Passing
    # ``reward_weights`` to GRPOTrainer.__init__ raises ``TypeError`` because
    # it is not a trainer constructor argument.  Keep the custom CLI knobs as
    # the single source of truth and write the computed values back to config.
    if training_args.reward_weights is not None and list(training_args.reward_weights) != reward_weights:
        logger.warning(
            "--reward_weights 与 --format_weight/--length_weight 不一致；"
            f"使用后者计算出的权重 {reward_weights}。"
        )
    training_args.reward_weights = reward_weights

    # 用 AR3POTrainer：自适应 rollout（难题多采样、简单题省算力）。
    # ar3po_stages=1 时行为与标准 GRPOTrainer 完全一致，方便 A/B 对比。
    #
    # TRL 0.29.1 的实验 replay trainer 是一份复制出来的旧 GRPO 实现：
    #   1) 它仅在 self.tools 非空时保留 tool_mask，而 AR3PO 正是靠 tool_mask=0
    #      让复制补齐的样本停梯度；混用会悄悄破坏 AR3PO 的数学约束。
    #   2) 第一批全零方差且 buffer 为空时，上游会迭代 None 并直接崩溃。
    # 因此 adaptive rollout 开启时必须禁用这个实验 mixin。等 TRL 提供能保留
    # env/tool mask 的稳定实现后再恢复；宁可少一个优化，也不能静默训练错目标。
    trainer_cls = AR3POTrainer
    replay_size = int(getattr(training_args, "replay_buffer_size", 0) or 0)
    ar3po_stages = int(getattr(training_args, "ar3po_stages", 1) or 1)
    if replay_size > 0 and ar3po_stages > 1:
        logger.warning(
            "TRL 0.29.1 replay buffer 与 AR3PO 的停梯度 mask 不兼容，已禁用 replay buffer。"
        )
        training_args.replay_buffer_size = 0
    elif replay_size > 0:
        try:
            from trl.experimental.grpo_with_replay_buffer import GRPOWithReplayBufferTrainer

            class _AR3POWithReplay(AR3POTrainer, GRPOWithReplayBufferTrainer):
                """AR3PO 自适应 rollout + TRL replay buffer。"""

                def update_with_replay_buffer(self, group_advantages, group_std_rewards, *args, **kwargs):
                    # Upstream 0.29.1 ReplayBuffer.sample() returns None for an
                    # empty buffer, but sample_from_replay_buffer iterates it.
                    # An all-zero-variance first batch has nothing to insert,
                    # so there is safely nothing to replace either.
                    if not self.replay_buffer.heap and not (group_std_rewards > 0).any():
                        return None
                    return super().update_with_replay_buffer(
                        group_advantages, group_std_rewards, *args, **kwargs
                    )

            trainer_cls = _AR3POWithReplay
            logger.info(
                f"已叠加 replay buffer（size={training_args.replay_buffer_size}；标准 GRPO 模式）"
            )
        except Exception as e:  # experimental API 变动时安全降级
            logger.warning(f"replay buffer 不可用，退回纯 AR3PO: {type(e).__name__}: {e}")

    trainer = trainer_cls(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset if training_args.eval_strategy != "no" else None,
    )
    logger.info(f"*** Trainer initialized: {trainer_cls.__name__} ***")
    logger.debug(f"Trainer: {trainer}")

    # Training
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        if is_main_process:
            logger.info(f"Checkpoint detected, resuming training at {last_checkpoint}.")

    if is_main_process:
        logger.info(
            f'*** Starting training {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for '
            f'{training_args.num_train_epochs} epochs ***'
        )

    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)

    # Log and save metrics on main process
    if is_main_process:
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info("*** Training complete ***")
        logger.info("*** Save model ***")

    # Save model
    trainer.model.config.use_cache = True
    if is_main_process:
        trainer.save_model(training_args.output_dir)
        logger.info(f"Model saved to {training_args.output_dir}")

    training_args.distributed_state.wait_for_everyone()

    if is_main_process:
        tokenizer.save_pretrained(training_args.output_dir)
        logger.info(f"Tokenizer saved to {training_args.output_dir}")

        # Create model card and save config
        kwargs = {
            "dataset_name": script_args.dataset_name,
            "tags": ["r1", "grpo"],
        }
        trainer.create_model_card(**kwargs)
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    if is_main_process:
        logger.info("*** Training complete! ***")


def main():
    parser = TrlParser((ModelConfig, ScriptArguments, AR3POConfig))
    model_args, script_args, training_args = parser.parse_args_and_config()

    # Run the main training loop
    grpo_train(model_args, script_args, training_args)


if __name__ == "__main__":
    main()
