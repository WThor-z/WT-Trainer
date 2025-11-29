"""
Unsloth model utilities for optimizing model loading and PEFT training.

This module provides functionality to load models and apply PEFT training using the Unsloth library
for improved training performance.
"""

import logging
from typing import TYPE_CHECKING

from ..train.args_process import _get_current_device

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel

    from wt_trainer.args import FinetuningArguments, ModelArguments, TrainingArguments


logger = logging.getLogger(__name__)


def load_unsloth_pretrained_model(
    model_args: "ModelArguments", ft_args: "FinetuningArguments"
) -> tuple["PreTrainedModel | None", "object | None"]:
    """Load a pretrained model using Unsloth optimization.

    Args:
        model_args: Model arguments containing model configuration.
        ft_args: Fine-tuning arguments.

    Returns:
        A tuple containing the loaded model and tokenizer, or (None, None) if loading failed.
    """
    from unsloth import FastLanguageModel
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    unsloth_kwargs = {
        "model_name": model_args.model_name_or_path,
        "max_seq_length": model_args.model_max_length or 4096,
        "dtype": model_args.compute_dtype,
        "load_in_4bit": model_args.quantization_bit == 4,
        "token": model_args.hf_hub_token,
        "full_finetuning": ft_args.finetuning_type == "full",
        "device_map": {"": _get_current_device()},
        "rope_scaling": getattr(config, "rope_scaling", None),
        "fix_tokenizer": False,
        "trust_remote_code": model_args.trust_remote_code,
        "use_gradient_checkpointing": ("unsloth",),
    }
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(**unsloth_kwargs)
    except NotImplementedError:
        logger.warning(
            "Unsloth does not support model type {}.".format(getattr(config, "model_type", None))
        )
        model = None
        model_args.use_unsloth = False
        tokenizer = None

    logger.info("Successfully loaded model with unsloth")
    return model, tokenizer


def get_unsloth_peft_model(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    ft_args: "FinetuningArguments",
    train_args: "TrainingArguments",
) -> "PreTrainedModel | None":
    """Apply PEFT training to a model using Unsloth optimization.

    Args:
        model: The base model to apply PEFT to.
        model_args: Model arguments containing model configuration.
        ft_args: Fine-tuning arguments.
        train_args: Training arguments.

    Returns:
        The model with PEFT applied, or None if the operation failed.
    """
    from unsloth import FastLanguageModel

    peft_kwargs = {
        "model": model,
        "max_seq_length": model_args.model_max_length,
        "r": ft_args.lora_rank,
        "target_modules": (
            ft_args.lora_target
            if ft_args.lora_target != ["all"]
            else [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        ),
        "lora_alpha": ft_args.lora_alpha,
        "lora_dropout": ft_args.lora_dropout,
        "bias": "none",
        "use_gradient_checkpointing": ("unsloth",),
        "random_state": (train_args.seed,),
        "use_rslora": ft_args.use_rslora,
        "use_dora": ft_args.use_dora,
        "modules_to_save": ft_args.additional_target,
    }
    try:
        model = FastLanguageModel.get_peft_model(**peft_kwargs)
    except NotImplementedError:
        logger.warning("Unable to use unsloth's PEFT feature. Current model is not supported")
        model = None
        model_args.use_unsloth = False

    logger.info("Successfully applied unsloth's PEFT feature")
    return model