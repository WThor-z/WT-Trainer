"""Custom optimizer implementations for WT Trainer.

This module provides custom optimizer implementations for various training methods including:
- LoRA+: A method that applies different learning rates to different LoRA layers
- BAdam: Block-wise optimization methods
- Adam-mini: An efficient optimizer for large language models
"""

import logging
from typing import TYPE_CHECKING

import torch
from transformers.modeling_utils import is_deepspeed_zero3_enabled  # type: ignore
from transformers import Trainer
from transformers.modeling_utils import is_fsdp_enabled
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names

if TYPE_CHECKING:
    from transformers import PreTrainedModel  # noqa: F401
    from trl import AutoModelForCausalLMWithValueHead  # noqa: F401

    from wt_trainer.args import FinetuningArguments  # noqa: F401
    from wt_trainer.args import TrainingArguments  # noqa: F401

logger = logging.getLogger(__name__)


def _get_decay_parameter_names(model: "PreTrainedModel") -> list[str]:
    """Return a list of names of parameters with weight decay.

    Args:
        model: The pre-trained model.

    Returns:
        A list of parameter names with weight decay (weights in non-layernorm layers).
    """
    decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)  # type: ignore
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    return decay_parameters


def create_custom_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> torch.optim.Optimizer | None:
    """Create a custom optimizer based on the provided arguments.

    Args:
        model: The pre-trained model.
        training_args: Training arguments.
        finetuning_args: Fine-tuning arguments.

    Returns:
        A custom optimizer or None if no custom optimizer is needed.
    """
    if finetuning_args.loraplus_lr_ratio:
        return _create_loraplus_optimizer(model, training_args, finetuning_args)
    if finetuning_args.use_badam:
        return _create_badam_optimizer(model, training_args, finetuning_args)
    if finetuning_args.use_adam_mini:
        return _create_adam_mini_optimizer(model, training_args)
    return None


def _create_loraplus_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> torch.optim.Optimizer | None:
    """Create a LoRA+ optimizer.

    Args:
        model: The pre-trained model.
        training_args: Training arguments.
        finetuning_args: Fine-tuning arguments.

    Returns:
        A LoRA+ optimizer.

    Raises:
        ValueError: If loraplus_lr_ratio is not specified.
    """
    default_lr = training_args.learning_rate
    if finetuning_args.loraplus_lr_ratio is not None:
        loraplus_lr = training_args.learning_rate * finetuning_args.loraplus_lr_ratio
    else:
        raise ValueError("Please specify loraplus_lr_ratio.")
    embedding_lr = finetuning_args.loraplus_lr_embedding

    decay_param_names = _get_decay_parameter_names(model)
    param_dict: dict[str, list[torch.nn.Parameter]] = {
        "lora_a": [],
        "lora_b": [],
        "lora_b_nodecay": [],
        "embedding": [],
    }
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "lora_embedding_B" in name:
                param_dict["embedding"].append(param)
            elif "lora_B" in name or param.ndim == 1:
                if name in decay_param_names:
                    param_dict["lora_b"].append(param)
                else:
                    param_dict["lora_b_nodecay"].append(param)
            else:
                param_dict["lora_a"].append(param)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        {
            "params": param_dict["lora_a"],
            "lr": default_lr,
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": param_dict["lora_b"],
            "lr": loraplus_lr,
            "weight_decay": training_args.weight_decay,
        },
        {"params": param_dict["lora_b_nodecay"], "lr": loraplus_lr, "weight_decay": 0.0},
        {
            "params": param_dict["embedding"],
            "lr": embedding_lr,
            "weight_decay": training_args.weight_decay,
        },
    ]
    optimizer = optim_class(param_groups, **optim_kwargs)
    logger.info(
        f"Using LoRA+ optimizer with loraplus lr ratio {finetuning_args.loraplus_lr_ratio:.2f}."
    )
    return optimizer


def _create_badam_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
    finetuning_args: "FinetuningArguments",
) -> torch.optim.Optimizer | None:
    """Create a BAdam optimizer.

    Args:
        model: The pre-trained model.
        training_args: Training arguments.
        finetuning_args: Fine-tuning arguments.

    Returns:
        A BAdam optimizer.
    """
    decay_params, nodecay_params = [], []
    decay_param_names = _get_decay_parameter_names(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name in decay_param_names:
                decay_params.append(param)
            else:
                nodecay_params.append(param)

    optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(training_args)
    param_groups = [
        {"params": nodecay_params, "weight_decay": 0.0},
        {"params": decay_params, "weight_decay": training_args.weight_decay},
    ]

    if finetuning_args.badam_mode == "layer":
        from badam import BlockOptimizer  # pylint: disable=import-outside-toplevel

        base_optimizer = optim_class(param_groups, **optim_kwargs)
        optimizer = BlockOptimizer(
            base_optimizer=base_optimizer,
            named_parameters_list=list(model.named_parameters()),
            block_prefix_list=None,
            switch_block_every=finetuning_args.badam_switch_interval,
            start_block=finetuning_args.badam_start_block,
            switch_mode=finetuning_args.badam_switch_mode,
            verbose=finetuning_args.badam_verbose,
            ds_zero3_enabled=is_deepspeed_zero3_enabled(),
        )
        logger.info(
            f"Using BAdam optimizer with layer-wise update, switch mode is {finetuning_args.badam_switch_mode}, "
            f"switch block every {finetuning_args.badam_switch_interval} steps, "
            f"default start block is {finetuning_args.badam_start_block}"
        )

    elif finetuning_args.badam_mode == "ratio":
        from badam import BlockOptimizerRatio  # pylint: disable=import-outside-toplevel

        assert finetuning_args.badam_update_ratio > 1e-6
        optimizer = BlockOptimizerRatio(
            param_groups=param_groups,
            named_parameters_list=list(model.named_parameters()),
            update_ratio=finetuning_args.badam_update_ratio,
            mask_mode=finetuning_args.badam_mask_mode,
            verbose=finetuning_args.badam_verbose,
            include_embedding=False,
            **optim_kwargs,
        )
        logger.info(
            f"Using BAdam optimizer with ratio-based update, update ratio is {finetuning_args.badam_update_ratio}, "
            f"mask mode is {finetuning_args.badam_mask_mode}"
        )
    else:
        raise ValueError(f"Unsupported BAdam mode: {finetuning_args.badam_mode}")

    return optimizer


def _create_adam_mini_optimizer(
    model: "PreTrainedModel",
    training_args: "TrainingArguments",
) -> torch.optim.Optimizer | None:
    """Create an Adam-mini optimizer.

    Args:
        model: The pre-trained model.
        training_args: Training arguments.

    Returns:
        An Adam-mini optimizer.
    """
    from adam_mini import Adam_mini  # pylint: disable=import-outside-toplevel

    hidden_size = getattr(model.config, "hidden_size", None)
    num_q_head = getattr(model.config, "num_attention_heads", None)
    num_kv_head = getattr(model.config, "num_key_value_heads", None)

    optimizer = Adam_mini(
        named_parameters=model.named_parameters(),
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        model_sharding=is_fsdp_enabled() or is_deepspeed_zero3_enabled(),  # type: ignore
        dim=hidden_size,
        n_heads=num_q_head,
        n_kv_heads=num_kv_head,
    )
    logger.info("Using Adam-mini optimizer.")
    return optimizer
