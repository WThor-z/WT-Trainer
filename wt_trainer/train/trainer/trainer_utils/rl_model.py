"""Reinforcement learning model utilities for training."""

import logging

import torch
from transformers import PreTrainedModel  # noqa: F401
from trl import AutoModelForCausalLMWithValueHead  # noqa: F401

from wt_trainer.args import FinetuningArguments
from wt_trainer.args import ModelArguments
from .tokenizer import load_tokenizer

logger = logging.getLogger(__name__)


def create_ref_model(
    model_args: ModelArguments,
    finetuning_args: FinetuningArguments,
    add_valuehead: bool = False,
) -> PreTrainedModel | AutoModelForCausalLMWithValueHead | None:
    """Create reference model for PPO/DPO training. Evaluation mode is not supported.

    The valuehead parameter is randomly initialized since it is useless for PPO training.

    Args:
        model_args: Model arguments.
        finetuning_args: Fine-tuning arguments.
        add_valuehead: Whether to add value head.

    Returns:
        Reference model or None.
    """
    if finetuning_args.ref_model is not None:
        ref_model_args = ModelArguments.copyfrom(
            model_args,
            model_name_or_path=finetuning_args.ref_model,
            adapter_name_or_path=finetuning_args.ref_model_adapters,
            quantization_bit=finetuning_args.ref_model_quantization_bit,
        )
        ref_finetuning_args = FinetuningArguments()
        tokenizer = load_tokenizer(ref_model_args)["tokenizer"]  # type: ignore
        ref_model = load_model(
            tokenizer,
            ref_model_args,
            ref_finetuning_args,
            is_trainable=False,
            add_valuehead=add_valuehead,
        )
        logger.info(f"Created reference model from {finetuning_args.ref_model}")
    else:
        if finetuning_args.finetuning_type == "lora":
            ref_model = None
        else:
            ref_model_args = ModelArguments.copyfrom(model_args)
            ref_finetuning_args = FinetuningArguments()
            tokenizer = load_tokenizer(ref_model_args)["tokenizer"]  # type: ignore
            ref_model = load_model(
                tokenizer,
                ref_model_args,
                ref_finetuning_args,
                is_trainable=False,
                add_valuehead=add_valuehead,
            )
            logger.info("Created reference model from the model itself.")

    return ref_model


def create_reward_model(
    model: AutoModelForCausalLMWithValueHead,
    model_args: ModelArguments,
    finetuning_args: FinetuningArguments,
) -> AutoModelForCausalLMWithValueHead | None:
    """Create reward model for PPO training.

    Args:
        model: Model for PPO training.
        model_args: Model arguments.
        finetuning_args: Fine-tuning arguments.

    Returns:
        Reward model or None.
    """
    if finetuning_args.reward_model_type == "api":
        assert finetuning_args.reward_model.startswith("http"), "Please provide full url."  # type: ignore
        logger.info("Use reward server %s", finetuning_args.reward_model)
        return finetuning_args.reward_model
    elif finetuning_args.reward_model_type == "lora":
        model.pretrained_model.load_adapter(finetuning_args.reward_model, "reward")
        for (
            name,
            param,
        ) in model.named_parameters():  # https://github.com/huggingface/peft/issues/1090
            if "default" in name:
                param.data = param.data.to(torch.float32)  # trainable params should in fp32
            elif "reward" in name and ("lora_A" in name or "lora_B" in name):
                param.data = param.data.to(torch.float32)
        vhead_params = load_valuehead_params(finetuning_args.reward_model, model_args)
        assert vhead_params is not None, "Reward model is not correctly loaded."
        model.register_buffer(
            "reward_head_weight", vhead_params["v_head.summary.weight"], persistent=False
        )
        model.register_buffer(
            "reward_head_bias", vhead_params["v_head.summary.bias"], persistent=False
        )
        model.register_buffer(
            "default_head_weight",
            torch.zeros_like(vhead_params["v_head.summary.weight"]),
            persistent=False,
        )
        model.register_buffer(
            "default_head_bias",
            torch.zeros_like(vhead_params["v_head.summary.bias"]),
            persistent=False,
        )
        logger.info(f"Loaded adapter weights of reward model from {finetuning_args.reward_model}")
        return None
    else:
        reward_model_args = ModelArguments.copyfrom(
            model_args,
            model_name_or_path=finetuning_args.reward_model,
            adapter_name_or_path=finetuning_args.reward_model_adapters,
            quantization_bit=finetuning_args.reward_model_quantization_bit,
        )
        reward_finetuning_args = FinetuningArguments()
        tokenizer = load_tokenizer(reward_model_args)["tokenizer"]  # type: ignore
        reward_model = load_model(
            tokenizer,
            reward_model_args,
            reward_finetuning_args,
            is_trainable=False,
            add_valuehead=True,
        )
        logger.info(f"Loaded full weights of reward model from {finetuning_args.reward_model}")
        logger.warning(
            "Please ensure the ppo model and reward model share SAME tokenizer and vocabulary."
        )
        return reward_model
