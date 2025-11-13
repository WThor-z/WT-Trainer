"""Model patching utilities for configuring and modifying model behavior.

This module provides functions to patch model configurations and models themselves
with various enhancements such as attention implementations, RoPE scaling,
MoE configuration, and more.
"""

import logging
from typing import TYPE_CHECKING

import torch

from .attention import configure_attn_implementation
from .attention import print_attn_implementation
from .checkpointing import prepare_model_for_training
from .kv_cache import configure_kv_cache
from .moe import configure_moe
from .packing import configure_packing
from .rope import configure_rope

if TYPE_CHECKING:
    from transformers import PretrainedConfig  # noqa: F401
    from transformers import PreTrainedModel  # noqa: F401
    from transformers import PreTrainedTokenizer  # noqa: F401
    from trl import AutoModelForCausalLMWithValueHead  # noqa: F401

    from wt_trainer.args import ModelArguments  # noqa: F401


logger = logging.getLogger(__name__)


def patch_config(
    config: PretrainedConfig,
    model_args: "ModelArguments",
    is_trainable: bool,
) -> None:
    """Patch the model configuration with various enhancements.

    This function configures the model with:
    - Attention implementation
    - RoPE scaling
    - MoE settings
    - Packing configuration
    - KV cache settings
    - Qwen-specific settings if applicable

    Args:
        config: The model configuration to patch.
        model_args: Model arguments containing configuration options.
        is_trainable: Whether the model is trainable.
    """
    configure_attn_implementation(config, model_args)
    configure_rope(config, model_args)
    configure_moe(config, model_args, is_trainable)
    configure_packing(model_args, is_trainable)
    configure_kv_cache(config, model_args, is_trainable)

    if getattr(config, "model_type", None) == "qwen":
        setattr(config, "use_flash_attn", model_args.flash_attn == "fa2")
        for dtype_name, dtype in [
            ("fp16", torch.float16),
            ("bf16", torch.bfloat16),
            ("fp32", torch.float32),
        ]:
            setattr(config, dtype_name, model_args.compute_dtype == dtype)


def patch_model(
    model: PreTrainedModel,
    model_args: "ModelArguments",
    is_trainable: bool,
) -> None:
    """Patch the model with various enhancements.

    This function modifies the model with:
    - Generation configuration fixes
    - Training preparation if trainable
    - Attention implementation printing
    - Model tagging

    Args:
        model: The model to patch.
        model_args: Model arguments containing configuration options.
        is_trainable: Whether the model is trainable.
    """
    gen_config = model.generation_config  # check and fix generation config
    if not gen_config.do_sample and (
        (gen_config.temperature is not None and gen_config.temperature != 1.0)
        or (gen_config.top_p is not None and gen_config.top_p != 1.0)
        or (gen_config.typical_p is not None and gen_config.typical_p != 1.0)
    ):
        gen_config.do_sample = True

    if is_trainable:
        prepare_model_for_training(model, model_args)

    if not model_args.use_unsloth:
        print_attn_implementation(model.config)


    model.add_model_tags(["WT-Trainer"])


def register_autoclass(
    config: "PretrainedConfig",
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizer"
) -> None:
    """Register the model components for auto-class loading.

    This function registers the config, model, and tokenizer for automatic
    class loading by the transformers library.

    Args:
        config: The model configuration.
        model: The model instance.
        tokenizer: The tokenizer instance.
    """
    if "AutoConfig" in getattr(config, "auto_map", {}):
        config.__class__.register_for_auto_class()
    if "AutoModelForCausalLM" in getattr(config, "auto_map", {}):
        model.__class__.register_for_auto_class()
    if "AutoTokenizer" in tokenizer.init_kwargs.get("auto_map", {}):
        tokenizer.__class__.register_for_auto_class()
