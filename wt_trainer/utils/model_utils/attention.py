"""Attention implementation configuration utilities."""

import logging
from typing import TYPE_CHECKING

from transformers.utils import is_flash_attn_2_available
from transformers.utils import is_torch_sdpa_available

from wt_trainer.utils.const import AttentionFunction

if TYPE_CHECKING:
    from transformers import PretrainedConfig  # noqa: F401

    from wt_trainer.args import ModelArguments  # noqa: F401


logger = logging.getLogger(__name__)


def configure_attn_implementation(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    """Configure attention implementation for the given model.

    Args:
        config: The model configuration.
        model_args: The model arguments containing attention settings.
    """
    if model_args.flash_attn == AttentionFunction.AUTO:
        return

    elif model_args.flash_attn == AttentionFunction.DISABLED:
        requested_attn_implementation = "eager"

    elif model_args.flash_attn == AttentionFunction.SDPA:
        if not is_torch_sdpa_available():
            logger.warning("torch>=2.1.1 is required for SDPA attention.")
            return

        requested_attn_implementation = "sdpa"
    elif model_args.flash_attn == AttentionFunction.FA2:
        if not is_flash_attn_2_available():
            logger.warning("FlashAttention-2 is not installed.")
            return

        requested_attn_implementation = "flash_attention_2"
    else:
        raise NotImplementedError(f"Unknown attention type: {model_args.flash_attn}")

    setattr(config, "_attn_implementation", requested_attn_implementation)


def print_attn_implementation(config: "PretrainedConfig") -> None:
    """Print information about the attention implementation being used.

    Args:
        config: The model configuration.
    """
    attn_implementation = getattr(config, "_attn_implementation", None)

    if attn_implementation == "flash_attention_2":
        logger.info("Using FlashAttention-2 for faster training and inference.")
    elif attn_implementation == "sdpa":
        logger.info("Using torch SDPA for faster training and inference.")
    else:
        logger.info("Using vanilla attention implementation.")
