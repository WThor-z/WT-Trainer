"""RoPE (Rotary Position Embedding) scaling configuration utilities.

This module provides functions to configure RoPE scaling for transformer models.
RoPE scaling allows models to extrapolate to sequence lengths longer than
they were trained on.
"""

import logging
import math
from typing import TYPE_CHECKING

from wt_trainer.utils.const import RopeScaling

if TYPE_CHECKING:
    from transformers import PretrainedConfig  # noqa: F401

    from wt_trainer.args import ModelArguments  # noqa: F401


logger = logging.getLogger(__name__)


def configure_rope(config: "PretrainedConfig", model_args: "ModelArguments") -> None:
    """Configure RoPE scaling for the model.

    This function sets up RoPE scaling based on the model arguments and current
    model configuration. It supports different scaling strategies like linear,
    dynamic, YARN, and LLAMA3.

    Args:
        config: The model configuration to be modified.
        model_args: Model arguments containing RoPE scaling configuration.
    """
    if model_args.rope_scaling is None:
        return

    if not hasattr(config, "rope_scaling"):
        logger.warning("Current model does not support RoPE scaling.")
        return

    if hasattr(config, "max_position_embeddings"):
        old_max_length = getattr(config, "max_position_embeddings", None)
    else:
        logger.warning("Cannot find the max position embeddings in the config.")
        return

    if model_args.model_max_length is not None:  # training
        if model_args.model_max_length <= old_max_length:
            logger.warning("Input length is smaller than max length. Disabling rope scaling.")
            return

        if model_args.rope_scaling == RopeScaling.DYNAMIC:
            logger.warning(
                "Dynamic NTK scaling may not work well with fine-tuning. "
                "See: https://github.com/huggingface/transformers/pull/24653"
            )

        rope_factor = float(math.ceil(model_args.model_max_length / old_max_length))
    else:  # inference
        rope_factor = 2.0

    rope_kwargs = {
        "rope_type": getattr(
            model_args.rope_scaling, "value", model_args.rope_scaling
        ),  # handle enum
        "factor": rope_factor,
    }
    setattr(config, "max_position_embeddings", old_max_length * rope_factor)
    logger.info(
        f"Enlarge max model length from {old_max_length} to {old_max_length * rope_factor}."
    )

    if model_args.rope_scaling in [RopeScaling.DYNAMIC, RopeScaling.YARN]:
        rope_kwargs["original_max_position_embeddings"] = old_max_length
    elif model_args.rope_scaling == RopeScaling.LLAMA3:
        rope_kwargs["original_max_position_embeddings"] = old_max_length
        rope_kwargs["low_freq_factor"] = 1.0
        rope_kwargs["high_freq_factor"] = 4.0

    setattr(config, "rope_scaling", rope_kwargs)
    logger.info(
        f"Using {rope_kwargs['rope_type']} scaling strategy and setting scaling factor to {rope_kwargs['factor']}."
    )
