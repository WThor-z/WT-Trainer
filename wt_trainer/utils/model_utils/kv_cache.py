"""KV cache configuration utilities."""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig  # noqa: F401

    from wt_trainer.args import ModelArguments  # noqa: F401

logger = logging.getLogger(__name__)


def configure_kv_cache(
    config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool
) -> None:
    """Configure KV cache for the model.

    Args:
        config: The model configuration.
        model_args: The model arguments containing cache settings.
        is_trainable: Whether the model is in training mode.
    """
    if not is_trainable:
        setattr(config, "use_cache", model_args.use_cache)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "use_cache", model_args.use_cache)

        if model_args.use_cache:
            logger.info("KV cache is enabled for faster generation.")
        else:
            logger.info("KV cache is disabled.")
    else:
        setattr(config, "use_cache", False)
        if hasattr(config, "text_config"):
            setattr(config.text_config, "use_cache", False)

        logger.info("KV cache is disabled during training.")
