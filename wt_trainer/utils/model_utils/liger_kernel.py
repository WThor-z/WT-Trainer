"""Liger kernel utilities for model optimization."""

import inspect
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig  # noqa: F401

    from wt_trainer.args import ModelArguments  # noqa: F401


logger = logging.getLogger(__name__)


def apply_liger_kernel(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    is_trainable: bool,
    require_logits: bool,
) -> None:
    """Apply liger kernel to the model for optimization.

    Args:
        config: The model configuration.
        model_args: The model arguments containing liger kernel settings.
        is_trainable: Whether the model is in training mode.
        require_logits: Whether logits are required.
    """
    if not is_trainable or not model_args.enable_liger_kernel:
        return

    model_type = getattr(config, "model_type", None)
    if model_type == "qwen3":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3 as apply_liger_kernel
    elif model_type == "qwen3_moe":
        from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe as apply_liger_kernel
    else:
        logger.warning("Current model does not support liger kernel.")
        return

    if (
        require_logits
        and "fused_linear_cross_entropy" in inspect.signature(apply_liger_kernel).parameters
    ):
        logger.info("Current training stage does not support chunked cross entropy.")
        kwargs = {"fused_linear_cross_entropy": False, "cross_entropy": True}
    else:
        kwargs = {}

    apply_liger_kernel(**kwargs)
    logger.info("Liger kernel has been applied to the model.")
