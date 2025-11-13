"""MoE (Mixture of Experts) model configuration utilities.

This module provides functions to configure Mixture of Experts models
with appropriate settings for training and inference.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn  # noqa: F401
    from transformers import PretrainedConfig  # noqa: F401

    from wt_trainer.args import ModelArguments  # noqa: F401


def configure_moe(
    config: "PretrainedConfig", model_args: "ModelArguments", is_trainable: bool
) -> None:
    """Configure the model for MoE (Mixture of Experts) training.

    This function sets up the appropriate MoE parameters based on the model type
    and training configuration. It configures router logits output and auxiliary
    loss coefficients for different MoE model types.

    Args:
        config: The model configuration to be modified.
        model_args: Model arguments containing MoE configuration options.
        is_trainable: Whether the model is being configured for training.
    """
    if not is_trainable or not model_args.moe_aux_loss_coef:
        return

    model_type = getattr(config, "model_type", None)
    if model_type in [
        "dbrx",
        "granitemoe",
        "jamba",
        "jetmoe",
        "llama4",
        "mixtral",
        "olmoe",
        "phimoe",
        "qwen2_moe",
        "qwen3_moe",
    ]:
        setattr(config, "output_router_logits", True)

    if model_type in [
        "granitemoe",
        "jamba",
        "llama4",
        "mixtral",
        "olmoe",
        "phimoe",
        "qwen2_moe",
        "qwen3_moe",
    ]:
        setattr(config, "router_aux_loss_coef", model_args.moe_aux_loss_coef)

    elif model_type == "deepseek":
        setattr(config, "aux_loss_alpha", model_args.moe_aux_loss_coef)

    elif model_type == "jetmoe":
        setattr(config, "aux_loss_coef", model_args.moe_aux_loss_coef)
