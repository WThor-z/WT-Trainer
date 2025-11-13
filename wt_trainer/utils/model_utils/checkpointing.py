"""Gradient checkpointing utilities for model training."""

from functools import partial
from functools import WRAPPER_ASSIGNMENTS
from functools import wraps
import logging
from types import MethodType
from typing import Any, Callable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel  # noqa: F401

    from wt_trainer.args import ModelArguments  # noqa: F401


logger = logging.getLogger(__name__)

LAYERNORM_NAMES = {"norm", "ln"}


def get_custom_gradient_checkpointing_func(gradient_checkpointing_func: Callable) -> Callable:
    """Only applies gradient checkpointing to trainable layers.

    Args:
        gradient_checkpointing_func: The original gradient checkpointing function.

    Returns:
        A wrapped gradient checkpointing function that only applies to trainable layers.
    """

    @wraps(gradient_checkpointing_func, assigned=WRAPPER_ASSIGNMENTS + ("__self__",))
    def custom_gradient_checkpointing_func(func: Callable, *args: torch.Tensor | Any, **kwargs):
        if isinstance(func, partial):
            module: torch.nn.Module = func.func.__self__
        else:
            module: torch.nn.Module = func.__self__

        has_grad = False
        if any(param.requires_grad for param in module.parameters()):
            has_grad = True
            for arg in args:
                if torch.is_tensor(arg) and torch.is_floating_point(arg):
                    arg.requires_grad_(True)
                    break  # assume the first tensor is always the hidden states

        if has_grad:
            return gradient_checkpointing_func(func, *args, **kwargs)

        return func(*args, **kwargs)

    return custom_gradient_checkpointing_func


def _gradient_checkpointing_enable(
    self: "PreTrainedModel",
    gradient_checkpointing_kwargs: dict[str, Any] | None = None,
) -> None:
    """Activates gradient checkpointing for the current model.

    Modification of the original method to enable gradient checkpointing for block-wise optimizer.

    Args:
        self: The model instance.
        gradient_checkpointing_kwargs: Keyword arguments for gradient checkpointing.
    """
    from torch.utils.checkpoint import checkpoint

    if not self.supports_gradient_checkpointing:
        raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}

    gradient_checkpointing_func = partial(checkpoint, **gradient_checkpointing_kwargs)

    gradient_checkpointing_func = get_custom_gradient_checkpointing_func(
        gradient_checkpointing_func
    )
    # have already enabled input require gradients
    self._set_gradient_checkpointing(
        enable=True, gradient_checkpointing_func=gradient_checkpointing_func
    )


def prepare_model_for_training(model: "PreTrainedModel", model_args: "ModelArguments") -> None:
    """Prepare the model before training.

    Include:
    (1) cast the layernorm in fp32
    (2) make output embedding layer require grads
    (3) add the upcasting of the lm_head in fp32.

    Args:
        model: The model to prepare for training.
        model_args: Model arguments containing training configuration.
    """
    if model_args.upcast_layernorm:
        logger.info("Upcasting layernorm weights in float32.")
        for name, param in model.named_parameters():
            if param.ndim == 1 and any(ln_name in name for ln_name in LAYERNORM_NAMES):
                param.data = param.data.to(torch.float32)

    if not model_args.disable_gradient_checkpointing:
        if not getattr(model, "supports_gradient_checkpointing", False):
            logger.warning("Current model does not support gradient checkpointing.")
        else:
            # use_reentrant=False might increase VRAM usage (have not been empirically verified yet)
            # According to: https://github.com/huggingface/transformers/issues/28339
            model.gradient_checkpointing_enable = MethodType(_gradient_checkpointing_enable, model)
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": model_args.use_reentrant_gc}
            )
            setattr(
                model.config, "use_cache", False
            )  # turn off when gradient checkpointing is enabled
            logger.info("Gradient checkpointing enabled.")
