"""Utility functions for computing loss during training."""

from typing import Any

import torch
from torch import nn
from transformers import Trainer


def custom_compute_loss(
    trainer: Trainer,
    model: nn.Module,
    inputs: dict[str, torch.Tensor | Any],
    return_outputs: bool = False,
    num_items_in_batch: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]] | None:
    """Compute the loss for the given inputs.

    This function needs to be implemented by the user according to their specific use case.
    (Note that it is necessary to implement forward instead of just the loss function.)
    The default implementation returns None.

    Args:
        trainer: The trainer instance.
        model: The model being trained.
        inputs: The inputs to the model.
        return_outputs: Whether to return the outputs along with the loss.
        num_items_in_batch: The number of items in the batch.

    Returns:
        The computed loss, or a tuple of loss and outputs if return_outputs is True,
        or None if not implemented.
    """
    return None
