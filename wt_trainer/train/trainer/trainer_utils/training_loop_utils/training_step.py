"""Training step implementation for supervised fine-tuning.

This module contains the training step function that performs forward and backward
passes for a single training sample, including loss computation and gradient calculation.
"""

from typing import Any

import torch
from torch import nn

from .forward import compute_loss


def training_step(
    self, 
    model: nn.Module, 
    sample: dict[str, torch.Tensor | Any]
) -> torch.Tensor:
    """Execute a single training step for the given model and sample.

    This function performs the forward pass, computes the loss, and executes the 
    backward pass to calculate gradients.

    Args:
        self: Trainer instance containing optimizer and other training components.
        model: The neural network model to train.
        sample: A dictionary containing the input data for this training step.

    Returns:
        torch.Tensor: The computed loss for this training step, detached from the 
        computation graph.
    """
    model.train()
    if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
        self.optimizer.train()

    inputs = self._prepare_inputs(sample)

    # Forward pass
    loss = compute_loss(model, inputs)

    del inputs

    torch.cuda.empty_cache()

    loss = loss / self.current_gradient_accumulation_steps

    # Backward pass
    self.accelerator.backward(loss)

    return loss.detach()