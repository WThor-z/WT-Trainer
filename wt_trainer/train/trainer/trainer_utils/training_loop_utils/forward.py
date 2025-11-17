"""Forward pass and loss computation utilities.

This module contains functions for performing forward passes through a model
and computing the loss for supervised fine-tuning tasks.
"""

from typing import Any


def compute_loss(
    model: Any, 
    inputs: dict[str, Any], 
    return_outputs: bool = False
) -> Any:
    """Compute the loss for a given model and inputs.

    Performs a forward pass through the model and extracts the loss from the outputs.
    Can optionally return both the loss and the full outputs.

    Args:
        model: The model to use for the forward pass.
        inputs: A dictionary of input tensors for the model.
        return_outputs: Whether to return the full outputs in addition to the loss.

    Returns:
        If return_outputs is True, returns a tuple of (loss, outputs).
        Otherwise, returns just the loss.
    """
    outputs = model(**inputs)

    loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    return (loss, outputs) if return_outputs else loss