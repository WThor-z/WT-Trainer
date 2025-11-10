"""Module for retrieving and configuring custom training callbacks.

This module provides functionality to get and configure custom callbacks
for the training process based on the training configuration.
"""

from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import TrainerCallback  # noqa: F401


def get_custom_callbacks(train_config: Dict) -> List["TrainerCallback"] | None:
    """Get custom callbacks based on training configuration.

    Args:
        train_config: Dictionary containing training configuration parameters.

    Returns:
        List of custom callbacks or None if no custom callbacks are configured.
    """

    callbacks: List["TrainerCallback"] = []

    return callbacks if callbacks else None
