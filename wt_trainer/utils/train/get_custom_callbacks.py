"""Module for retrieving and configuring custom training callbacks.

This module provides functionality to get and configure custom callbacks
for the training process based on the training configuration.
"""

import logging
from typing import TYPE_CHECKING, List

from transformers import PrinterCallback

from wt_trainer.args import TrainingArguments

if TYPE_CHECKING:
    from transformers import TrainerCallback  # noqa: F401

logger = logging.getLogger(__name__)


def get_custom_callbacks(training_args: TrainingArguments) -> List["TrainerCallback"] | None:
    """Get custom callbacks based on training configuration.

    Args:
        train_config: Dictionary containing training configuration parameters.

    Returns:
        List of custom callbacks or None if no custom callbacks are configured.
    """

    callbacks: list["TrainerCallback"] = []

    # Remove default callbacks to avoid duplication
    callbacks.append(PrinterCallback())

    # Add wandb callback if configured
    if training_args.report_to and "wandb" in training_args.report_to:

        from wt_trainer.utils.callbacks import WandbCallback

        # Get project name and run name from config if available
        project_name = getattr(training_args, "wandb_project", "WT-Trainer")
        run_name = getattr(training_args, "run_name", "Test")

        callbacks.append(WandbCallback(project_name=project_name, run_name=run_name))
        logger.info("Added wandb callback")

    return callbacks if callbacks else None
