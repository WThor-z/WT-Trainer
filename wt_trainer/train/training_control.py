"""Training control module for managing the training workflow.

This module provides functionality to control and execute the training process
by coordinating different training utilities.
"""

import logging
from typing import TYPE_CHECKING

from wt_trainer.utils.train import get_custom_callbacks
from wt_trainer.utils.train import read_train_args
from wt_trainer.utils.train import train_method_select

if TYPE_CHECKING:
    from transformers import TrainerCallback  # noqa: F401

logger = logging.getLogger(__name__)


def run_train(config_file_path: str | None = None) -> None:
    """Run the training process.

    This function orchestrates the training workflow by reading training arguments,
    getting custom callbacks, and selecting the appropriate training method.

    Args:
        config_file_path: Path to the configuration file. If None, default configuration will be used.
    """
    train_config = read_train_args(config_file_path)
    trainer_callbacks: list["TrainerCallback"] = get_custom_callbacks(train_config)
    train_method_select(train_config, trainer_callbacks)


if __name__ == "__main__":
    run_train()
