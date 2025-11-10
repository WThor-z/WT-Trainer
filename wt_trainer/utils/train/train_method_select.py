"""Module for selecting and executing the appropriate training method based on configuration.

This module provides functionality to select the training method based on the
configuration parameters and execute the corresponding training process.
"""

import logging
from typing import Any, TYPE_CHECKING

from .args_process import get_train_args

if TYPE_CHECKING:
    from transformers import TrainerCallback  # noqa: F401

logger = logging.getLogger(__name__)


def train_method_select(
    train_config: dict[str, Any], trainer_callbacks: list["TrainerCallback"] | None = None
) -> None:
    """Select and execute the appropriate training method based on configuration.

    Args:
        train_config: Training configuration parameters.
        trainer_callbacks: List of trainer callbacks.

    Raises:
        Exception: If training execution fails.
    """
    try:
        model_args, data_args, train_args, ft_args, gen_args = get_train_args(train_config)

        if ft_args.stage == "sft":
            from wt_trainer.train.train_utils.SFT import run_sft

            run_sft((model_args, data_args, train_args, ft_args, gen_args), trainer_callbacks)

    except Exception as e:
        logger.error(f"Training execution failed: {str(e)}")
        raise
