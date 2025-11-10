"""Custom learning rate scheduler implementations for WT Trainer.

This module provides custom learning rate scheduler implementations.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wt_trainer.args import TrainingArguments  # noqa: F401


def create_custom_scheduler(
    training_args: "TrainingArguments",
    num_training_steps: int,
) -> None:
    """Create a custom scheduler based on the provided arguments.

    Args:
        training_args: Training arguments.
        num_training_steps: Number of training steps.
        optimizer: Optimizer object or None.
    """
    if training_args.lr_scheduler_type == "warmup_stable_decay":
        num_warmup_steps = training_args.get_warmup_steps(num_training_steps)
        remaining_steps = num_training_steps - num_warmup_steps
        num_stable_steps = remaining_steps // 3  # use 1/3 for stable by default
        num_decay_steps = remaining_steps - num_stable_steps
        scheduler_kwargs = training_args.lr_scheduler_kwargs or {}
        default_kwargs = {
            "num_stable_steps": num_stable_steps,
            "num_decay_steps": num_decay_steps,
        }
        for key, value in default_kwargs.items():
            if key not in scheduler_kwargs:
                scheduler_kwargs[key] = value  # type: ignore

        training_args.lr_scheduler_kwargs = scheduler_kwargs
