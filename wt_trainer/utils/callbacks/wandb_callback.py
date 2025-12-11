"""Wandb callback for training monitoring."""

import logging
from typing import Any

from transformers import TrainerCallback
from transformers import TrainerControl
from transformers import TrainerState
from transformers import TrainingArguments

logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    WANDB_AVAILABLE = False
    logger.warning("Wandb is not installed. Please install it with 'pip install wandb'")


class WandbCallback(TrainerCallback):
    """Custom Wandb callback for logging training metrics."""

    def __init__(self, project_name: str = "wt-trainer", run_name: str | None = None) -> None:
        """Initialize Wandb callback.

        Args:
            project_name: Name of the wandb project.
            run_name: Name of the wandb run.
        """
        if not WANDB_AVAILABLE:
            raise ImportError("Please install wandb to use WandbCallback")
            
        self.project_name = project_name
        self.run_name = run_name
        self.initialized = False

    def setup(self, args: TrainingArguments, state: TrainerState, **kwargs: Any) -> None:
        """Setup wandb at the beginning of training.

        Args:
            args: Training arguments.
            state: Trainer state.
            **kwargs: Additional keyword arguments.
        """
        if not self.initialized:
            # Initialize wandb run
            wandb.init(
                project=self.project_name,
                name=self.run_name,
                config=args.to_dict()
            )
            self.initialized = True

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        """Event called at the beginning of training.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            **kwargs: Additional keyword arguments.
        """
        self.setup(args, state, **kwargs)

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: dict[str, float] | None = None, **kwargs: Any) -> None:
        """Event called after logging the last logs.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            logs: Metrics to log.
            **kwargs: Additional keyword arguments.
        """
        if not self.initialized:
            self.setup(args, state, **kwargs)

        if logs is not None:
            # Log metrics to wandb
            wandb.log(logs, step=state.global_step)

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs: Any) -> None:
        """Event called at the end of training.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
            **kwargs: Additional keyword arguments.
        """
        if self.initialized:
            # Finish wandb run
            wandb.finish()