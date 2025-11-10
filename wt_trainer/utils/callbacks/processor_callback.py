"""Callback for saving processor during training."""

import os
from typing import TYPE_CHECKING

from typing_extensions import override
from transformers import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (  # noqa: F401
    TrainingArguments,
    TrainerState,
    TrainerControl,
    PreTrainedTokenizer,
)

if TYPE_CHECKING:
    from wt_trainer.args import ModelArguments, FinetuningArguments  # noqa: F401


class SaveProcessorCallback(TrainerCallback):
    """A callback for saving the processor."""

    def __init__(self, tokenizer: "PreTrainedTokenizer") -> None:
        """Initialize the callback.

        Args:
            tokenizer: The tokenizer to save.
        """
        self.tokenizer = tokenizer

    @override
    def on_save(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ) -> None:
        """Save tokenizer on checkpoint.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
        """
        if args.should_save:
            output_dir = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )
            self.tokenizer.save_pretrained(output_dir)

    @override
    def on_train_end(
        self, args: "TrainingArguments", state: "TrainerState", control: "TrainerControl", **kwargs
    ) -> None:
        """Save tokenizer at the end of training.

        Args:
            args: Training arguments.
            state: Trainer state.
            control: Trainer control.
        """
        if args.should_save:
            self.tokenizer.save_pretrained(args.output_dir)
