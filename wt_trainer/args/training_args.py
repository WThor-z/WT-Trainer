"""Training arguments for configuring the training process.

This module provides a dataclass for configuring training-related arguments,
extending the Seq2SeqTrainingArguments from the transformers' library.
"""

from dataclasses import dataclass
from dataclasses import field

from transformers import Seq2SeqTrainingArguments


@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    r"""Arguments pertaining to the trainer.

    Training modes:

    1. Single-process multi-GPU training (default): The system will use single-process
       multi-GPU training (DataParallel) by directly running the command.
       Use case: 1. Want to quickly start training 2. Don't care much about training performance

    2. Distributed training: Configure 'distributed_training_mode: true' in yaml,
       and use CUDA_VISIBLE_DEVICES=0,1,... to configure the GPUs used (placed before the command)
       Use case: 1. High-performance training required 2. Large-scale model training

    3. Specify GPU: Determine which GPU to run by specifying CUDA_VISIBLE_DEVICES
    """

    distributed_training_mode: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use distributed training mode in multi-GPU environments. "
                "If set to False, single GPU or single-process multi-GPU training will be used. "
                "If set to True, distributed training will be used which requires launching with torchrun or similar."
            )
        },
    )

    def __post_init__(self) -> None:
        Seq2SeqTrainingArguments.__post_init__(self)  # type: ignore
