"""SFT training workflow module.

This module defines the supervised fine-tuning (SFT) training workflow, 
including data loading, model initialization, and other core training logic.
"""

from transformers import TrainerCallback

from wt_trainer.args import DataArguments
from wt_trainer.args import FinetuningArguments
from wt_trainer.args import GeneratingArguments
from wt_trainer.args import ModelArguments
from wt_trainer.args import TrainingArguments
from wt_trainer.train.trainer.trainer_utils import load_tokenizer
from wt_trainer.utils.data import get_dataset
from wt_trainer.utils.data import get_template_and_fix_tokenizer

_TRAIN_CLS = tuple[
    ModelArguments,
    DataArguments,
    TrainingArguments,
    FinetuningArguments,
    GeneratingArguments,
]


def run(
    arguments: _TRAIN_CLS,
    callbacks: list[TrainerCallback] | None = None,
) -> None:
    """Execute the main training logic.

    This function executes the training process in the following steps:
    1. Load tokenizer
    2. Get template
    3. Get dataset

    Args:
        arguments: A tuple containing model, data, training, fine-tuning, and generation arguments
        callbacks: List of training callback functions, default is None

    Returns:
        None
    """
    (
        model_args,
        data_args,
        train_args,
        ft_args,
        gen_args,
    ) = arguments  # Order cannot be changed, this is not a dictionary

    # step1 : load tokenizer
    tokenizer = load_tokenizer(model_args)

    # step2 : get template
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    # step3 : get dataset
    dataset_module = get_dataset(template, model_args, data_args, train_args, "sft", tokenizer)

    pass