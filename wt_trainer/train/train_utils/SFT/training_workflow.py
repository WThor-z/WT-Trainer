from typing import TYPE_CHECKING

from wt_trainer.args import *
from wt_trainer.train.trainer.trainer_utils import load_tokenizer
from wt_trainer.utils.data.template import get_template_and_fix_tokenizer

if TYPE_CHECKING:

    from transformers import TrainerCallback  # noqa: F401

_TRAIN_CLS = tuple[
    ModelArguments,
    DataArguments,
    TrainingArguments,
    FinetuningArguments,
    GeneratingArguments,
]

def run(
    arguments: "_TRAIN_CLS",
    callbacks: list["TrainerCallback"] | None = None,
):
    """
    主要训练逻辑
    """
    (
        model_args,
        data_args,
        train_args,
        ft_args,
        gen_args,
    ) = arguments  # 顺序不能改动，这不是字典

    # step1 : load tokenizer
    tokenizer = load_tokenizer(model_args)

    # step2 : get template
    template = get_template_and_fix_tokenizer(tokenizer, data_args)

    pass
