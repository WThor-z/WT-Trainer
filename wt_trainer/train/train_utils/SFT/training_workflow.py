"""Supervised Fine-Tuning (SFT) training workflow module.

This module defines the supervised fine-tuning (SFT) training workflow,
including tokenizer loading, template processing, dataset loading, model loading,
and other core training logic. It orchestrates the entire training process by
integrating various components and utilities.

The main entry point is the `run` function which executes the complete SFT training pipeline.
"""

from types import MethodType

import torch
from transformers import TrainerCallback

from wt_trainer.args import DataArguments
from wt_trainer.args import FinetuningArguments
from wt_trainer.args import GeneratingArguments
from wt_trainer.args import ModelArguments
from wt_trainer.args import TrainingArguments
from wt_trainer.train.trainer import CustomSeq2SeqTrainer
from wt_trainer.train.trainer.trainer_utils import load_tokenizer
from wt_trainer.train.trainer.trainer_utils import sft_train
from wt_trainer.utils.data import get_dataset
from wt_trainer.utils.data import get_template_and_fix_tokenizer
from wt_trainer.utils.model_utils import init_adapter
from wt_trainer.utils.model_utils import load_model
from wt_trainer.utils.model_utils import OptimizedSFTDataCollator
from wt_trainer.utils.model_utils import patch_model
from wt_trainer.utils.model_utils import register_autoclass

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
    """Execute the main SFT training logic.

    This function executes the SFT training process in the following steps:
    1. Load tokenizer using [load_tokenizer][wt_trainer.train.trainer.trainer_utils.load_tokenizer]
    2. Get and fix template using [get_template_and_fix_tokenizer][wt_trainer.utils.data.get_template_and_fix_tokenizer]
    3. Get dataset using [get_dataset][wt_trainer.utils.data.get_dataset]
    4. Load model using [load_model][wt_trainer.utils.model_utils.load_model]
    5. Patch model using [patch_model][wt_trainer.utils.model_utils.patch_model]
    6. Register autoclass using [register_autoclass][wt_trainer.utils.model_utils.register_autoclass]
    7. Initialize adapter using [init_adapter][wt_trainer.utils.model_utils.init_adapter]
    8. Create data collator using [OptimizedSFTDataCollator][wt_trainer.utils.model_utils.OptimizedSFTDataCollator]
    9. Prepare generation kwargs from [GeneratingArguments][wt_trainer.args.GeneratingArguments]
    10. Initialize trainer using [CustomSeq2SeqTrainer][wt_trainer.train.trainer.CustomSeq2SeqTrainer]
    11. Train the model if [TrainingArguments.do_train][wt_trainer.args.TrainingArguments.do_train] is True

    Args:
        arguments: A tuple containing model, data, training, fine-tuning, and generation arguments
            - model_args ([ModelArguments][wt_trainer.args.ModelArguments]): Model configuration arguments
            - data_args ([DataArguments][wt_trainer.args.DataArguments]): Data processing arguments
            - train_args ([TrainingArguments][wt_trainer.args.TrainingArguments]): Training configuration arguments
            - ft_args ([FinetuningArguments][wt_trainer.args.FinetuningArguments]): Fine-tuning specific arguments
            - gen_args ([GeneratingArguments][wt_trainer.args.GeneratingArguments]): Text generation arguments
        callbacks: Optional list of training callback functions, defaults to None

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

    # step4 : load model
    model = load_model(model_args, ft_args, train_args.do_train)

    # step5 : patch model
    patch_model(model, model_args, train_args.do_train)
    register_autoclass(model.config, model, tokenizer)

    # step6 : init adapter
    model = init_adapter(model, model_args, ft_args, train_args.do_train)

    # step7 : datacollator
    model.train()
    data_collator = OptimizedSFTDataCollator(
        tokenizer=tokenizer,
        model=model if not train_args.predict_with_generate else None,
        padding=True,
        pad_to_multiple_of=(8 if train_args.do_train else None),
        block_diag_attn=model_args.block_diag_attn,
        attn_implementation=getattr(model.config, "_attn_implementation", None),
        compute_dtype=model_args.compute_dtype,
    )

    # step8 : preprocess for model generate
    gen_kwargs = gen_args.to_dict(obey_generation_config=True)
    gen_kwargs["eos_token_id"] = [tokenizer.eos_token_id] + tokenizer.additional_special_tokens_ids
    gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    # step9 : init trainer
    sft_trainer = CustomSeq2SeqTrainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        finetuning_args=ft_args,
        processing_class=tokenizer,
        callbacks=callbacks,
        gen_kwargs=gen_kwargs,
        **dataset_module,
    )

    # step10 : train
    if train_args.do_train:
        sft_trainer.sft_train = MethodType(sft_train, sft_trainer)
        train_output = sft_trainer.sft_train(
            args=train_args, train_device=torch.device(model_args.device)
        )

    pass
