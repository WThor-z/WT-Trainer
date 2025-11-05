"""Module for processing training arguments and validating configurations."""

import logging
import os
from typing import Any

import torch
import transformers
from transformers import HfArgumentParser
from transformers.hf_argparser import DataClass
from transformers.integrations import is_deepspeed_zero3_enabled  # type: ignore
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ParallelMode
from transformers.utils import is_torch_bf16_gpu_available  # type: ignore
from transformers.utils import is_torch_cuda_available  # type: ignore

from wt_trainer.args import *
from wt_trainer.utils.const import CHECKPOINT_NAMES
from wt_trainer.utils.const import EngineName

logger = logging.getLogger(__name__)


def _get_current_device() -> "torch.device":
    """Get current device information, prioritizing GPU if available.

    Returns:
        torch.device: The current device.
    """
    if is_torch_cuda_available():  # type: ignore
        device = f"cuda:{os.getenv('LOCAL_RANK', '0')}"
    else:
        device = "cpu"

    return torch.device(device)


def _parse_train_args(train_args: dict[str, Any]) -> tuple[DataClass, ...]:
    """Parse incoming training arguments.

    Args:
        train_args: Dictionary of training arguments.

    Returns:
        Tuple of parsed data classes.
    """
    parser = HfArgumentParser(
        [
            ModelArguments,
            DataArguments,
            TrainingArguments,
            FinetuningArguments,
            GeneratingArguments,
        ]
    )

    return parser.parse_dict(train_args)


def get_train_args(train_args: dict[str, Any]) -> tuple[DataClass, ...]:
    """Get and validate training arguments.

    Args:
        train_args: Dictionary of training arguments.

    Returns:
        Tuple of validated model arguments.
    """
    (
        model_args,
        data_args,
        training_args,
        finetuning_args,
        generating_args,
    ) = _parse_train_args(train_args)

    if finetuning_args.stage != "sft":
        if training_args.predict_with_generate:
            raise ValueError("`predict_with_generate` cannot be set as True except SFT.")

        if data_args.neat_packing:
            raise ValueError("`neat_packing` cannot be set as True except SFT.")

        if data_args.train_on_prompt or data_args.mask_history:
            raise ValueError(
                "`train_on_prompt` or `mask_history` cannot be set as True except SFT."
            )

    # When predict_with_generate=False, the model only predicts the next token rather than
    # generating a complete sequence. During evaluation, the model's generate() method will be
    # used to generate text, rather than just using forward propagation to predict the next token.
    # When predict_with_generate=True, the model generates complete sequences,
    # enabling generative evaluation metrics such as ROUGE and BLEU
    if (
        finetuning_args.stage == "sft"
        and training_args.do_predict
        and not training_args.predict_with_generate
    ):
        raise ValueError("Please enable `predict_with_generate` to save model predictions.")

    if finetuning_args.stage in ["rm", "ppo"] and training_args.load_best_model_at_end:
        raise ValueError("RM and PPO stages do not support `load_best_model_at_end`.")

    if finetuning_args.stage == "ppo":
        if not training_args.do_train:
            raise ValueError(
                "PPO training does not support evaluation, use the SFT stage to evaluate models."
            )

        if model_args.shift_attn:
            raise ValueError("PPO training is incompatible with S^2-Attn.")

        if finetuning_args.reward_model_type == "lora" and model_args.use_unsloth:
            raise ValueError("Unsloth does not support lora reward model.")

        if training_args.report_to and training_args.report_to[0] not in [
            "wandb",
            "tensorboard",
        ]:
            raise ValueError("PPO only accepts wandb or tensorboard logger.")

    # -1 means not setting the maximum steps, but calculating based on dataset size and num_train_epochs
    # In streaming mode (streaming=True), data is infinite, so the maximum training steps must be
    # explicitly specified rather than using automatic calculation based on dataset size
    if training_args.max_steps == -1 and data_args.streaming:
        raise ValueError("Please specify `max_steps` in streaming mode.")

    # Control training
    if training_args.do_train and data_args.dataset is None:
        raise ValueError("Please specify dataset for training.")

    # Control evaluation
    if (training_args.do_eval or training_args.do_predict) and (
        data_args.eval_dataset is None and data_args.val_size < 1e-6
    ):
        raise ValueError("Please specify dataset for evaluation.")

    # Configuration to note when enabling direct sentence generation during inference
    if training_args.predict_with_generate:

        if data_args.eval_dataset is None:
            raise ValueError("Cannot use `predict_with_generate` if `eval_dataset` is None.")

        if finetuning_args.compute_accuracy:
            raise ValueError("Cannot use `predict_with_generate` and `compute_accuracy` together.")

        if training_args.do_train and model_args.quantization_device_map == "auto":
            raise ValueError("Cannot use device map for quantized models in training.")

    if finetuning_args.pure_bf16:
        if not is_torch_bf16_gpu_available():
            raise ValueError("This device does not support `pure_bf16`.")

    # VLLM高速推理不适用在普通的对话中，普通的chat只用HF-BACKEND即可
    if model_args.infer_backend != EngineName.HF:
        raise ValueError("vLLM/SGLang backend is only available for API, CLI.")

    # Unsloth加速单卡，对于多卡没有帮助，且两个优化框架本身应该冲突
    if model_args.use_unsloth and is_deepspeed_zero3_enabled():  # type: ignore
        raise ValueError("Unsloth is incompatible with DeepSpeed ZeRO-3.")

    if model_args.adapter_name_or_path is not None and finetuning_args.finetuning_type != "lora":
        raise ValueError("Adapter is only valid for the LoRA method.")

    if model_args.quantization_bit is not None:
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("Quantization is only compatible with the LoRA method.")

        if finetuning_args.pissa_init:
            raise ValueError(
                "Please use scripts/pissa_init.py to initialize PiSSA for a quantized model."
            )

        if model_args.resize_vocab:
            raise ValueError("Cannot resize embedding layers of a quantized model.")

        if model_args.adapter_name_or_path is not None and finetuning_args.create_new_adapter:
            raise ValueError("Cannot create new adapter upon a quantized model.")

        if (
            model_args.adapter_name_or_path is not None
            and len(model_args.adapter_name_or_path) != 1
        ):
            raise ValueError("Quantized model only accepts a single adapter. Merge them first.")

    if training_args is not None:

        if (
            training_args.do_train
            and finetuning_args.finetuning_type == "lora"
            and model_args.quantization_bit is None
            and model_args.resize_vocab
            and finetuning_args.additional_target is None
        ):
            logger.warning(
                "Remember to add embedding layers to `additional_target` to make the added tokens trainable."
            )

        if (
            training_args.do_train
            and model_args.quantization_bit is not None
            and (not model_args.upcast_layernorm)
        ):
            logger.warning("We recommend enable `upcast_layernorm` in quantized training.")

    if training_args.do_train and (not training_args.fp16) and (not training_args.bf16):
        logger.warning("We recommend enable mixed precision training.")

    if (
        training_args.do_train
        and (finetuning_args.use_galore or finetuning_args.use_apollo)
        and not finetuning_args.pure_bf16
    ):
        logger.warning(
            "Using GaLore or APOLLO with mixed precision training may significantly increases GPU memory usage."
        )

        if (not training_args.do_train) and model_args.quantization_bit is not None:
            logger.warning("Evaluating model in 4/8-bit mode may cause lower scores.")

    if (
        (not training_args.do_train)
        and finetuning_args.stage == "dpo"
        and finetuning_args.ref_model is None
    ):
        logger.warning("Specify `ref_model` for computing rewards at evaluation.")

    # Post-process training arguments
    training_args.generation_max_length = (
        training_args.generation_max_length or data_args.cutoff_len
    )
    training_args.generation_num_beams = (
        data_args.eval_num_beams or training_args.generation_num_beams
    )
    training_args.remove_unused_columns = False  # important for multimodal dataset

    if finetuning_args.finetuning_type == "lora":
        # https://github.com/huggingface/transformers/blob/v4.50.0/src/transformers/trainer.py#L782
        training_args.label_names = training_args.label_names or ["labels"]

    if finetuning_args.stage in ["rm", "ppo"] and finetuning_args.finetuning_type in [
        "full",
        "freeze",
    ]:
        can_resume_from_checkpoint = False
        if training_args.resume_from_checkpoint is not None:
            logger.warning("Cannot resume from checkpoint in current stage.")
            training_args.resume_from_checkpoint = None
    else:
        can_resume_from_checkpoint = True

    if (
        training_args.resume_from_checkpoint is None
        and training_args.do_train
        and os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
        and can_resume_from_checkpoint
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)  # type: ignore
        if last_checkpoint is None and any(
            os.path.isfile(os.path.join(training_args.output_dir, name))
            for name in CHECKPOINT_NAMES
        ):
            raise ValueError(
                "Output directory already exists and is not empty. Please set `overwrite_output_dir`."
            )  # To resume training from a checkpoint, the output folder needs to be overwritable
            # (this situation is when there is no checkpoint folder)

        if last_checkpoint is not None:
            training_args.resume_from_checkpoint = (
                last_checkpoint  # Get the last training checkpoint path
            )
            logger.info(f"Resuming training from {training_args.resume_from_checkpoint}.")
            logger.info("Change `output_dir` or use `overwrite_output_dir` to avoid.")

    if (
        finetuning_args.stage in ["rm", "ppo"]
        and finetuning_args.finetuning_type == "lora"
        and training_args.resume_from_checkpoint is not None
    ):
        logger.warning(
            f"Add {training_args.resume_from_checkpoint} to `adapter_name_or_path` to resume training from checkpoint."
        )

    # Model training precision
    if training_args.bf16 or finetuning_args.pure_bf16:
        model_args.compute_dtype = torch.bfloat16
    elif training_args.fp16:
        model_args.compute_dtype = torch.float16

    model_args.device_map = {"": _get_current_device()}
    model_args.model_max_length = data_args.cutoff_len
    model_args.block_diag_attn = data_args.neat_packing
    data_args.packing = (
        data_args.packing if data_args.packing is not None else finetuning_args.stage == "pt"
    )

    # Log distributed information
    logger.info(
        f"Process ID: {training_args.process_index} | "
        f"Parallel processes: {training_args.world_size} | Device mapping: {training_args.device} | "
        f"Distributed training: {training_args.parallel_mode == ParallelMode.DISTRIBUTED} | "
        f"Compute precision: {str(model_args.compute_dtype)}"
    )
    transformers.set_seed(
        training_args.seed
    )  # Set random seed for reproducible experimental results
    # (default value 42 will be used if not specially set)

    return model_args, data_args, training_args, finetuning_args, generating_args
