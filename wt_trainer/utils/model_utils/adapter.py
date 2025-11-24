"""Adapter utilities for model fine-tuning.

This module provides functions to initialize and configure adapters for different
fine-tuning methods such as Full fine-tuning, Freeze fine-tuning, and LoRA/DoRA.
"""

import logging
import re
from typing import TYPE_CHECKING

from peft import get_peft_model
from peft import LoraConfig
from peft import PeftModel
from peft import TaskType
import torch

from wt_trainer.utils.const import QuantizationMethod

if TYPE_CHECKING:
    from transformers import PretrainedConfig  # noqa: F401
    from transformers import PreTrainedModel  # noqa: F401

    from wt_trainer.args import FinetuningArguments  # noqa: F401
    from wt_trainer.args import ModelArguments  # noqa: F401


logger = logging.getLogger(__name__)


def _setup_full_tuning(
    model: "PreTrainedModel",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    """Set up full fine-tuning.

    In full fine-tuning, all model parameters are trainable.

    Args:
        model: The model to set up for fine-tuning.
        is_trainable: Whether the model should be trainable.
        cast_trainable_params_to_fp32: Whether to cast trainable parameters to float32.
    """
    if not is_trainable:
        return

    logger.info("Fine-tuning method: Full")
    for param in model.parameters():
        if cast_trainable_params_to_fp32:
            param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)


def _setup_freeze_tuning(
    model: "PreTrainedModel",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> None:
    """Set up freeze fine-tuning.

    In freeze fine-tuning, only selected layers are trainable.

    Args:
        model: The model to set up for fine-tuning.
        finetuning_args: Fine-tuning arguments.
        is_trainable: Whether the model should be trainable.
        cast_trainable_params_to_fp32: Whether to cast trainable parameters to float32.
    """
    if not is_trainable:
        return

    logger.info("Fine-tuning method: Freeze")
    config = model.config

    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("num_hidden_layers attribute not found in model config")

    if finetuning_args.use_llama_pro:
        if num_layers % finetuning_args.freeze_trainable_layers != 0:
            raise ValueError(
                f"`num_layers` {num_layers} should be "
                f"divisible by `num_layer_trainable` {finetuning_args.freeze_trainable_layers}."
            )

        stride = num_layers // finetuning_args.freeze_trainable_layers
        trainable_layer_ids = range(stride - 1, num_layers + stride - 1, stride)
    elif (
        finetuning_args.freeze_trainable_layers > 0
    ):  # fine-tuning the last n layers if num_layer_trainable > 0
        trainable_layer_ids = range(
            max(0, num_layers - finetuning_args.freeze_trainable_layers), num_layers
        )
    else:  # fine-tuning the first n layers if num_layer_trainable < 0
        trainable_layer_ids = range(min(-finetuning_args.freeze_trainable_layers, num_layers))

    hidden_modules = set()
    non_hidden_modules = set()
    for name, _ in model.named_parameters():
        if ".0." in name:
            hidden_modules.add(name.split(".0.")[-1].split(".")[0])
        elif ".1." in name:  # MoD starts from layer 1
            hidden_modules.add(name.split(".1.")[-1].split(".")[0])

        if re.search(r"\.\d+\.", name) is None:
            non_hidden_modules.add(name.split(".")[-2])  # remove weight/bias

    trainable_layers = []
    freeze_trainable_modules: list[str] = finetuning_args.freeze_trainable_modules or []
    for module_name in freeze_trainable_modules:
        if module_name != "all" and module_name not in hidden_modules:
            raise ValueError(
                f"Module {module_name} is not found, please choose from {', '.join(hidden_modules)}"
            )

        for idx in trainable_layer_ids:
            trainable_layers.append(f".{idx:d}.{module_name if module_name != 'all' else ''}")

    freeze_extra_modules: list[str] = finetuning_args.freeze_extra_modules or []
    if freeze_extra_modules:
        for module_name in freeze_extra_modules:
            if module_name not in non_hidden_modules:
                raise ValueError(
                    f"Module {module_name} is not found, please choose from {', '.join(non_hidden_modules)}"
                )

            trainable_layers.append(module_name)

    for name, param in model.named_parameters():
        if any(trainable_layer in name for trainable_layer in trainable_layers):
            if cast_trainable_params_to_fp32:
                param.data = param.data.to(torch.float32)
        else:
            param.requires_grad_(False)

    logger.info(f"Set trainable layers: {','.join(trainable_layers)}")


def _setup_lora_tuning(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
    cast_trainable_params_to_fp32: bool,
) -> "PeftModel":
    """Set up LoRA/DoRA fine-tuning.

    LoRA (Low-Rank Adaptation) and DoRA (Weight-Decomposed Low-Rank Adaptation)
    are parameter-efficient fine-tuning methods.

    Args:
        model: The model to set up for fine-tuning.
        model_args: Model arguments.
        finetuning_args: Fine-tuning arguments.
        is_trainable: Whether the model should be trainable.
        cast_trainable_params_to_fp32: Whether to cast trainable parameters to float32.

    Returns:
        The model with LoRA/DoRA adapters configured.
    """
    if is_trainable:
        logger.info(f"Fine-tuning method: {'DoRA' if finetuning_args.use_dora else 'LoRA'}")

    adapter_to_resume = None

    if model_args.adapter_name_or_path is not None:
        is_mergeable = True
        if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
            is_mergeable = False

        if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
            adapter_to_merge = model_args.adapter_name_or_path
            adapter_to_resume = model_args.adapter_name_or_path
        else:
            adapter_to_merge = model_args.adapter_name_or_path

        init_kwargs = {
            "subfolder": model_args.adapter_folder,
            "offload_folder": model_args.offload_folder,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }

        if is_mergeable and not adapter_to_resume:
            merged_model = PeftModel.from_pretrained(
                model,
                adapter_to_merge,
                subfolder=init_kwargs["subfolder"],
                offload_folder=init_kwargs["offload_folder"],
                cache_dir=init_kwargs["cache_dir"],
                revision=init_kwargs["revision"],
                token=init_kwargs["token"],
            )
            model = merged_model.merge_and_unload()

        if adapter_to_resume:  # resume lora training
            model = PeftModel.from_pretrained(
                model,
                adapter_to_resume,
                is_trainable=is_trainable,
                subfolder=init_kwargs["subfolder"],
                offload_folder=init_kwargs["offload_folder"],
                cache_dir=init_kwargs["cache_dir"],
                revision=init_kwargs["revision"],
                token=init_kwargs["token"],
            )

        logger.info(f"Loaded adapter(s): {','.join(model_args.adapter_name_or_path)}")

    if is_trainable and adapter_to_resume is None:  # create new lora weights while training
        lora_target: list[str] = finetuning_args.lora_target or []
        if lora_target and lora_target[0] == "all":
            target_modules_set = set()
            for name, module in model.named_modules():
                if (
                    "Linear" in module.__class__.__name__
                    and "Embedding" not in module.__class__.__name__
                    and "lm_head" not in name
                ):
                    target_modules_set.add(name.split(".")[-1])
            target_modules = list(target_modules_set)
        else:
            target_modules = lora_target

        if (
            finetuning_args.use_dora
            and getattr(model, "quantization_method", None) is not None
            and getattr(model, "quantization_method", None) != QuantizationMethod.BNB
        ):
            raise ValueError("DoRA is not compatible with PTQ-quantized models.")

        peft_kwargs = {
            "r": finetuning_args.lora_rank,
            "target_modules": target_modules,
            "lora_alpha": finetuning_args.lora_alpha,
            "lora_dropout": finetuning_args.lora_dropout,
            "use_rslora": finetuning_args.use_rslora,
            "use_dora": finetuning_args.use_dora,
            "modules_to_save": finetuning_args.additional_target,
        }

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            **peft_kwargs,
        )

        # For quantized models, we need to prepare the model for k-bit training
        # especially for non-BNB quantized models (e.g., GPTQ, AWQ, etc.)
        if getattr(model, "quantization_method", None) is not None:

            from peft import prepare_model_for_kbit_training

            # 该函数作用有以下几点：
            # 1、将 LayerNorm 层转换为 fp32
            # 2、使输出嵌入层需要梯度
            # 3、将语言模型头向上转换为 fp32
            # 4、冻结基础模型层
            # !! 所以重点在于不能将embedding设置为requires_grad,否则极大影响训练loss
            model = prepare_model_for_kbit_training(model)
            model.embed_tokens.requires_grad = False
            model.lm_head.requires_grad = False
            model.base_model.embed_tokens.weight.data = (
                model.base_model.embed_tokens.weight.data.to(torch.bfloat16)
            )
            model.lm_head.weight.data = model.lm_head.weight.data.to(torch.bfloat16)

        model = get_peft_model(model, lora_config)

    if is_trainable and cast_trainable_params_to_fp32:
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)

    return model


def init_adapter(
    model: "PreTrainedModel",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    is_trainable: bool,
) -> "PreTrainedModel":
    """Initialize the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.

    Args:
        model: The model to initialize adapters for.
        model_args: Model arguments.
        finetuning_args: Fine-tuning arguments.
        is_trainable: Whether the model should be trainable.

    Returns:
        The model with adapters initialized.

    Raises:
        ValueError: If trying to use non-LoRA fine-tuning with quantized models.
        NotImplementedError: If an unknown fine-tuning type is specified.
    """
    if is_trainable and getattr(model, "quantization_method", None) is not None:
        if finetuning_args.finetuning_type != "lora":
            raise ValueError("Quantized models can only be used for the LoRA tuning.")

        if finetuning_args.pissa_init:
            raise ValueError("Cannot initialize PiSSA adapter on quantized models.")

    # cast trainable parameters to float32 if:
    # 1. is_trainable and not pure_bf16 and not badam and quantization_bit is not None (qlora)
    # 2. is_trainable and not pure_bf16 and not badam and not zero3 (zero3 already in fp32)
    cast_trainable_params_to_fp32 = False
    if is_trainable:
        if finetuning_args.pure_bf16 or finetuning_args.use_badam:
            logger.info("Pure bf16 / BAdam detected, remaining trainable params in half precision.")
        else:
            logger.info("Upcasting trainable params to float32.")
            cast_trainable_params_to_fp32 = True

    if finetuning_args.finetuning_type == "full":
        _setup_full_tuning(model, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "freeze":
        _setup_freeze_tuning(model, finetuning_args, is_trainable, cast_trainable_params_to_fp32)
    elif finetuning_args.finetuning_type == "lora":
        model = _setup_lora_tuning(
            model,
            model_args,
            finetuning_args,
            is_trainable,
            cast_trainable_params_to_fp32,
        )
    else:
        raise NotImplementedError(f"Unknown finetuning type: {finetuning_args.finetuning_type}.")

    return model
