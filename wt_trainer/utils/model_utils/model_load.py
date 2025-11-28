"""Utility functions for loading model state dictionaries with memory analysis.

This module provides functions for loading models with detailed memory analysis,
supporting quantization and LoRA adapters. It includes functions for:
- Calculating memory requirements for different parameter types
- Loading models with minimal memory footprint
- Supporting various quantization schemes like NF4
"""

import inspect
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from safetensors import safe_open
import torch
from tqdm import tqdm
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import AutoModelForImageTextToText
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoModelForTextToWaveform
from transformers import AutoModelForVision2Seq
from transformers import PreTrainedModel
from transformers.modeling_utils import is_deepspeed_zero3_enabled  # type: ignore
from transformers.modeling_utils import is_fsdp_enabled
from transformers.quantizers import AutoHfQuantizer  # type: ignore
from transformers.utils import is_accelerate_available  # type: ignore
from transformers.utils.quantization_config import BitsAndBytesConfig

# Import arguments for better integration
from wt_trainer.args import FinetuningArguments
from wt_trainer.args import ModelArguments
from .model_patch import patch_config

if is_accelerate_available():
    from accelerate import dispatch_model

MEMORY_THRESHOLD = 1024**2

logger = logging.getLogger(__name__)


def calculate_nf4_quant_state_memory(
    in_feature: int, out_feature: int, alignment: int = 512
) -> Dict[str, int]:
    """Calculate the total memory usage of a QuantState object under nf4 quantization.

    This calculation is based on the documentation:
    https://spacemit.feishu.cn/docx/TSV3dUPgSouqdbxlJqLcOV0knkd?from=from_copylink

    Args:
        in_feature: Number of input features.
        out_feature: Number of output features.
        alignment: Memory alignment unit, default is 512 bytes.

    Returns:
        A dictionary containing memory usage in bytes for each component:
        - absmax_bytes: Memory for absmax values
        - code_bytes: Memory for code values
        - offset_bytes: Memory for offset values
        - nest_absmax_bytes: Memory for nested absmax values
        - nest_code_bytes: Memory for nested code values
    """

    def _align_memory(size: int, alignment: int) -> int:
        """Align memory to specified boundary."""
        return ((size + alignment - 1) // alignment) * alignment

    # 1. absmax
    num_blocks = (in_feature * out_feature + 64 - 1) // 64
    absmax_bytes = _align_memory(num_blocks * 1, alignment)

    # 2. code - nf4 projection table has 16 entries
    code_bytes = _align_memory(16 * 4, alignment)

    # 3. offset
    offset_bytes = _align_memory(4, alignment)

    # 4. nested absmax
    nested_num_blocks = ((in_feature * out_feature + 64 - 1) // 64 + 256 - 1) // 256
    nest_absmax_bytes = _align_memory(nested_num_blocks * 4, alignment)

    # 5. nested code
    nest_code_bytes = _align_memory(256 * 4, alignment)

    return {
        "absmax_bytes": absmax_bytes,
        "code_bytes": code_bytes,
        "offset_bytes": offset_bytes,
        "nest_absmax_bytes": nest_absmax_bytes,
        "nest_code_bytes": nest_code_bytes,
    }


def _calculate_weight_memory(
    param: torch.nn.Parameter, quantization_dtype: str | None, alignment: int
) -> tuple[int, int]:
    """Calculate memory usage for a parameter.

    Args:
        param: The parameter to calculate memory for.
        quantization_dtype: Quantization type, e.g., "nf4".
        alignment: Memory alignment in bytes.

    Returns:
        A tuple of (param_bytes, warmup_bytes) for the parameter.
    """
    param_bytes = 0
    warmup_bytes = 0

    if quantization_dtype == "nf4":
        if hasattr(param, "quant_state"):
            # This parameter is quantized, use precise calculation
            param_bytes = int(param.numel() * 0.5)  # Meta quantization info is not trusted
            quant_mem = calculate_nf4_quant_state_memory(param.shape[0], param.shape[1])
            for _, mem in quant_mem.items():
                param_bytes += mem
                warmup_bytes += mem if mem >= MEMORY_THRESHOLD else 0
        else:
            param_bytes = param.nbytes
    else:
        # Not quantized (e.g., embed, lm_head, norm)
        param_bytes = param.nbytes

    aligned = ((int(param_bytes) + alignment - 1) // alignment) * alignment
    warmup_bytes += aligned if aligned >= MEMORY_THRESHOLD else 0

    return aligned, warmup_bytes


def _calculate_lora_memory(
    name: str,
    param: torch.nn.Parameter,
    lora_r: int | None,
    lora_target_modules: List[str],
    lora_bytes_per_param: int,
    alignment: int,
) -> tuple[int, int]:
    """Calculate LoRA adapter memory usage.

    Args:
        name: Parameter name
        param: The parameter to calculate LoRA memory for.
        lora_r: LoRA rank.
        lora_target_modules: List of module names to inject LoRA.
        lora_bytes_per_param: Bytes per LoRA parameter (FP32 = 4 bytes).
        alignment: Memory alignment in bytes.

    Returns:
        A tuple of (lora_bytes, warmup_bytes) for LoRA adapters.
    """
    lora_bytes = 0
    warmup_bytes = 0

    if lora_r is not None and lora_r > 0:
        module_name = name.split(".")[-1] if hasattr(param, "name") else ""

        if module_name in lora_target_modules:
            out_f, in_f = param.shape
            lora_params = lora_r * (in_f + out_f)
            lora_param_bytes = lora_params * lora_bytes_per_param
            lora_aligned = ((lora_param_bytes + alignment - 1) // alignment) * alignment
            lora_bytes += lora_aligned
            warmup_bytes += lora_aligned if lora_aligned >= MEMORY_THRESHOLD else 0

    return lora_bytes, warmup_bytes


def _calculate_buffer_memory(model: PreTrainedModel, alignment: int) -> tuple[int, int]:
    """Calculate buffer memory usage.

    Args:
        model: The model to analyze buffers for.
        alignment: Memory alignment in bytes.

    Returns:
        A tuple of (buffer_bytes, warmup_bytes) for model buffers.
    """
    buffer_bytes = 0
    warmup_bytes = 0

    for _, buf in model.named_buffers():
        if buf.device.type == "meta":
            buf_bytes = buf.nbytes
            aligned = ((int(buf_bytes) + alignment - 1) // alignment) * alignment
            buffer_bytes += aligned
            warmup_bytes += aligned if aligned >= MEMORY_THRESHOLD else 0

    return buffer_bytes, warmup_bytes


def model_load_memory_analyze(
    model: PreTrainedModel,
    quantization_dtype: str | None = None,  # e.g., "nf4"
    lora_r: int | None = None,
    lora_target_modules: List[str] | None = None,
    is_moe: bool | None = None,
    alignment: int = 512,
) -> Dict[str, Any]:
    """Analyze theoretical memory usage for loading a meta-device model,and provide a reasonable
    memory preheating plan.

    Args:
        model: A model on meta device (from `torch.device('meta')`)
        quantization_dtype: If "nf4", assume linear weights are quantized to 0.5B/param
        lora_r: LoRA rank; if given, estimate LoRA adapter memory (FP32)
        lora_target_modules: List of module names to inject LoRA (e.g., ["q_proj", "v_proj", ...])
        is_moe: Whether the model is a MoE model
        alignment: CUDA memory alignment unit (default 512B)

    Returns:
        dict with keys: total_aligned_bytes, weight_bytes, buffer_bytes, lora_bytes, etc.

    Raises:
        ValueError: If model is not on meta device.
    """
    if str(model.device) != "meta":
        raise ValueError("Model must be on meta device for accurate analysis.")

    # Get bytes per element for tensor dtype
    # (Note: this refers to the original model weights, quantized models need to be
    # considered based on quantization_dtype)

    lora_bytes_per_param = 4  # FP32

    # TODO: Add LoRA configuration (LoRA scheme for MoE models has not been added yet)
    if lora_target_modules is None:
        if not is_moe:
            lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        else:
            lora_target_modules = [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ]  # MoE models do not add LoRA to gate layers by default

    weight_bytes = 0
    lora_bytes = 0
    total_params = 0
    warm_up_bytes = 0  # Record required GPU memory warmup size

    for name, param in model.named_parameters():
        numel = param.numel()
        total_params += numel

        # Calculate parameter memory
        param_aligned, param_warmup = _calculate_weight_memory(param, quantization_dtype, alignment)
        weight_bytes += param_aligned
        warm_up_bytes += param_warmup

        # Calculate LoRA memory
        lora_param_bytes, lora_warmup = _calculate_lora_memory(
            name, param, lora_r, lora_target_modules, lora_bytes_per_param, alignment
        )
        lora_bytes += lora_param_bytes
        warm_up_bytes += lora_warmup

    # Analyze buffers (e.g., rotary_emb.inv_freq), default is meta device
    buffer_bytes, buffer_warmup = _calculate_buffer_memory(model, alignment)
    warm_up_bytes += buffer_warmup

    total_aligned = weight_bytes + lora_bytes + buffer_bytes

    return {
        "total_aligned_bytes": total_aligned,
        "weight_bytes": weight_bytes,
        "lora_bytes": lora_bytes,
        "buffer_bytes": buffer_bytes,
        "total_params": total_params,
        "warm_up_bytes": warm_up_bytes,
    }


def model_load_with_low_cost(
    model_args: ModelArguments,
    ft_args: FinetuningArguments,
    is_trainable: bool,
) -> PreTrainedModel:
    """Load a pretrained model with memory analysis.

    Args:
        model_args: Model arguments containing model configuration.
        ft_args: Finetune arguments containing LoRA etc. configuration.

    Returns:
        Loaded model.
    """
    model_name_or_path = model_args.model_name_or_path
    device_map = model_args.device_map
    device = "cuda" if torch.cuda.is_available() else "cpu"  # 默认设备
    if device_map and "" in device_map:
        device = device_map[""].type
    dtype = model_args.compute_dtype if model_args.compute_dtype is not None else torch.float16

    # Determine quantization type from model_args
    quantization_dtype = None
    if model_args.quantization_bit is not None and model_args.quantization_bit == 4:
        quantization_dtype = model_args.quantization_type  # "nf4" or "fp4"

    # Automatically identify configuration
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

    # Update the model configuration and whether to use the liger kernel
    patch_config(config, model_args, is_trainable)
    if model_args.enable_liger_kernel:
        from .liger_kernel_model import liger_kernel_model

        liger_kernel_model(
            config,
            model_args,
            ft_args,
            is_trainable,
            require_logits=(ft_args.stage not in ["pt", "sft"]),
        )

    # Build quantization configuration
    hf_quantizer = None
    if quantization_dtype == "nf4":
        quantization_config = BitsAndBytesConfig(  # type: ignore
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        hf_quantizer = AutoHfQuantizer.from_config(quantization_config, pre_quantized=False)
        hf_quantizer.validate_environment()

    if type(config) in AutoModelForVision2Seq._model_mapping.keys():
        model_class = AutoModelForVision2Seq
    elif type(config) in AutoModelForImageTextToText._model_mapping.keys():
        model_class = AutoModelForImageTextToText  # type: ignore
    elif type(config) in AutoModelForSeq2SeqLM._model_mapping.keys():
        model_class = AutoModelForSeq2SeqLM  # type: ignore
    elif type(config) in AutoModelForTextToWaveform._model_mapping.keys():
        model_class = AutoModelForTextToWaveform  # type: ignore
    else:
        model_class = AutoModelForCausalLM  # type: ignore

    # We've tried Automodel.from_pretrained(device_map = "meta"), should avoid this approach
    # Because it traverses weight files to know the structure, this will cause an extreme
    # memory peak and ultra-high CPU utilization. This approach is safer with less CPU pressure
    with torch.device("meta"):
        model = model_class.from_config(config, trust_remote_code=True)  # type: ignore

    # Preprocess base_model into quantization_model
    if hf_quantizer is not None:
        hf_quantizer.preprocess_model(
            model=model,
            device_map={"": device} if device_map is None else device_map,
            keep_in_fp32_modules=getattr(model, "_keep_in_fp32_modules", None),
            config=config,
            use_kernels=False,
        )
        model.hf_quantizer = hf_quantizer

    analysis = model_load_memory_analyze(
        model=model,
        quantization_dtype=quantization_dtype,
        lora_r=ft_args.lora_rank if hasattr(ft_args, "lora_rank") else None,
        # Users need to modify it themselves, and the LoRA target modules are the same as above
    )

    logger.info(
        f"Model loaded in {dtype if quantization_dtype is None else quantization_dtype} format | "
        f"Expected to need {analysis['weight_bytes']/1024**2:.1f} MB | "
        f"Full training estimated to need {analysis['total_aligned_bytes']/1024**2:.1f} MB"
    )

    # GPU memory warmup
    warm_up = torch.empty((1, analysis["warm_up_bytes"]), dtype=torch.uint8, device=device)
    del warm_up

    # TODO: Currently for language model loading, not sure if other task models can also be loaded
    model = load_model(
        model_name_or_path, model, hf_quantizer=hf_quantizer, dtype=dtype, device=device
    )

    # 确保模型在正确的设备上
    if hasattr(model, "to"):
        model = model.to(device)

    model.eval()
    model.requires_grad_(False)

    return model


@torch.no_grad()
def load_model(
    model_name_or_path: str,
    model: PreTrainedModel,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    hf_quantizer: AutoHfQuantizer | None = None,
) -> PreTrainedModel:
    """Load model weights from files.

    Args:
        model_name_or_path: Path to the model directory.
        model: Model instance to load weights into.
        dtype: Data type for loaded weights.
        device: Device to load the model on.
        hf_quantizer: Quantizer for handling quantized parameters.

    Returns:
        Model with loaded weights.
    """
    single_file_path = Path(model_name_or_path) / "model.safetensors"
    shard_files_path = Path(model_name_or_path) / "model.safetensors.index.json"

    # Process model file information
    weight_map = {}
    if single_file_path.exists():
        # Single file mode
        is_sharded = False
    elif shard_files_path.exists():
        # Sharded mode
        with open(shard_files_path, "r", encoding="utf-8") as f:
            index = json.load(f)
        weight_map = index["weight_map"]
        is_sharded = True
    else:
        raise FileNotFoundError(
            f"Could not find safetensors weight files. Please confirm that {model_name_or_path} "
            "contains `model.safetensors` or `model.safetensors.index.json`."
        )

    # Read weight information
    if is_sharded:
        shard_to_params: dict[Path, list[Any]] = {}
        for param_name, shard_name in weight_map.items():
            shard_path = Path(model_name_or_path) / shard_name
            shard_to_params.setdefault(shard_path, []).append(param_name)
    else:
        all_param_names = []
        with safe_open(single_file_path, framework="pt", device="cpu") as f:  # type: ignore
            all_param_names = list(f.keys())
        shard_to_params = {single_file_path: all_param_names}

    # Prepare Progress Bar
    seen_shards = []
    shard_order = []
    for name, _ in model.named_parameters():
        if is_sharded:
            if name == "model.lm_head" and model.config.tie_word_embeddings:
                continue
            shard_path = Path(model_name_or_path) / weight_map[name]
            if shard_path not in seen_shards:
                seen_shards.append(shard_path)
                shard_order.append(shard_path)
        else:
            shard_order = [single_file_path]
            break

    pbar = tqdm(total=len(shard_order), desc="Loading shards", unit="file")
    current_shard_index = -1

    # Current shard cache (avoid repeated opening)
    current_shard_data: dict[str, Any] | None = None
    current_shard_path = None

    total_mem = 0

    if model is None:
        raise AttributeError(
            f"Model does not exist! Please check if {model_name_or_path} "
            f"and its subfolders contain model weight files."
        )

    is_quantized = hf_quantizer is not None  # Quantization flag

    alignment = 512  # Very important, each architecture's memory alignment strategy
    # is different, CUDA uses 512B as the minimum memory unit

    for name, _ in model.named_parameters():
        param = model.get_parameter(name)

        # Select file to load
        target_shard = None
        if is_sharded:
            if name not in weight_map:
                raise KeyError(
                    f"Param {name} not in weight_map. Keys start with: "
                    f"{list(weight_map.keys())[:3]}"
                )
            target_shard = Path(model_name_or_path) / weight_map[name]
        else:
            target_shard = single_file_path

        # If shard is not loaded, load it (and unload the previous one)
        if current_shard_path != target_shard:
            # Explicitly clear cache
            if current_shard_data is not None:
                current_shard_data = None

            # Update Progress Bar
            if target_shard in shard_order:
                current_shard_index = shard_order.index(target_shard)
                pbar.update(current_shard_index - pbar.n)
                pbar.set_postfix({"file": target_shard.name})

            current_shard_data = {}

            # It's better to use CPU to read weight files and then gradually load to GPU
            # This avoids GPU memory pressure and CPU pressure is not that high
            # since only one shard file is loaded at a time
            current_shard_data = safe_open(target_shard, framework="pt", device="cpu")  # type: ignore
            current_shard_path = target_shard

        # Additional check here because some models are a bit confusing -
        # model.config allows shared weights, but model files put lm_head and embed_tokens in
        if name == "model.lm_head":
            if model.config.tie_word_embeddings:
                logger.info(
                    "Skipping loading lm_head.weight (will share weights with embed_tokens)"
                )
                continue

        # Quantization process
        if is_quantized and hf_quantizer.requires_parameters_quantization:  # type: ignore
            # Need to determine if quantizer needs quantization
            # (some formats are already quantized)
            if hf_quantizer.check_quantized_param(  # type: ignore
                model,
                # At this point, because it's meta, the model parameters are not loaded by default
                param_value=None,
                param_name=name,
                # There are many optional items, quantizer depends on param_name, here we use weight_map
                state_dict=weight_map,
                device_map={"": device},
            ):
                # Core logic
                hf_quantizer.create_quantized_param(  # type: ignore
                    model,
                    current_shard_data.get_tensor(name),  # type: ignore
                    name,
                    device,
                    state_dict={},  # Same as above
                    unexpected_keys=[],
                )
                continue  # Skip the loading logic below

        # Read weights and create a new Parameter object
        weight = current_shard_data.get_tensor(name).to(dtype).to(device)  # type: ignore
        new_param = torch.nn.Parameter(weight)

        # Replace meta Parameter in the model
        *path, last = name.split(".")
        parent = model
        for p in path:
            parent = getattr(parent, p)
        setattr(parent, last, new_param)

        aligned = ((param.nbytes + alignment - 1) // alignment) * alignment
        total_mem += aligned

    # Cleanup
    if current_shard_data is not None:
        current_shard_data = None

    # Ensure the progress bar shows 100% completion
    pbar.n = len(shard_order)
    pbar.refresh()
    pbar.close()  # close the progress bar

    # Shared weights (will be determined based on Model Config)
    model.tie_weights()  # type: ignore

    # Cannot directly patch the entire model because some buffers are still metadata
    # Need to initialize this part of data first
    if device is not None:
        for name, buffer in list(model.named_buffers()):
            if buffer.device.type == "meta":
                # rotary_emb.inv_freq is generally float32
                new_buffer = torch.empty_like(buffer, device=device)
                *path, last = name.split(".")
                parent = model
                for p in path:
                    parent = getattr(parent, p)
                setattr(parent, last, new_buffer)
                aligned = ((buffer.nbytes + alignment - 1) // alignment) * alignment
                total_mem += aligned

    if device is not None:  # TODO: This is only for buffer usage, usability is not that high
        device_map_kwargs = {
            "device_map": {"": device},
            "offload_dir": None,
            "offload_index": None,
            "offload_buffers": False,
        }
        if "skip_keys" in inspect.signature(dispatch_model).parameters:
            device_map_kwargs["skip_keys"] = model._skip_keys_device_placement

        if not is_fsdp_enabled() and not is_deepspeed_zero3_enabled():  # type: ignore
            dispatch_model(model, **device_map_kwargs)

    torch.cuda.empty_cache()

    return model
