"""Model arguments for configuring model loading and training.

This module provides dataclasses for configuring model-related arguments,
including base model arguments, quantization arguments, and a combined
ModelArguments class that inherits from both.
"""

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from dataclasses import fields
from typing import Any, List, Literal

from wt_trainer.utils.const import AttentionFunction
from wt_trainer.utils.const import EngineName
from wt_trainer.utils.const import QuantizationMethod
from wt_trainer.utils.const import RopeScaling


@dataclass
class _BaseModelArguments:
    """Base model arguments for configuring model loading and training."""

    model_name_or_path: str | None = field(
        default=None,
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    adapter_name_or_path: str | List[str] | None = field(
        default=None,
        metadata={
            "help": (
                "Path to the adapter weight or identifier from huggingface.co/models. "
                "Use commas to separate multiple adapters."
            )
        },
    )
    adapter_folder: str | None = field(
        default=None,
        metadata={"help": "The folder containing the adapter weights to load."},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={
            "help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."
        },
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."
        },
    )
    resize_vocab: bool = field(
        default=False,
        metadata={"help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    split_special_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the special tokens should be split during the tokenization process."
        },
    )
    add_tokens: str | List[str] | None = field(
        default=None,
        metadata={
            "help": "Non-special tokens to be added into the tokenizer. Use commas to separate multiple tokens."
        },
    )
    add_special_tokens: str | List[str] | None = field(
        default=None,
        metadata={
            "help": "Special tokens to be added into the tokenizer. Use commas to separate multiple tokens."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    rope_scaling: RopeScaling | None = field(
        default=None,
        metadata={"help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    )
    flash_attn: AttentionFunction = field(
        default=AttentionFunction.AUTO,
        metadata={"help": "Enable FlashAttention for faster training and inference."},
    )
    shift_attn: bool = field(
        default=False,
        metadata={"help": "Enable shift short model (S^2-Attn) proposed by LongLoRA."},
    )
    mixture_of_depths: Literal["convert", "load"] | None = field(
        default=None,
        metadata={"help": "Convert the model to mixture-of-depths (MoD) or load the MoD model."},
    )
    use_unsloth: bool = field(
        default=False,
        metadata={"help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )
    use_unsloth_gc: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use unsloth's gradient checkpointing (no need to install unsloth)."
        },
    )
    enable_liger_kernel: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable liger kernel for faster training."},
    )
    moe_aux_loss_coef: float | None = field(
        default=None,
        metadata={"help": "Coefficient of the auxiliary router loss in mixture-of-experts model."},
    )
    disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    use_reentrant_gc: bool = field(
        default=True,
        metadata={"help": "Whether or not to use reentrant gradient checkpointing."},
    )
    upcast_layernorm: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the layernorm weights in fp32."},
    )
    upcast_lmhead_output: bool = field(
        default=False,
        metadata={"help": "Whether or not to upcast the output of lm_head in fp32."},
    )
    train_from_scratch: bool = field(
        default=False,
        metadata={"help": "Whether or not to randomly initialize the model weights."},
    )
    infer_backend: EngineName = field(
        default=EngineName.HF,
        metadata={"help": "Backend engine used at inference."},
    )
    offload_folder: str = field(
        default="offload",
        metadata={"help": "Path to offload model weights."},
    )
    use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    infer_dtype: Literal["auto", "float16", "bfloat16", "float32"] = field(
        default="auto",
        metadata={"help": "Data type for model weights and activations at inference."},
    )
    hf_hub_token: str | None = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    ms_hub_token: str | None = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    om_hub_token: str | None = field(
        default=None,
        metadata={"help": "Auth token to log in with Modelers Hub."},
    )
    print_param_status: bool = field(
        default=False,
        metadata={
            "help": "For debugging purposes, print the status of the parameters in the model."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether to trust the execution of code from datasets/models defined on the Hub or not."
        },
    )

    def __post_init__(self) -> None:
        """Additional initialization operations."""
        if self.model_name_or_path is None:
            raise ValueError("Please provide `model_name_or_path`.")

        if self.split_special_tokens and self.use_fast_tokenizer:
            raise ValueError("`split_special_tokens` is only supported for slow tokenizers.")

        if self.adapter_name_or_path is not None:  # support merging multiple lora weights
            self.adapter_name_or_path = [
                path.strip() for path in self.adapter_name_or_path.split(",")  # type: ignore
            ]

        if self.add_tokens is not None:  # support multiple tokens
            self.add_tokens = [token.strip() for token in self.add_tokens.split(",")]  # type: ignore

        if self.add_special_tokens is not None:  # support multiple special tokens
            self.add_special_tokens = [
                token.strip() for token in self.add_special_tokens.split(",")  # type: ignore
            ]


@dataclass
class _QuantizationArguments:
    """Arguments pertaining to the quantization method."""

    quantization_method: QuantizationMethod = field(
        default=QuantizationMethod.BNB,
        metadata={"help": "Quantization method to use for on-the-fly quantization."},
    )
    quantization_bit: int | None = field(
        default=None,
        metadata={
            "help": "The number of bits to quantize the model using on-the-fly quantization."
        },
    )
    quantization_type: Literal["fp4", "nf4"] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in bitsandbytes int4 training."},
    )
    double_quantization: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use double quantization in bitsandbytes int4 training."
        },
    )
    quantization_device_map: Literal["auto"] | None = field(
        default=None,
        metadata={
            "help": "Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."
        },
    )


@dataclass
class ModelArguments(_QuantizationArguments, _BaseModelArguments):
    """Model arguments combining base model arguments and quantization arguments."""

    import torch

    compute_dtype: torch.dtype | None = field(
        default=None,
        init=False,
        metadata={
            "help": "Torch data type for computing model outputs, derived from `fp/bf16`. Do not specify it."
        },
    )
    device_map: str | dict[str, Any] | None = field(
        default=None,
        init=False,
        metadata={
            "help": "Device map for model placement, derived from training stage. Do not specify it."
        },
    )
    model_max_length: int | None = field(
        default=None,
        init=False,
        metadata={
            "help": "The maximum input length for model, derived from `cutoff_len`. Do not specify it."
        },
    )
    block_diag_attn: bool = field(
        default=False,
        init=False,
        metadata={
            "help": "Whether use block diag model or not, derived from `neat_packing`. Do not specify it."
        },
    )

    def __post_init__(self) -> None:
        """Initialize parent classes."""
        _BaseModelArguments.__post_init__(self)

    @classmethod
    def copyfrom(cls, source: "ModelArguments", **kwargs: Any) -> "ModelArguments":
        """Copy from source object with optional overrides.

        Args:
            source: Source object to copy from
            **kwargs: Additional arguments to override

        Returns:
            New instance with copied and overridden attributes
        """
        init_args, lazy_args = {}, {}
        for attr in fields(source):
            if attr.init:
                init_args[attr.name] = getattr(source, attr.name)
            else:
                lazy_args[attr.name] = getattr(source, attr.name)

        init_args.update(kwargs)
        result = cls(**init_args)
        for name, value in lazy_args.items():
            setattr(result, name, value)

        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary.

        Returns:
            Dictionary representation of the object
        """
        args = asdict(self)
        args = {k: f"<{k.upper()}>" if k.endswith("token") else v for k, v in args.items()}
        return args
