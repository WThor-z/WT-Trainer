"""
Liger Kernel model utilities for optimizing model training.

This module provides functionality to apply Liger Kernel optimizations to transformer models
for improved training performance.
"""

import inspect
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PretrainedConfig  # noqa: F401

    from wt_trainer.args.args_params import ModelArguments  # noqa: F401
    from wt_trainer.args.args_params import FineTuningArguments  # noqa: F401


logger = logging.getLogger(__name__)


def _parse_liger_kernel_strategy(strategy: str) -> str | list[str]:
    """Parse the Liger Kernel strategy string into appropriate format.

    Args:
        strategy: The strategy string to parse. Can be "auto", "patch", or
                  a comma-separated list of components.

    Returns:
        Either a string ("auto" or "patch") or a list of components.
    """
    strategy = strategy.strip()
    if strategy in ("auto", "patch"):
        return strategy
    else:
        # Assume it's a comma-separated list
        return [part.strip() for part in strategy.split(",") if part.strip()]


def liger_kernel_model(
    config: "PretrainedConfig",
    model_args: "ModelArguments",
    ft_args: "FinetuningArguments",
    is_trainable: bool,
    require_logits: bool,
) -> None:
    """Apply Liger Kernel optimizations to the model.

    Args:
        config: The model configuration object.
        model_args: Model arguments containing Liger Kernel settings.
        ft_args: Fine-tuning arguments.
        is_trainable: Whether the model is trainable.
        require_logits: Whether logits are required.

    Raises:
        ValueError: If Liger Kernel strategy is not specified or is invalid.
        NotImplementedError: If auto strategy is used or model type is not supported.
    """
    liger_kernel_strategy = model_args.liger_kernel_strategy
    if liger_kernel_strategy and is_trainable:
        liger_kernel_strategy = _parse_liger_kernel_strategy(liger_kernel_strategy)
        if liger_kernel_strategy == "auto":
            raise NotImplementedError("Auto loading of Liger Kernel is not yet supported")
        elif liger_kernel_strategy == "patch":
            model_type = getattr(config, "model_type", None)
            if model_type == "llama4":
                from liger_kernel.transformers import apply_liger_kernel_to_llama4 as apply_lk
            elif model_type == "llama":
                from liger_kernel.transformers import apply_liger_kernel_to_llama as apply_lk
            elif model_type == "mistral":
                from liger_kernel.transformers import apply_liger_kernel_to_mistral as apply_lk
            elif model_type == "mixtral":
                from liger_kernel.transformers import apply_liger_kernel_to_mixtral as apply_lk
            elif model_type == "gemma":
                from liger_kernel.transformers import apply_liger_kernel_to_gemma as apply_lk
            elif model_type == "gemma2":
                from liger_kernel.transformers import apply_liger_kernel_to_gemma2 as apply_lk
            elif model_type == "gemma3_text":
                from liger_kernel.transformers import apply_liger_kernel_to_gemma3_text as apply_lk
            elif model_type == "qwen2":
                from liger_kernel.transformers import apply_liger_kernel_to_qwen2 as apply_lk
            elif model_type == "qwen3":
                from liger_kernel.transformers import apply_liger_kernel_to_qwen3 as apply_lk
            elif model_type == "qwen3_moe":
                from liger_kernel.transformers import apply_liger_kernel_to_qwen3_moe as apply_lk
            elif model_type == "glm4":
                from liger_kernel.transformers import apply_liger_kernel_to_glm4 as apply_lk
            else:
                raise NotImplementedError(
                    f"Liger Kernel is not supported for model type: {model_type}"
                )

            # Enable different cross-entropy loss calculations based on training method
            if (
                require_logits
                and "fused_linear_cross_entropy" in inspect.signature(apply_lk).parameters
            ):
                logger.info("Current training stage does not support chunked cross entropy.")
                kwargs = {"fused_linear_cross_entropy": False, "cross_entropy": True}
            else:
                kwargs = {}

            apply_lk(**kwargs)

        else:
            # TODO: Custom module usage is not yet supported
            if isinstance(liger_kernel_strategy, list):
                if "RMSNorm" in liger_kernel_strategy:
                    from liger_kernel.transformers import LigerRMSNorm
                if "LayerNorm" in liger_kernel_strategy:
                    from liger_kernel.transformers import LigerLayerNorm
                if "RoPE" in liger_kernel_strategy:
                    from liger_kernel.transformers import liger_rotary_pos_emb
                if "SwiGLU" in liger_kernel_strategy:
                    from liger_kernel.transformers import LigerSwiGLUMLP
                if "GeGLU" in liger_kernel_strategy:
                    from liger_kernel.transformers import LigerGEGLUMLP
                if "CrossEntropy" in liger_kernel_strategy:
                    from liger_kernel.transformers import LigerCrossEntropyLoss
                if "Fused Linear CrossEntropy" in liger_kernel_strategy:
                    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
                if "Multi Token Attention" in liger_kernel_strategy:
                    from liger_kernel.transformers import LigerMultiTokenAttention
                if "Softmax" in liger_kernel_strategy:
                    from liger_kernel.transformers import LigerSoftmax
                if "Sparsemax" in liger_kernel_strategy:
                    from liger_kernel.transformers import LigerSparsemax

                if "alignment_kernel" in liger_kernel_strategy:
                    if ft_args.stage == "cpo":
                        from liger_kernel.chunked_loss import LigerFusedLinearCPOLoss
                    elif ft_args.stage == "dpo":
                        from liger_kernel.chunked_loss import LigerFusedLinearDPOLoss
                    elif ft_args.stage == "orpo":
                        from liger_kernel.chunked_loss import LigerFusedLinearORPOLoss
                    elif ft_args.stage == "simpo":
                        from liger_kernel.chunked_loss import LigerFusedLinearSimPOLoss
                    elif ft_args.stage == "kto":
                        from liger_kernel.chunked_loss import LigerFusedLinearKTOLoss

                if "distillation_kernel" in liger_kernel_strategy:
                    from liger_kernel.transformers import LigerKLDIVLoss
                    from liger_kernel.transformers import LigerJSD
                    from liger_kernel.transformers import LigerFusedLinearJSD
                    from liger_kernel.transformers import LigerTVDLoss

                if "exp_kernel" in liger_kernel_strategy:
                    from liger_kernel.transformers.experimental import LigerEmbedding
                    from liger_kernel.transformers.experimental import matmul

            else:
                raise ValueError("Invalid format for liger_kernel_strategy parameter")

    else:
        raise ValueError("Liger kernel strategy must be explicitly specified")

    logger.info("Liger kernel has been applied to the model.")
