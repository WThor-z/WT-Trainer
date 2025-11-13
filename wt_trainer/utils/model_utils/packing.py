"""Packing utilities for efficient sequence processing.

This module provides functions for packing sequences and preparing attention masks
for efficient processing in transformer models. It includes utilities for
handling block diagonal attention patterns.
"""

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from wt_trainer.args import ModelArguments  # noqa: F401


logger = logging.getLogger(__name__)


def get_seqlens_in_batch(attention_mask: "torch.Tensor") -> "torch.Tensor":
    """Get the sequence lengths in the current batch.

    This function analyzes the attention mask to determine the lengths of individual
    sequences in a packed batch.

    Example:
        Input:
        ```
        [
            [1, 1, 2, 2, 2, 0],
            [1, 2, 2, 3, 3, 3],
        ]
        ```
        
        Output:
        ```
        [2, 3, 1, 2, 3]
        ```

    Args:
        attention_mask: The attention mask tensor of shape (batch_size, sequence_length).

    Returns:
        A tensor containing the lengths of each sequence in the batch.
    """
    bsz = attention_mask.size(0)
    dtype, device = attention_mask.dtype, attention_mask.device
    max_num = torch.max(attention_mask).item()
    counts: torch.Tensor = torch.zeros((bsz, max_num), dtype=dtype, device=device)
    for i in range(max_num):
        counts[:, i] = torch.sum(attention_mask == (i + 1), dim=-1)

    counts = counts.flatten()
    seqlens = counts[counts.nonzero().squeeze(dim=-1)]
    return seqlens


def get_unpad_data(attention_mask: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", int]:
    """Prepare the indices and sequence lengths for flash attention varlen function.

    This function prepares the necessary data structures for using variable-length
    attention mechanisms in flash attention.

    Example:
        Input:
        ```
        [
            [1, 1, 2, 2, 2, 0],
            [1, 2, 2, 3, 3, 3],
        ]
        ```
        
        Output:
        ```
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]
        [0, 2, 5, 6, 8, 11]
        3
        ```

    Args:
        attention_mask: The attention mask tensor of shape (batch_size, sequence_length).

    Returns:
        A tuple containing:
        - indices: indices of non-masked tokens from the flattened sequence.
        - cu_seqlens: the cumulative sequence lengths in the current batch, always starts from 0.
        - max_seqlen_in_batch: the largest sequence length in the current batch.
    """
    seqlens_in_batch = get_seqlens_in_batch(attention_mask)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch


def configure_packing(model_args: "ModelArguments", is_trainable: bool) -> None:
    """Configure model for sequence packing using block diagonal attention.

    This function sets up the model to use block diagonal attention for more
    efficient sequence packing without cross-attention.

    Args:
        model_args: Model arguments containing packing configuration options.
        is_trainable: Whether the model is being configured for training.
    """
    if not is_trainable or not model_args.block_diag_attn:
        return

    import transformers.modeling_flash_attention_utils

    transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data
    logger.info(
        "Using block diagonal attention for sequence packing without cross-attention."
    )
