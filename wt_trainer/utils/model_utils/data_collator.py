"""Data collator utilities for model training."""

from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import DataCollatorForSeq2Seq


@dataclass
class OptimizedSFTDataCollator(DataCollatorForSeq2Seq):
    """Optimized data collator for pure text SFT tasks.
    
    This data collator:
    - Removes multimodal processing
    - Efficiently supports sequence packing (sample_packing)
    - Avoids O(n^2) 4D masks, prioritizing FlashAttention2's packing optimization
    """

    # Whether block diagonal attention is needed (usually not, as FlashAttention2 is optimized)
    block_diag_attn: bool = False
    attn_implementation: str = "flash_attention_2"
    compute_dtype: torch.dtype = torch.bfloat16

    def __call__(
        self, features: List[Dict[str, List[int] | torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Execute data collation.
        
        Assumes input features contain 'input_ids', 'attention_mask', and 'labels'.
        
        Args:
            features: List of feature dictionaries containing input data.
            
        Returns:
            Batched tensor dictionary ready for model training.
        """
        batch = super().__call__(features)
        if self.block_diag_attn and self.attn_implementation != "flash_attention_2":
            batch["attention_mask"] = prepare_4d_attention_mask(
                batch["attention_mask"], self.compute_dtype
            )

        for key in ["input_ids", "attention_mask", "labels"]:
            if key in batch and torch.is_tensor(batch[key]) and torch.is_floating_point(batch[key]):
                batch[key] = batch[key].to(self.compute_dtype)

        return batch


def prepare_4d_attention_mask(
    attention_mask_with_indices: torch.Tensor, dtype: torch.dtype, max_seq_len_for_4d: int = 2048
) -> torch.Tensor:
    """Prepare 4D attention mask from attention mask with indices.
    
    Args:
        attention_mask_with_indices: Input attention mask tensor with indices.
        dtype: Data type for the output tensor.
        max_seq_len_for_4d: Maximum sequence length allowed for 4D attention mask.
        
    Returns:
        4D attention mask tensor.
        
    Raises:
        ValueError: If sequence length exceeds the maximum allowed length.
    """
    _, seq_len = attention_mask_with_indices.size()
    if seq_len > max_seq_len_for_4d:
        raise ValueError(
            f"Sequence length {seq_len} exceeds maximum allowed length {max_seq_len_for_4d} for 4D attention mask. "
            f"This would cause excessive memory usage. Consider using FlashAttention2 or reducing sequence length."
        )

    min_dtype = torch.finfo(dtype).min
    zero_tensor = torch.tensor(0, dtype=dtype)

    non_padding_mask = (attention_mask_with_indices != 0).unsqueeze(1).unsqueeze(2)
    indices = attention_mask_with_indices.unsqueeze(1).unsqueeze(2)
    indices_t = attention_mask_with_indices.unsqueeze(1).unsqueeze(3)
    tril_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    attention_mask_4d = (indices == indices_t) & non_padding_mask & tril_mask
    attention_mask_4d = torch.where(attention_mask_4d, zero_tensor, min_dtype)
    return attention_mask_4d
