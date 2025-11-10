"""Tokenizer utilities for model training."""

import logging
from types import MethodType
from typing import Any, TYPE_CHECKING

from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from wt_trainer.args import ModelArguments  # noqa: F401

logger = logging.getLogger(__name__)


def try_download_model_from_other_hub(model_args: "ModelArguments") -> str:
    """Try to download model from other hub.

    Args:
        model_args: Model arguments.

    Returns:
        Model name or path.
    """
    return model_args.model_name_or_path  # type: ignore


def _get_init_kwargs(model_args: "ModelArguments") -> dict[str, Any]:
    """Get arguments for loading config/tokenizer/model.

    Args:
        model_args: Model arguments.

    Returns:
        Dictionary with initialization arguments.
    """
    model_args.model_name_or_path = try_download_model_from_other_hub(model_args)
    return {
        "trust_remote_code": model_args.trust_remote_code,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }


def _patch_tokenizer(
    tokenizer: "AutoTokenizer",
    model_args: "ModelArguments",
) -> None:
    """Patch the tokenizer with additional configurations.

    Args:
        tokenizer: Tokenizer to patch.
        model_args: Model arguments.
    """
    # __func__ is used to get the unbound function object behind a bound method.
    # This allows direct manipulation of the underlying function without going
    # through the class or instance binding mechanism.
    if "PreTrainedTokenizerBase" not in str(tokenizer._pad.__func__):
        # MethodType binds the _pad method of PreTrainedTokenizerBase to the current tokenizer instance.
        # .pad acts as the padding method for the tokenizer, with padding strategy chosen internally.
        tokenizer._pad = MethodType(PreTrainedTokenizerBase._pad, tokenizer)

    # Check if the parameter is greater than the current maximum length of the tokenizer.
    # If so, update the tokenizer's model_max_length attribute.
    if (
        model_args.model_max_length is not None
        and tokenizer.model_max_length < model_args.model_max_length
    ):
        tokenizer.model_max_length = model_args.model_max_length

    # Add non-special tokens to the tokenizer
    if model_args.add_tokens is not None:
        # num_added_tokens is the number of tokens added
        num_added_tokens = tokenizer.add_tokens(
            new_tokens=model_args.add_tokens, special_tokens=False
        )
        logger.info(f"Add tokens {','.join(model_args.add_tokens)} to tokenizer's vocabulary.")
        if num_added_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True  # Redefine vocabulary size
            logger.warning("New tokens have been added, changed `resize_vocab` to True.")

    # Handle special tokens
    if model_args.add_special_tokens is not None:
        num_added_special_tokens = tokenizer.add_tokens(
            new_tokens=model_args.add_special_tokens, special_tokens=True
        )
        logger.info(
            f"Add special tokens {','.join(model_args.add_special_tokens)} to tokenizer's vocabulary."
        )
        if num_added_special_tokens > 0 and not model_args.resize_vocab:
            model_args.resize_vocab = True
            logger.warning("New special tokens have been added, changed `resize_vocab` to True.")


def load_tokenizer(model_args: "ModelArguments") -> "AutoTokenizer":
    """Load a pre-trained tokenizer with optional processor.

    Note: Includes in-place operations on model arguments.

    Args:
        model_args: Model arguments.

    Returns:
        Loaded tokenizer.

    Raises:
        OSError: If failed to load tokenizer.
    """
    init_kwargs = _get_init_kwargs(model_args)
    try:
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore
            model_args.model_name_or_path,
            use_fast=model_args.use_fast_tokenizer,
            split_special_tokens=model_args.split_special_tokens,
            padding_side="right",
            **init_kwargs,
        )
    except ValueError:  # Do not use padding
        tokenizer = AutoTokenizer.from_pretrained(  # type: ignore
            model_args.model_name_or_path,
            use_fast=not model_args.use_fast_tokenizer,
            padding_side="right",
            **init_kwargs,
        )
    except Exception as e:
        raise OSError("Failed to load tokenizer.") from e

    _patch_tokenizer(tokenizer, model_args)

    return tokenizer
