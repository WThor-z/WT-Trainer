"""Dataset loading and processing module.

This module provides functions for loading, processing, and preparing datasets
for training and evaluation. It handles various dataset formats and applies
necessary preprocessing steps.
"""

import json
import logging
import os
from typing import Literal, TYPE_CHECKING

from datasets import Dataset
from datasets import IterableDataset
from datasets import load_dataset
from datasets.load import DatasetModule
import numpy as np
from transformers import PreTrainedTokenizer

from wt_trainer.args import DataArguments
from wt_trainer.args import ModelArguments
from wt_trainer.args import TrainingArguments
from wt_trainer.utils.const import FILEEXT2TYPE

from .dataset import align_dataset
from .dataset import DatasetAttr
from .dataset import get_dataset_module
from .dataset import merge_dataset
from .dataset import split_dataset
from .processor import PackedSupervisedDatasetProcessor
from .processor import SupervisedDatasetProcessor
from .template import Template

if TYPE_CHECKING:
    from os import PathLike  # noqa: F401

    from .processor import DatasetProcessor  # noqa: F401

logger = logging.getLogger(__name__)


def get_dataset_list(dataset_names: list[str] | None, dataset_dir: str) -> list[DatasetAttr]:
    """Get the attributes of the datasets.

    Args:
        dataset_names: List of dataset names to load, or None if no datasets specified.
        dataset_dir: Directory containing the dataset information JSON file.

    Returns:
        List of DatasetAttr objects containing dataset attributes.

    Raises:
        ValueError: If the dataset info file cannot be opened and dataset names are provided.
    """
    if dataset_names is None:
        dataset_names = []

    config_path = os.path.join(dataset_dir, "dataset_info.json")

    try:
        with open(config_path, encoding="utf-8") as f:
            dataset_info = json.load(f)
    except FileNotFoundError as err:
        if len(dataset_names) != 0:
            raise ValueError(f"Cannot open {config_path} due to {str(err)}.") from err

        dataset_info = None

    dataset_list: list[DatasetAttr] = []
    for name in dataset_names:
        dataset_attr = DatasetAttr("file", dataset_name=dataset_info[name]["file_name"])

        if dataset_info is not None:
            dataset_attr.join(dataset_info[name])
        dataset_list.append(dataset_attr)

    return dataset_list


def _load_single_dataset(
    dataset_attr: DatasetAttr,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dataset | IterableDataset:
    """Load a single dataset and aligns it to the standard format.

    Args:
        dataset_attr: Attributes of the dataset to load.
        model_args: Model arguments for loading.
        data_args: Data arguments for loading.
        training_args: Training arguments for loading.

    Returns:
        Loaded dataset aligned to the standard format.

    Raises:
        ValueError: If files are not found or have mismatched types.
    """
    logger.info(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None

    data_files = []
    local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
    if os.path.isdir(local_path):  # is directory
        for file_name in os.listdir(local_path):
            data_files.append(os.path.join(local_path, file_name))
    elif os.path.isfile(local_path):  # is file
        data_files.append(local_path)
    else:
        raise ValueError(f"File {local_path} not found.")

    data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
    if data_path is None:
        raise ValueError(f"Allowed file types: {','.join(FILEEXT2TYPE.keys())}.")

    if any(
        data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None)
        for data_file in data_files
    ):
        raise ValueError("File types should be identical.")

    dataset = load_dataset(
        path=data_path,
        name=data_name,
        data_dir=data_dir,
        data_files=data_files,
        split=dataset_attr.split,
        cache_dir=model_args.cache_dir,
        token=model_args.hf_hub_token,
        num_proc=data_args.preprocessing_num_workers,
        trust_remote_code=model_args.trust_remote_code,
        streaming=data_args.streaming and dataset_attr.load_from != "file",
    )

    if data_args.streaming and dataset_attr.load_from == "file":
        dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    if dataset_attr.num_samples is not None and not data_args.streaming:
        target_num = dataset_attr.num_samples
        indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
        target_num -= len(indexes)
        if target_num > 0:
            expand_indexes = np.random.choice(len(dataset), target_num)
            indexes = np.concatenate((indexes, expand_indexes), axis=0)

        assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
        dataset = dataset.select(indexes)
        logger.info(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None:  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: list[str] | None,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    return_dict: bool = False,
) -> Dataset | IterableDataset | dict[str, Dataset] | None:
    """Return the merged datasets in the standard format.

    Args:
        dataset_names: List of dataset names to merge, or None if no datasets.
        model_args: Model arguments for loading.
        data_args: Data arguments for processing.
        training_args: Training arguments for processing.
        stage: Training stage (pt, sft, rm, ppo, kto).
        return_dict: Whether to return a dictionary of datasets or merged dataset.

    Returns:
        Merged dataset, dictionary of datasets, or None if no datasets specified.

    Raises:
        ValueError: If a dataset is not applicable for the current training stage.
    """
    if dataset_names is None:
        return None

    datasets = {}
    for dataset_name, dataset_attr in zip(
        dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)
    ):
        if (stage == "rm" and dataset_attr.ranking is False) or (
            stage != "rm" and dataset_attr.ranking is True
        ):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets[dataset_name] = _load_single_dataset(
            dataset_attr, model_args, data_args, training_args
        )

    if return_dict:
        return datasets

    return merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)


def _get_dataset_processor(
    data_args: DataArguments,
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: Template,
    tokenizer: PreTrainedTokenizer,
    do_generate: bool = False,
) -> "DatasetProcessor":
    """Return the corresponding dataset processor.

    Args:
        data_args: Data arguments for processing.
        stage: Training stage (pt, sft, rm, ppo, kto).
        template: Template for formatting.
        tokenizer: Tokenizer for processing.
        do_generate: Whether to prepare for generation.

    Returns:
        Appropriate dataset processor instance.
    """
    dataset_processor_class = SupervisedDatasetProcessor
    if stage == "sft" and not do_generate:
        if data_args.packing:
            if data_args.neat_packing:  # hack datasets to have int32 model mask
                from datasets.arrow_writer import OptimizedTypedSequence
                from datasets.arrow_writer import TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            dataset_processor_class = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_class = SupervisedDatasetProcessor

    return dataset_processor_class(template=template, tokenizer=tokenizer, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Dataset | IterableDataset | None,
    data_args: DataArguments,
    training_args: TrainingArguments,
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: Template,
    tokenizer: PreTrainedTokenizer,
    is_eval: bool = False,
) -> Dataset | IterableDataset | None:
    """Preprocesses the dataset, including format checking and tokenization.

    Args:
        dataset: Dataset to preprocess, or None if no dataset.
        data_args: Data arguments for processing.
        training_args: Training arguments for processing.
        stage: Training stage (pt, sft, rm, ppo, kto).
        template: Template for formatting.
        tokenizer: Tokenizer for processing.
        is_eval: Whether this is for evaluation.

    Returns:
        Preprocessed dataset or None if input dataset was None.

    Raises:
        RuntimeError: If no valid samples can be found in the dataset.
    """
    if dataset is None:
        return None

    dataset_processor = _get_dataset_processor(
        data_args,
        stage,
        template,
        tokenizer,
        do_generate=(training_args.predict_with_generate and is_eval),
    )
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:
        kwargs = {
            "num_proc": data_args.preprocessing_num_workers,
            "load_from_cache_file": (not data_args.overwrite_cache)
            or (training_args.local_process_index != 0),
            "desc": "Running tokenizer on dataset",
        }

    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            logger.info("Data processing success!")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration as exc:
            if stage == "pt":
                raise RuntimeError(
                    "Cannot find sufficient samples, consider increasing dataset size."
                ) from exc

            raise RuntimeError(
                "Cannot find valid samples, check `data/README.md` for the data format."
            ) from exc

    return dataset


def get_dataset(
    template: Template,
    model_args: ModelArguments,
    data_args: DataArguments,
    training_args: TrainingArguments,
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: PreTrainedTokenizer,
) -> DatasetModule:
    """Get the train dataset and optionally gets the evaluation dataset.

    Args:
        template: Template for formatting the dataset.
        model_args: Model arguments for loading.
        data_args: Data arguments for processing.
        training_args: Training arguments for processing.
        stage: Training stage (pt, sft, rm, ppo, kto).
        tokenizer: Tokenizer for processing.

    Returns:
        Dataset module containing the processed datasets.
    """
    # Load and preprocess dataset
    with training_args.main_process_first(
        desc="loading dataset", local=(not data_args.data_shared_file_system)
    ):
        dataset = _get_merged_dataset(
            data_args.dataset, model_args, data_args, training_args, stage
        )

    with training_args.main_process_first(
        desc="pre-process dataset", local=(not data_args.data_shared_file_system)
    ):
        dataset = _get_preprocessed_dataset(
            dataset, data_args, training_args, stage, template, tokenizer, is_eval=False
        )

        dataset_dict = split_dataset(dataset, data_args, seed=training_args.seed)
        if data_args.tokenized_path is not None:  # save tokenized dataset to disk
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info(
                    f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`."
                )

        return get_dataset_module(dataset_dict)


def has_tokenized_data(path: "PathLike") -> bool:
    """Check if the path has a tokenized dataset.

    Args:
        path: Path to check for tokenized dataset.

    Returns:
        True if the path contains a tokenized dataset, False otherwise.
    """
    return os.path.isdir(path) and len(os.listdir(path)) > 0
