"""Dataset utilities for data processing.

This module provides utility functions for merging datasets, converting dataset
formats, and splitting datasets for training and validation.
"""

import logging

from datasets import concatenate_datasets
from datasets import DatasetDict
from datasets import interleave_datasets
from datasets.load import DatasetModule
from datasets import Dataset  # noqa: F401
from datasets import IterableDataset

from wt_trainer.args import DataArguments

logger = logging.getLogger(__name__)


def merge_dataset(
    all_datasets: list[Dataset | IterableDataset], data_args: DataArguments, seed: int
) -> Dataset | IterableDataset:
    """Merge multiple datasets to a unified dataset.

    Args:
        all_datasets: List of datasets to merge.
        data_args: Data arguments containing mixing strategy information.
        seed: Random seed for shuffling.

    Returns:
        Merged dataset according to the specified mixing strategy.

    Raises:
        ValueError: If an unknown mixing strategy is provided.
    """
    if len(all_datasets) == 1:
        return all_datasets[0]

    elif data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning(
                "The samples between different datasets will not be mixed in streaming mode."
            )

        return concatenate_datasets(all_datasets)

    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning("We recommend using `mix_strategy=concat` in non-streaming mode.")

        return interleave_datasets(
            datasets=all_datasets,
            probabilities=data_args.interleave_probs,
            seed=seed,
            stopping_strategy=(
                "first_exhausted" if data_args.mix_strategy.endswith("under") else "all_exhausted"
            ),
        )

    else:
        raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")


def get_dataset_module(dataset: Dataset | DatasetDict) -> DatasetModule:
    """Convert dataset or dataset dict to dataset module.

    Args:
        dataset: Dataset or dataset dictionary to convert.

    Returns:
        Dataset module containing the train dataset.
    """
    dataset_module: DatasetModule = {}
    if isinstance(dataset, DatasetDict):  # dataset dict
        if "train" in dataset:
            dataset_module["train_dataset"] = dataset["train"]

    else:  # single dataset
        dataset_module["train_dataset"] = dataset

    return dataset_module


def split_dataset(
    dataset: Dataset | IterableDataset | None,
    data_args: DataArguments,
    seed: int,
) -> DatasetDict:
    """Split the dataset and returns a dataset dict containing train set and validation set.

    Support both map dataset and iterable dataset.

    Args:
        dataset: Dataset to split, or None if no dataset is provided.
        data_args: Data arguments for processing.
        seed: Random seed for shuffling.

    Returns:
        DatasetDict containing the train dataset.
    """
    dataset_dict = {}
    if dataset is not None:
        if data_args.streaming:
            dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

        dataset_dict["train"] = dataset

    return DatasetDict(dataset_dict)
