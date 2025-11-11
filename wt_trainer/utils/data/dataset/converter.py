"""Dataset converter module for standardizing various dataset formats.

This module provides classes and functions to convert different dataset formats
(e.g., Alpaca, ShareGPT) into a unified format suitable for training.
"""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
import logging
from typing import Any

from datasets import Dataset
from datasets import IterableDataset

from wt_trainer.args import DataArguments
from wt_trainer.args import TrainingArguments
from wt_trainer.utils.const import Role
from .dataset_attr import DatasetAttr

logger = logging.getLogger(__name__)


@dataclass
class DatasetConverter(ABC):
    """Abstract base class for dataset converters.

    This class defines the interface for converting various dataset formats
    into a standardized format.

    Attributes:
        dataset_attr: Dataset attributes containing format information.
        data_args: Data arguments for processing.
    """

    dataset_attr: DatasetAttr
    data_args: DataArguments

    @abstractmethod
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert a single example in the dataset to the standard format.

        Args:
            example: A dictionary representing a single dataset example.

        Returns:
            A dictionary with the standardized format containing keys:
            _prompt, _response, _system, and _tools.
        """
        ...


@dataclass
class AlpacaDatasetConverter(DatasetConverter):
    """Converter for Alpaca-style datasets.

    Converts datasets in the Alpaca format to the standardized format.
    """

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert an Alpaca-style example to the standard format.

        Args:
            example: A dictionary representing a single Alpaca-style example.

        Returns:
            A dictionary with the standardized format.
        """
        prompt = []
        if self.dataset_attr.history and isinstance(example[self.dataset_attr.history], list):
            for old_prompt, old_response in example[self.dataset_attr.history]:
                prompt.append({"role": Role.USER.value, "content": old_prompt})
                prompt.append({"role": Role.ASSISTANT.value, "content": old_response})

        query = []
        if self.dataset_attr.prompt and example[self.dataset_attr.prompt]:
            query.append(example[self.dataset_attr.prompt])

        if self.dataset_attr.query and example[self.dataset_attr.query]:
            query.append(example[self.dataset_attr.query])

        prompt.append({"role": Role.USER.value, "content": "\n".join(query)})  # "prompt\nquery"

        if self.dataset_attr.kto_tag and isinstance(
            example[self.dataset_attr.kto_tag], bool
        ):  # kto example
            response = [
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}
            ]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], str)
            and isinstance(example[self.dataset_attr.rejected], str)
        ):  # pairwise example
            response = [
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.chosen]},
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.rejected]},
            ]
        elif self.dataset_attr.response and isinstance(
            example[self.dataset_attr.response], str
        ):  # normal example
            response = [
                {"role": Role.ASSISTANT.value, "content": example[self.dataset_attr.response]}
            ]
        else:  # unsupervised
            response = []

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": example[self.dataset_attr.system] if self.dataset_attr.system else "",
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
        }
        return output


@dataclass
class SharegptDatasetConverter(DatasetConverter):
    """Converter for ShareGPT-style datasets.

    Converts datasets in the ShareGPT format to the standardized format.
    """

    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        """Convert a ShareGPT-style example to the standard format.

        Args:
            example: A dictionary representing a single ShareGPT-style example.

        Returns:
            A dictionary with the standardized format.
        """
        tag_mapping = {
            self.dataset_attr.user_tag: Role.USER.value,
            self.dataset_attr.assistant_tag: Role.ASSISTANT.value,
            self.dataset_attr.observation_tag: Role.OBSERVATION.value,
            self.dataset_attr.function_tag: Role.FUNCTION.value,
            self.dataset_attr.system_tag: Role.SYSTEM.value,
        }
        odd_tags = (self.dataset_attr.user_tag, self.dataset_attr.observation_tag)
        even_tags = (self.dataset_attr.assistant_tag, self.dataset_attr.function_tag)
        accept_tags = (odd_tags, even_tags)
        messages = example[self.dataset_attr.messages]
        if (
            self.dataset_attr.system_tag
            and len(messages) != 0
            and messages[0][self.dataset_attr.role_tag] == self.dataset_attr.system_tag
        ):
            system = messages[0][self.dataset_attr.content_tag]
            messages = messages[1:]
        else:
            system = example[self.dataset_attr.system] if self.dataset_attr.system else ""

        aligned_messages = []
        broken_data = False
        for turn_idx, message in enumerate(messages):
            if message[self.dataset_attr.role_tag] not in accept_tags[turn_idx % 2]:
                logger.warning(f"Invalid role tag in {messages}.")
                broken_data = True
                break

            aligned_messages.append(
                {
                    "role": tag_mapping[message[self.dataset_attr.role_tag]],
                    "content": message[self.dataset_attr.content_tag],
                }
            )

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning(f"Invalid message count in {messages}.")
            broken_data = True

        if broken_data:
            logger.warning("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(
            example[self.dataset_attr.kto_tag], bool
        ):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning(f"Invalid role tag in {[chosen, rejected]}.")
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
        }
        return output


DATASET_CONVERTERS = {
    "alpaca": AlpacaDatasetConverter,
    "sharegpt": SharegptDatasetConverter,
}


def get_dataset_converter(
    name: str, dataset_attr: DatasetAttr, data_args: DataArguments
) -> DatasetConverter:
    """Get a dataset converter by name.

    Args:
        name: Name of the converter to retrieve.
        dataset_attr: Dataset attributes for the converter.
        data_args: Data arguments for processing.

    Returns:
        An instance of the requested dataset converter.

    Raises:
        ValueError: If the requested converter name is not found.
    """
    if name not in DATASET_CONVERTERS:
        raise ValueError(f"Dataset converter {name} not found.")

    return DATASET_CONVERTERS[name](dataset_attr, data_args)


def align_dataset(
    dataset: Dataset | IterableDataset,
    dataset_attr: DatasetAttr,
    data_args: DataArguments,
    training_args: TrainingArguments,
) -> Dataset | IterableDataset:
    """Align the dataset to a specific format.

    Aligned dataset:
    _prompt: [{"role": "user", "content": "..."}] * (2T - 1)
    _response: [{"role": "assistant", "content": "..."}] * N (N > 1 for ranking dataset)
    _system: "..."
    _tools: "..."
    _images: []
    _videos: []
    _audios: []

    Args:
        dataset: The dataset to align.
        dataset_attr: Dataset attributes containing format information.
        data_args: Data arguments for processing.
        training_args: Training arguments for processing.

    Returns:
        The aligned dataset.
    """
    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not data_args.streaming:

        kwargs = {
            "num_proc": data_args.preprocessing_num_workers,
            "load_from_cache_file": not data_args.overwrite_cache
            or training_args.local_process_index != 0,
            "desc": "Converting format of dataset",
        }

    dataset_converter = get_dataset_converter(dataset_attr.formatting, dataset_attr, data_args)
    return dataset.map(
        dataset_converter,
        batched=False,
        remove_columns=column_names,
        **kwargs,
    )
