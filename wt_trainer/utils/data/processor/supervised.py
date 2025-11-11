"""Supervised dataset processors module.

This module provides dataset processors for supervised training scenarios.
It includes implementations for regular and packed supervised dataset processing,
handling data encoding, preprocessing, and formatting for model training.
"""

from collections import defaultdict
from dataclasses import dataclass
import logging
from typing import Any

from .processor_utils import DatasetProcessor
from .processor_utils import greedy_knapsack
from .processor_utils import infer_seqlen

IGNORE_INDEX = -100

logger = logging.getLogger(__name__)


@dataclass
class SupervisedDatasetProcessor(DatasetProcessor):
    """Processor for supervised datasets.

    This processor handles the encoding and preprocessing of supervised datasets
    for training language models. It supports multi-turn conversations and
    various masking strategies.
    """

    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: str | None,
    ) -> tuple[list[int], list[int]]:
        """Encode a single data example into input IDs and labels.

        Args:
            prompt: List of prompt messages.
            response: List of response messages.
            system: System message, if any.

        Returns:
            Tuple of (input_ids, labels) for the encoded example.
        """
        messages = prompt + response
        input_ids, labels = [], []
        encoded_pairs = self.template.encode_multiturn(self.tokenizer, messages, system)
        total_length = len(input_ids) + (1 if self.template.efficient_eos else 0)
        if self.data_args.mask_history:
            encoded_pairs = encoded_pairs[::-1]  # high priority for last turns

        for turn_idx, (source_ids, target_ids) in enumerate(encoded_pairs):
            if total_length >= self.data_args.cutoff_len:
                break

            source_len, target_len = infer_seqlen(
                len(source_ids), len(target_ids), self.data_args.cutoff_len - total_length
            )
            source_ids = source_ids[:source_len]
            target_ids = target_ids[:target_len]
            total_length += source_len + target_len

            if self.data_args.train_on_prompt:
                source_label = source_ids
            elif self.template.efficient_eos:
                source_label = [self.tokenizer.eos_token_id] + [IGNORE_INDEX] * (source_len - 1)
            else:
                source_label = [IGNORE_INDEX] * source_len

            if self.data_args.mask_history and turn_idx != 0:  # train on the last turn only
                target_label = [IGNORE_INDEX] * target_len
            else:
                target_label = target_ids

            if self.data_args.mask_history:  # reversed sequences
                input_ids = source_ids + target_ids + input_ids
                labels = source_label + target_label + labels
            else:
                input_ids += source_ids + target_ids
                labels += source_label + target_label

        if self.template.efficient_eos:
            input_ids += [self.tokenizer.eos_token_id]
            labels += [self.tokenizer.eos_token_id]

        return input_ids, labels

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Preprocess the dataset by encoding examples into model inputs.

        Builds inputs with format `<bos> X Y <eos>` and labels with format
        `<ignore> ... <ignore> Y <eos>`. For multiturn examples, we only
        mask the prompt part in each prompt-response pair.

        Args:
            examples: Dictionary of examples to preprocess.

        Returns:
            Dictionary of preprocessed model inputs.
        """
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                prompt_and_response = examples["_prompt"][i] + examples["_response"][i]
                logger.warning(f"Dropped invalid example: {prompt_and_response}")
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
            )
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids))
            model_inputs["labels"].append(labels)

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """Print a data example to stdout.

        Args:
            example: Dictionary containing the example data to print.
        """
        valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
        print(f"input_ids:\n{example['input_ids']}")
        inputs = self.tokenizer.decode(example["input_ids"], skip_special_tokens=False)
        print(f"inputs:\n{inputs}")
        print(f"label_ids:\n{example['labels']}")
        labels = self.tokenizer.decode(valid_labels, skip_special_tokens=False)
        print(f"labels:\n{labels}")


@dataclass
class PackedSupervisedDatasetProcessor(SupervisedDatasetProcessor):
    """Processor for packed supervised datasets.

    This processor handles the encoding and preprocessing of packed supervised datasets
    for training language models. It uses a knapsack algorithm to efficiently pack
    multiple examples into fixed-length sequences.
    """

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Preprocess the dataset by packing examples into fixed-length sequences.

        TODO: use `position_ids` to achieve packing
        Builds inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`

        Args:
            examples: Dictionary of examples to preprocess.

        Returns:
            Dictionary of preprocessed model inputs with packed sequences.
        """
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        valid_num = 0
        batch_input_ids, batch_labels = [], []
        lengths = []
        length2indexes = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) != 1:
                prompt_and_response = examples["_prompt"][i] + examples["_response"][i]
                logger.warning(f"Dropped invalid example: {prompt_and_response}")
                continue

            input_ids, labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
            )
            length = len(input_ids)
            if length > self.data_args.cutoff_len:
                logger.warning(
                    f"Dropped lengthy example with length {length} > {self.data_args.cutoff_len}."
                )
            else:
                lengths.append(length)
                length2indexes[length].append(valid_num)
                batch_input_ids.append(input_ids)
                batch_labels.append(labels)
                valid_num += 1

        model_inputs = defaultdict(list)
        knapsacks = greedy_knapsack(lengths, self.data_args.cutoff_len)

        # Process each knapsack
        for knapsack in knapsacks:
            # Initialize packed sequences
            packed_sequences = {
                "input_ids": [],
                "position_ids": [],
                "labels": [],
                "attention_masks": [],
            }

            for i, length in enumerate(knapsack):
                index = length2indexes[length].pop()
                packed_sequences["input_ids"] += batch_input_ids[index]
                packed_sequences["position_ids"] += list(
                    range(len(batch_input_ids[index]))
                )  # NOTE: pad_to_multiple_of ignore this
                packed_sequences["labels"] += batch_labels[index]

                if self.data_args.neat_packing:
                    packed_sequences["attention_masks"] += [i + 1] * len(
                        batch_input_ids[index]
                    )  # start from 1
                else:
                    packed_sequences["attention_masks"] += [1] * len(batch_input_ids[index])

            # Handle padding
            if (
                len(packed_sequences["input_ids"]) < self.data_args.cutoff_len + 1
            ):  # avoid flash_attn drops attn mask
                pad_length = self.data_args.cutoff_len - len(packed_sequences["input_ids"]) + 1
                packed_sequences["input_ids"] += [self.tokenizer.pad_token_id] * pad_length
                packed_sequences["position_ids"] += [0] * pad_length
                packed_sequences["labels"] += [IGNORE_INDEX] * pad_length
                if self.data_args.neat_packing:
                    packed_sequences["attention_masks"] += [0] * pad_length
                else:
                    packed_sequences["attention_masks"] += [
                        1
                    ] * pad_length  # more efficient flash_attn

            if len(packed_sequences["input_ids"]) != self.data_args.cutoff_len + 1:
                raise ValueError(
                    "The length of packed example should be identical to the cutoff length."
                )

            model_inputs["input_ids"].append(packed_sequences["input_ids"])
            model_inputs["attention_mask"].append(packed_sequences["attention_masks"])
            model_inputs["position_ids"].append(packed_sequences["position_ids"])
            model_inputs["labels"].append(packed_sequences["labels"])

        return model_inputs
