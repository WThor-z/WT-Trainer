"""Dataset processor utilities module.

This module provides abstract base classes and utility functions for processing datasets.
It includes implementations for data preprocessing, knapsack algorithms for packing optimization,
and sequence length inference.
"""

from abc import ABC
from abc import abstractmethod
import bisect
from dataclasses import dataclass
from typing import Any

from transformers import PreTrainedTokenizer  # noqa: F401
from transformers import ProcessorMixin  # noqa: F401

from wt_trainer.args import DataArguments  # noqa: F401
from ..template import Template  # noqa: F401


@dataclass
class DatasetProcessor(ABC):
    """A class for data processors.

    Abstract base class for dataset processors that defines the interface for
    preprocessing datasets and printing data examples.

    Attributes:
        template: Template for formatting the data.
        tokenizer: Tokenizer for processing the data.
        data_args: Data arguments for processing.
    """

    template: Template
    tokenizer: PreTrainedTokenizer
    data_args: DataArguments

    @abstractmethod
    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        """Build model inputs from the examples.

        Args:
            examples: Dictionary of examples to preprocess.

        Returns:
            Dictionary of preprocessed model inputs.
        """
        ...

    @abstractmethod
    def print_data_example(self, example: dict[str, list[int]]) -> None:
        """Print a data example to stdout.

        Args:
            example: Dictionary containing the example data to print.
        """
        ...


def search_for_fit(numbers: list[int], capacity: int) -> int:
    """Find the index of largest number that fits into the knapsack with the given capacity.

    Args:
        numbers: List of numbers to search through.
        capacity: Capacity of the knapsack.

    Returns:
        Index of the largest number that fits, or -1 if none fit.
    """
    index = bisect.bisect(numbers, capacity)
    return -1 if index == 0 else (index - 1)


def greedy_knapsack(numbers: list[int], capacity: int) -> list[list[int]]:
    """Implement efficient greedy algorithm with binary search for the knapsack problem.

    Args:
        numbers: List of numbers to pack.
        capacity: Capacity of each knapsack.

    Returns:
        List of knapsacks, each containing a list of packed numbers.
    """
    numbers.sort()  # sort numbers in ascending order for binary search
    knapsacks = []

    while numbers:
        current_knapsack = []
        remaining_capacity = capacity

        while True:
            index = search_for_fit(numbers, remaining_capacity)
            if index == -1:
                break  # no more numbers fit in this knapsack

            remaining_capacity -= numbers[index]  # update the remaining capacity
            current_knapsack.append(numbers.pop(index))  # add the number to knapsack

        knapsacks.append(current_knapsack)

    return knapsacks


def infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    """Compute the real sequence length after truncation by the cutoff_len.

    Args:
        source_len: Length of the source sequence.
        target_len: Length of the target sequence.
        cutoff_len: Maximum allowed total sequence length.

    Returns:
        Tuple of (new_source_len, new_target_len) after truncation.
    """
    if target_len * 2 < cutoff_len:  # truncate source
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:  # truncate target
        max_target_len = cutoff_len - source_len
    else:  # truncate both
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))

    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len
