"""Module for formatting text templates with placeholders."""

from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
import re

from typing_extensions import override

SLOTS = list[str | set[str] | dict[str, str]]


@dataclass
class Formatter(ABC):
    """Abstract base class for formatters.

    Attributes:
        slots: A list of slots to form.
    """

    slots: SLOTS = field(default_factory=list, metadata={"help": "List of slots to form."})

    @property
    def has_placeholders(self) -> bool:
        """Check if any slot contains placeholders.

        Returns:
            True if any slot contains placeholders, False otherwise.
        """
        return any(
            isinstance(s, str) and re.search(r"\{\{[a-zA-Z_][a-zA-Z0-9_]*}}", s) for s in self.slots
        )

    @abstractmethod
    def apply(self, **kwargs) -> SLOTS:
        """Forms a list of slots according to the inputs to encode.

        Args:
            **kwargs: Keyword arguments to replace placeholders.

        Returns:
            A list of formatted slots.
        """
        ...


@dataclass
class EmptyFormatter(Formatter):
    """Formatter for empty templates without placeholders."""

    def __post_init__(self) -> None:
        """Initialize the empty formatter and validate no placeholders exist.

        Raises:
            ValueError: If the formatter contains placeholders.
        """
        if self.has_placeholders:
            raise ValueError("Empty formatter should not contain any placeholder.")

    @override
    def apply(self, **kwargs) -> SLOTS:
        """Apply the empty formatter.

        Args:
            **kwargs: Keyword arguments (ignored for empty formatter).

        Returns:
            The original slots without modification.
        """
        return self.slots


@dataclass
class StringFormatter(Formatter):
    """Formatter for string templates with placeholders."""

    def __post_init__(self) -> None:
        """Initialize the string formatter and validate placeholders exist.

        Raises:
            ValueError: If the formatter does not contain any placeholders.
        """
        if not self.has_placeholders:
            raise ValueError("A placeholder is required in the string formatter.")

    @override
    def apply(self, **kwargs) -> SLOTS:
        """Apply the string formatter by replacing placeholders.

        Args:
            **kwargs: Keyword arguments to replace placeholders.

        Returns:
            A list of formatted slots with placeholders replaced.

        Raises:
            RuntimeError: If a value is not a string.
            RuntimeError: If slot is not a string, set or dict.
        """
        elements = []
        for slot in self.slots:
            if isinstance(slot, str):
                for name, value in kwargs.items():
                    if not isinstance(value, str):
                        raise RuntimeError(f"Expected a string, got {value}")

                    slot = slot.replace("{{" + name + "}}", value, 1)
                elements.append(slot)
            elif isinstance(slot, (dict, set)):
                elements.append(slot)
            else:
                raise RuntimeError(
                    f"Input must be string, set[str] or dict[str, str], got {type(slot)}."
                )

        return elements
