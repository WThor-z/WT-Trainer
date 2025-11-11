"""Dataset attributes module.

This module defines the DatasetAttr dataclass which contains attributes
for configuring dataset loading and processing.
"""

from dataclasses import dataclass
from typing import Any, Literal


@dataclass
class DatasetAttr:
    """Dataset attributes.

    Attributes:
        load_from: Source of the dataset.
        dataset_name: Name of the dataset.
        formatting: Format of the dataset, either "alpaca" or "sharegpt". Defaults to "alpaca".
        ranking: Whether the dataset is for ranking tasks. Defaults to False.
        subset: Subset of the dataset, if applicable. Defaults to None.
        split: Dataset split to use. Defaults to "train".
        folder: Folder containing the dataset, if applicable. Defaults to None.
        num_samples: Number of samples to use from the dataset. Defaults to None.
        system: Column name for system messages. Defaults to None.
        tools: Column name for tools information. Defaults to None.
        images: Column name for images. Defaults to None.
        videos: Column name for videos. Defaults to None.
        audios: Column name for audios. Defaults to None.
        chosen: Column name for chosen responses in DPO tasks. Defaults to None.
        rejected: Column name for rejected responses in DPO tasks. Defaults to None.
        kto_tag: Column name for KTO tags. Defaults to None.
        prompt: Column name for prompts in Alpaca format. Defaults to "instruction".
        query: Column name for queries in Alpaca format. Defaults to "input".
        response: Column name for responses in Alpaca format. Defaults to "output".
        history: Column name for history in Alpaca format. Defaults to None.
        messages: Column name for messages in ShareGPT format. Defaults to "conversations".
        role_tag: Tag name for roles in ShareGPT format. Defaults to "from".
        content_tag: Tag name for content in ShareGPT format. Defaults to "value".
        user_tag: Tag name for user roles in ShareGPT format. Defaults to "human".
        assistant_tag: Tag name for assistant roles in ShareGPT format. Defaults to "gpt".
        observation_tag: Tag name for observation roles in ShareGPT format. Defaults to "observation".
        function_tag: Tag name for function roles in ShareGPT format. Defaults to "function_call".
        system_tag: Tag name for system roles in ShareGPT format. Defaults to "system".
    """

    # basic configs
    load_from: str
    dataset_name: str
    formatting: Literal["alpaca", "sharegpt"] = "alpaca"
    ranking: bool = False
    # extra configs
    subset: str | None = None
    split: str = "train"
    folder: str | None = None
    num_samples: int | None = None
    # common columns
    system: str | None = None
    tools: str | None = None
    images: str | None = None
    videos: str | None = None
    audios: str | None = None
    # dpo columns
    chosen: str | None = None
    rejected: str | None = None
    kto_tag: str | None = None
    # alpaca columns
    prompt: str | None = "instruction"
    query: str | None = "input"
    response: str | None = "output"
    history: str | None = None
    # sharegpt columns
    messages: str | None = "conversations"
    # sharegpt tags
    role_tag: str | None = "from"
    content_tag: str | None = "value"
    user_tag: str | None = "human"
    assistant_tag: str | None = "gpt"
    observation_tag: str | None = "observation"
    function_tag: str | None = "function_call"
    system_tag: str | None = "system"

    def __repr__(self) -> str:
        """Return the dataset name as string representation.

        Returns:
            The name of the dataset.
        """
        return self.dataset_name

    def set_attr(self, key: str, obj: dict[str, Any], default: Any | None = None) -> None:
        """Set an attribute from a dictionary with a default value.

        Args:
            key: The attribute name to set.
            obj: The dictionary to get the value from.
            default: The default value if key is not found. Defaults to None.
        """
        setattr(self, key, obj.get(key, default))

    def join(self, attr: dict[str, Any]) -> None:
        """Join attributes from a dictionary.

        Args:
            attr: Dictionary containing attributes to join.
        """
        self.set_attr("formatting", attr, default="alpaca")
        self.set_attr("ranking", attr, default=False)
        self.set_attr("subset", attr)
        self.set_attr("split", attr, default="train")
        self.set_attr("folder", attr)
        self.set_attr("num_samples", attr)

        if "columns" in attr:
            column_names = ["prompt", "query", "response", "history", "messages", "system", "tools"]
            column_names += ["images", "videos", "audios", "chosen", "rejected", "kto_tag"]
            for column_name in column_names:
                self.set_attr(column_name, attr["columns"])

        if "tags" in attr:
            tag_names = ["role_tag", "content_tag"]
            tag_names += [
                "user_tag",
                "assistant_tag",
                "observation_tag",
                "function_tag",
                "system_tag",
            ]
            for tag in tag_names:
                self.set_attr(tag, attr["tags"])
