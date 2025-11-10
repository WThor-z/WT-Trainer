"""Module for reading and processing training arguments from configuration files.

This module provides utilities for reading training arguments from various sources
and converting them into the appropriate data structures for model training.
"""

from pathlib import Path
from typing import Any, Dict

from omegaconf import OmegaConf


def read_train_args(config_file_path: str | None = None) -> Dict[str, Any]:
    """Read and parse training arguments from configuration file.

    Args:
        config_file_path: Path to configuration file. Can be either a string path
            to a YAML/JSON file or None. If None, will use default configuration.

    Returns:
        Dictionary containing the training configuration.

    Raises:
        FileNotFoundError: If the configuration file path is invalid.
        ValueError: If the configuration file format is not supported.
    """
    if config_file_path is None:
        return {}

    config_path = Path(config_file_path).absolute()
    # Load configuration file
    json_config = OmegaConf.load(config_path)
    train_config = OmegaConf.to_container(json_config)

    if not isinstance(train_config, dict):
        raise ValueError("Configuration file must contain a dictionary")
    return train_config
