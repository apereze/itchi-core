"""
Configuration utilities for ITCHI.

This module provides a small and explicit YAML reader for project
configuration files such as configs/default.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load an ITCHI YAML configuration file.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the YAML configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration file is empty or invalid.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Configuration path is not a file: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if not isinstance(config, dict):
        raise ValueError(f"Configuration file must define a dictionary: {path}")

    return config


def get_project_root() -> Path:
    """
    Return the project root directory.

    This assumes the standard repository structure:

    itchi-core/
    ├── configs/
    ├── src/
    │   └── itchi/
    └── pyproject.toml

    Returns
    -------
    pathlib.Path
        Repository root path.
    """
    return Path(__file__).resolve().parents[2]


def load_default_config() -> dict[str, Any]:
    """
    Load the default ITCHI configuration file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration from configs/default.yaml.
    """
    project_root = get_project_root()
    config_path = project_root / "configs" / "default.yaml"

    return load_config(config_path)