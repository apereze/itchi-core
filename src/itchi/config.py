"""
Configuration utilities for ITCHI.

This module provides a small and explicit YAML reader for project
configuration files such as configs/default.yaml.
"""

from __future__ import annotations

from collections.abc import Mapping
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


def itchi_v02_settings_from_config(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Keyword arguments of ``pipeline.compute_itchi_snapshot_v02`` from a config.

    Usage::

        settings = itchi_v02_settings_from_config(load_default_config())
        result = compute_itchi_snapshot_v02(..., **settings)

    With configs/default.yaml the result equals the keyword defaults of
    compute_itchi_snapshot_v02 (checked in tests/test_v02_snapshot.py), so an
    experiment is defined by editing a copy of the YAML file.

    Parameters
    ----------
    config : Mapping or None, default=None
        Parsed configuration; None loads configs/default.yaml.

    Returns
    -------
    dict[str, Any]
        precip_matrix, wind_tail_h34, wind_tail_end_kt, w06_attribution_mode,
        w06_direct_fallback, r_w06_kwargs, precipitation_nan_value,
        radius_fill_strategy and the ITCHI weights.
    """
    cfg = load_default_config() if config is None else config

    matrix_cfg = cfg["precipitation"]["matrix_v02"]
    wind_cfg = cfg["wind"]
    radii_cfg = cfg["parametric_radii"]
    r_w06_cfg = radii_cfg["r_w06"]
    weights = cfg["itchi"]

    tail_end = wind_cfg.get("tail_end_wind_kt")
    nan_value = matrix_cfg.get("nan_hazard_value", 0.0)

    return {
        "precip_matrix": {
            "p6_nodes": tuple(str(v) for v in matrix_cfg["p6_nodes"]),
            "base": tuple(float(v) for v in matrix_cfg["base"]),
            "escalated": tuple(float(v) for v in matrix_cfg["escalated"]),
            "escalation": {
                str(name): tuple(str(v) for v in bounds)
                for name, bounds in matrix_cfg["escalation"].items()
            },
            "combine": str(matrix_cfg.get("combine", "all")),
        },
        "wind_tail_h34": float(wind_cfg["tail_h34"]),
        "wind_tail_end_kt": None if tail_end is None else float(tail_end),
        "w06_attribution_mode": str(radii_cfg["attribution_radius_mode"]),
        "w06_direct_fallback": bool(radii_cfg.get("use_as_direct_fallback", True)),
        "r_w06_kwargs": {
            "rmax_variant": str(r_w06_cfg["rmax_variant"]),
            "use_observed_rmw": bool(r_w06_cfg["use_observed_rmw"]),
            "calibrate_to_r34": bool(r_w06_cfg["calibrate_to_r34"]),
            "outer_threshold_ms": float(r_w06_cfg["outer_threshold_ms"]),
        },
        "precipitation_nan_value": None if nan_value is None else float(nan_value),
        "radius_fill_strategy": str(cfg.get("geometry", {}).get("radius_fill_strategy", "mean_available")),
        "alpha_wind": float(weights["alpha_wind"]),
        "beta_precip_direct": float(weights["beta_precip_direct"]),
        "lambda_direct": float(weights["lambda_direct"]),
        "mu_indirect": float(weights["mu_indirect"]),
    }
