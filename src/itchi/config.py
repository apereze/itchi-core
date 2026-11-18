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

from itchi.constants import SAFFIR_SIMPSON_HAZARD_NODES

KNOT_TO_MS: float = 0.514444  # same value as itchi.wind_profiles.KNOT_TO_MS


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


def merge_config(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    """
    Recursive merge: values of ``overlay`` replace those of ``base``; nested
    mappings are merged key by key. Inputs are not modified.
    """
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_experiment_config(
    overlay_path: str | Path,
    base: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Configuration of an experiment: configs/default.yaml with an overlay.

    The overlay (a YAML file under configs/experiments/) lists only the keys
    that change; ``experiment`` (name, description) is kept for traceability.

    Parameters
    ----------
    overlay_path : str or pathlib.Path
        YAML file with the changed keys.
    base : Mapping or None, default=None
        Base configuration; None loads configs/default.yaml.

    Returns
    -------
    dict[str, Any]
        Merged configuration.
    """
    cfg = load_default_config() if base is None else base
    return merge_config(cfg, load_config(overlay_path))

def itchi_settings_from_config(
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Keyword arguments of ``pipeline.compute_itchi_snapshot`` from a config.

    Usage::

        settings = itchi_settings_from_config(load_experiment_config(overlay))
        result = compute_itchi_snapshot(..., **settings)

    With configs/default.yaml the result equals the keyword defaults of
    compute_itchi_snapshot (checked in tests/test_pipeline.py); an experiment
    is an overlay on configs/default.yaml (load_experiment_config).

    Parameters
    ----------
    config : Mapping or None, default=None
        Full configuration; None loads configs/default.yaml.

    Returns
    -------
    dict[str, Any]
        precip_matrix, precipitation_nan_value, wind_tail_end_kt,
        wind_absolute_nodes, w06_attribution_mode, w06_direct_fallback,
        r_w06_kwargs, radius_fill_strategy and the ITCHI weights.
    """
    cfg = load_default_config() if config is None else config

    matrix_cfg = cfg["precipitation"]["matrix"]
    wind_cfg = cfg["wind"]
    radii_cfg = cfg["parametric_radii"]
    r_w06_cfg = radii_cfg["r_w06"]
    weights = cfg["itchi"]

    tail_end_kt = wind_cfg.get("tail_end_wind_kt")
    tail_end_ms = wind_cfg.get("tail_end_wind_ms")
    if tail_end_kt is not None and tail_end_ms is not None:
        raise ValueError("Give wind.tail_end_wind_kt or wind.tail_end_wind_ms, not both.")
    if tail_end_ms is not None:
        tail_end_kt = float(tail_end_ms) / KNOT_TO_MS
    nan_value = matrix_cfg.get("nan_hazard_value", 0.0)

    precip_spec: dict[str, Any] = {
        "p6_nodes": tuple(str(v) for v in matrix_cfg["p6_nodes"]),
        "base": tuple(float(v) for v in matrix_cfg["base"]),
        "escalated": tuple(float(v) for v in matrix_cfg["escalated"]),
        "escalation": {
            str(name): tuple(str(v) for v in bounds)
            for name, bounds in matrix_cfg["escalation"].items()
        },
        "combine": str(matrix_cfg.get("combine", "all")),
    }

    return {
        "precip_matrix": precip_spec,
        "precipitation_nan_value": None if nan_value is None else float(nan_value),
        "wind_tail_end_kt": None if tail_end_kt is None else float(tail_end_kt),
        "wind_absolute_nodes": tuple(
            (float(v), float(h)) for v, h in wind_cfg.get("saffir_simpson_nodes", SAFFIR_SIMPSON_HAZARD_NODES)
        ),
        "w06_attribution_mode": str(radii_cfg["attribution_radius_mode"]),
        "w06_direct_fallback": bool(radii_cfg.get("use_as_direct_fallback", True)),
        "r_w06_kwargs": {
            "rmax_variant": str(r_w06_cfg["rmax_variant"]),
            "use_observed_rmw": bool(r_w06_cfg["use_observed_rmw"]),
            "calibrate_to_r34": bool(r_w06_cfg["calibrate_to_r34"]),
            "outer_threshold_ms": float(r_w06_cfg["outer_threshold_ms"]),
        },
        "radius_fill_strategy": str(cfg.get("geometry", {}).get("radius_fill_strategy", "mean_available")),
        "alpha_wind": float(weights["alpha_wind"]),
        "beta_precip_direct": float(weights["beta_precip_direct"]),
        "lambda_direct": float(weights["lambda_direct"]),
        "mu_indirect": float(weights["mu_indirect"]),
    }
