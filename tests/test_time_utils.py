"""
Tests for generic temporal utility functions.
"""

from __future__ import annotations

import pandas as pd
import pytest

from itchi.time_utils import each6h, filter_synoptic_dataframe, normalize_hour_value


def test_normalize_hour_value_accepts_integer_hours() -> None:
    """
    Test normalization of integer-like hour values.
    """
    assert normalize_hour_value(0) == 0
    assert normalize_hour_value(6) == 6
    assert normalize_hour_value(18) == 18


def test_normalize_hour_value_accepts_hhmm_values() -> None:
    """
    Test normalization of compact HHMM values.
    """
    assert normalize_hour_value(0) == 0
    assert normalize_hour_value(600) == 6
    assert normalize_hour_value(1200) == 12
    assert normalize_hour_value(1800) == 18


def test_normalize_hour_value_rejects_invalid_values() -> None:
    """
    Test invalid hour handling.
    """
    with pytest.raises(ValueError):
        normalize_hour_value(2400)

    with pytest.raises(ValueError):
        normalize_hour_value(1261)


def test_filter_synoptic_dataframe_keeps_00_06_12_18() -> None:
    """
    Test generic DataFrame filtering to synoptic hours.
    """
    df = pd.DataFrame(
        {
            "hh": [0, 3, 6, 9, 12, 15, 18, 21],
            "value": list(range(8)),
        }
    )

    result = filter_synoptic_dataframe(df)

    assert result["hh"].tolist() == [0, 6, 12, 18]
    assert result["value"].tolist() == [0, 2, 4, 6]
    assert result.index.tolist() == [0, 1, 2, 3]


def test_filter_synoptic_dataframe_accepts_hhmm() -> None:
    """
    Test synoptic filtering with compact HHMM values.
    """
    df = pd.DataFrame(
        {
            "hh": [0, 300, 600, 900, 1200, 1500, 1800, 2100],
            "value": list(range(8)),
        }
    )

    result = filter_synoptic_dataframe(df)

    assert result["hh"].tolist() == [0, 600, 1200, 1800]


def test_filter_synoptic_dataframe_missing_column() -> None:
    """
    Test missing hour-column error.
    """
    df = pd.DataFrame({"time": [0, 6, 12, 18]})

    with pytest.raises(KeyError):
        filter_synoptic_dataframe(df)


def test_each6h_alias() -> None:
    """
    Test backward-compatible each6h alias.
    """
    df = pd.DataFrame({"hh": [0, 1, 6, 18], "value": [10, 20, 30, 40]})

    result = each6h(df)

    assert result["value"].tolist() == [10, 30, 40]
