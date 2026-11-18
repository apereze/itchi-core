"""
Tests for the precipitation hazard: node ramps, P6 x (R24, A72) matrix
(maximum at the 6-h P95, precipitation.PRECIP_MATRIX) and antecedent windows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from itchi.precipitation import (
    PRECIP_MATRIX,
    compute_antecedent_precipitation,
    compute_precipitation_hazard_ramp,
    compute_precipitation_matrix_hazard,
)

Q6 = {"Q75": 8.0, "Q90": 18.0, "Q95": 28.0}
Q24 = {"Q90": 40.0, "Q95": 55.0}
Q72 = {"Q75": 30.0, "Q90": 60.0, "Q95": 80.0}
QESC = {"R24": Q24, "A72": Q72}
BASE_NODES = {"Q75": 0.0, "Q90": 0.25, "Q95": 1.0}
ESC_NODES = {"Q75": 0.0, "Q90": 0.8, "Q95": 1.0}
P_NODES = np.array([0.0, 4.0, 8.0, 13.0, 18.0, 23.0, 28.0, 39.0, 50.0, 80.0])


def _matrix(p, r24, a72, **kw):
    return compute_precipitation_matrix_hazard(p, Q6, {"R24": r24, "A72": a72}, QESC, **kw)


# ------------------------------------------------------------------ definición


def test_matrix_definition() -> None:
    assert PRECIP_MATRIX["p6_nodes"] == ("Q75", "Q90", "Q95")       # máximo en P95, sin P99
    assert PRECIP_MATRIX["base"] == (0.0, 0.0, 0.25, 1.0)
    assert PRECIP_MATRIX["escalated"] == (0.0, 0.0, 0.8, 1.0)       # la escalada solo sube P90 -> 0.8
    assert PRECIP_MATRIX["escalation"] == {"R24": ("Q90", "Q95"), "A72": ("0", "Q75")}
    assert PRECIP_MATRIX["combine"] == "all"


# ------------------------------------------------------------------ rampas


def test_ramp_nodes() -> None:
    h = compute_precipitation_hazard_ramp(P_NODES, Q6, BASE_NODES)
    np.testing.assert_allclose(h, [0.0, 0.0, 0.0, 0.125, 0.25, 0.625, 1.0, 1.0, 1.0, 1.0])


def test_ramp_propagates_nan_and_rejects_decreasing_nodes() -> None:
    h = compute_precipitation_hazard_ramp(np.array([np.nan, 10.0]), Q6, BASE_NODES)
    assert np.isnan(h[0]) and np.isfinite(h[1])
    with pytest.raises(ValueError):
        compute_precipitation_hazard_ramp(P_NODES, Q6, {"Q75": 0.5, "Q90": 0.25})


# ------------------------------------------------------------------ matriz


@pytest.mark.parametrize(("r24", "a72"), [(0.0, 100.0), (40.0, 100.0), (100.0, 0.0), (np.nan, 100.0), (100.0, np.nan)])
def test_matrix_without_escalation_is_base_ramp(r24: float, a72: float) -> None:
    np.testing.assert_allclose(_matrix(P_NODES, r24, a72), compute_precipitation_hazard_ramp(P_NODES, Q6, BASE_NODES))


def test_matrix_full_escalation() -> None:
    h = _matrix(P_NODES, 60.0, 30.0)          # R24 >= Q95_24 and A72 >= Q75_72
    np.testing.assert_allclose(h, compute_precipitation_hazard_ramp(P_NODES, Q6, ESC_NODES))
    np.testing.assert_allclose(h[[0, 2, 4, 6, 8]], [0.0, 0.0, 0.8, 1.0, 1.0])


def test_matrix_partial_escalation_is_product_of_weights() -> None:
    base = compute_precipitation_hazard_ramp(P_NODES, Q6, BASE_NODES)
    esc = compute_precipitation_hazard_ramp(P_NODES, Q6, ESC_NODES)
    h = _matrix(P_NODES, 47.5, 15.0)          # f_R24 = 0.5, f_A72 = 0.5 -> f = 0.25
    np.testing.assert_allclose(h, 0.75 * base + 0.25 * esc)
    h_any = _matrix(P_NODES, 47.5, 15.0, matrix={**PRECIP_MATRIX, "combine": "any"})
    np.testing.assert_allclose(h_any, 0.5 * base + 0.5 * esc)


def test_matrix_maximum_at_p95() -> None:
    p6 = np.linspace(0.0, 60.0, 601)
    for esc, h90 in (({"R24": 0.0, "A72": 0.0}, 0.25), ({"R24": 60.0, "A72": 40.0}, 0.8)):
        h = _matrix(p6, esc["R24"], esc["A72"])
        np.testing.assert_allclose(np.interp([8.0, 18.0, 28.0, 50.0], p6, h), [0.0, h90, 1.0, 1.0], atol=1e-9)


def test_matrix_is_monotone_and_bounded() -> None:
    p = np.linspace(0.0, 80.0, 81)[None, None, :]
    r24 = np.linspace(0.0, 80.0, 41)[None, :, None]
    a72 = np.linspace(0.0, 60.0, 31)[:, None, None]
    h = _matrix(p, r24, a72)
    for axis in range(3):
        assert (np.diff(h, axis=axis) >= -1e-12).all()
    assert h.min() >= 0.0 and h.max() <= 1.0


def test_matrix_xarray() -> None:
    p = xr.DataArray(P_NODES, dims="x")
    q6 = {k: xr.full_like(p, v) for k, v in Q6.items()}
    h = compute_precipitation_matrix_hazard(p, q6, {"R24": xr.full_like(p, 60.0), "A72": xr.full_like(p, 30.0)}, QESC)
    assert isinstance(h, xr.DataArray)
    np.testing.assert_allclose(h.values, _matrix(P_NODES, 60.0, 30.0))


# ------------------------------------------------------------------ ventanas


def test_antecedent_precipitation_window_and_gap() -> None:
    t = pd.date_range("2020-11-30 00:00", periods=4, freq="6h").append(
        pd.date_range("2021-05-01 06:00", periods=20, freq="6h"))
    p = xr.DataArray(np.ones(t.size), dims="time", coords={"time": t})
    a = compute_antecedent_precipitation(p, hours=72)
    r = compute_antecedent_precipitation(p, hours=24, exclude_current=False)
    # 2021: R24 valid from the 4th step; A72 needs 12 previous steps
    assert np.isnan(r.values[4:7]).all() and r.values[7] == 4.0
    assert np.isnan(a.values[4:16]).all() and a.values[16] == 12.0
