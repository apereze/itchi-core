"""
Tests for ITCHI v0.2: P6 x (R24, A72) precipitation matrix and W06 wind with outer tail.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from itchi.components import compute_hazard_components, compute_indirect_component
from itchi.parametric_radii import compute_w06_wind_hazard_with_tail
from itchi.pipeline import compute_itchi_snapshot_from_grid
from itchi.precipitation import (
    RAMP_R4_NODES,
    compute_antecedent_precipitation,
    compute_precipitation_hazard,
    compute_precipitation_hazard_ramp,
    compute_precipitation_matrix_hazard,
)
from itchi.wind import normalize_wind_hazard_with_tail

Q6 = {"Q75": 8.0, "Q90": 18.0, "Q95": 28.0, "Q99": 50.0}
Q24 = {"Q90": 40.0, "Q95": 55.0}
Q72 = {"Q75": 30.0, "Q90": 60.0, "Q95": 80.0}
QESC = {"R24": Q24, "A72": Q72}
ESC_NODES = {"Q75": 0.0, "Q90": 0.25, "Q95": 0.8, "Q99": 1.0}


def _matrix(p, r24, a72, **kw):
    return compute_precipitation_matrix_hazard(p, Q6, {"R24": r24, "A72": a72}, QESC, **kw)
P_NODES = np.array([0.0, 4.0, 8.0, 13.0, 18.0, 23.0, 28.0, 39.0, 50.0, 80.0])


# ------------------------------------------------------------------ rampas


def test_ramp_reproduces_v01_hazard() -> None:
    p = np.linspace(0.0, 80.0, 801)
    r3 = compute_precipitation_hazard_ramp(p, Q6, {"Q90": 0.0, "Q95": 0.5, "Q99": 1.0})
    ref = compute_precipitation_hazard(p, Q6["Q90"], Q6["Q95"], Q6["Q99"])
    np.testing.assert_allclose(r3, ref, atol=1e-12)


def test_ramp_r4_nodes() -> None:
    h = compute_precipitation_hazard_ramp(P_NODES, Q6, RAMP_R4_NODES)
    expected = [0.0, 0.0, 0.0, 0.125, 0.25, 0.375, 0.5, 0.75, 1.0, 1.0]
    np.testing.assert_allclose(h, expected)


def test_ramp_propagates_nan_and_rejects_decreasing_nodes() -> None:
    h = compute_precipitation_hazard_ramp(np.array([np.nan, 10.0]), Q6, RAMP_R4_NODES)
    assert np.isnan(h[0]) and np.isfinite(h[1])
    with pytest.raises(ValueError):
        compute_precipitation_hazard_ramp(P_NODES, Q6, {"Q75": 0.5, "Q90": 0.25})


# ------------------------------------------------------------------ matriz


@pytest.mark.parametrize(("r24", "a72"), [(0.0, 100.0), (40.0, 100.0), (100.0, 0.0), (np.nan, 100.0), (100.0, np.nan)])
def test_matrix_without_escalation_is_r4(r24: float, a72: float) -> None:
    np.testing.assert_allclose(
        _matrix(P_NODES, r24, a72), compute_precipitation_hazard_ramp(P_NODES, Q6, RAMP_R4_NODES)
    )


def test_matrix_full_escalation() -> None:
    h = _matrix(P_NODES, 60.0, 30.0)          # R24 >= Q95_24 and A72 >= Q75_72
    np.testing.assert_allclose(h, compute_precipitation_hazard_ramp(P_NODES, Q6, ESC_NODES))
    np.testing.assert_allclose(h[[0, 2, 4, 6, 8]], [0.0, 0.0, 0.25, 0.8, 1.0])


def test_matrix_partial_escalation_is_product_of_weights() -> None:
    r4 = compute_precipitation_hazard_ramp(P_NODES, Q6, RAMP_R4_NODES)
    esc = compute_precipitation_hazard_ramp(P_NODES, Q6, ESC_NODES)
    h = _matrix(P_NODES, 47.5, 15.0)          # f_R24 = 0.5, f_A72 = 0.5 -> f = 0.25
    np.testing.assert_allclose(h, 0.75 * r4 + 0.25 * esc)
    from itchi.precipitation import PRECIP_MATRIX_V02
    h_any = _matrix(P_NODES, 47.5, 15.0, matrix={**PRECIP_MATRIX_V02, "combine": "any"})
    np.testing.assert_allclose(h_any, 0.5 * r4 + 0.5 * esc)


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
    h = compute_precipitation_matrix_hazard(
        p, q6, {"R24": xr.full_like(p, 60.0), "A72": xr.full_like(p, 30.0)}, QESC
    )
    assert isinstance(h, xr.DataArray)
    np.testing.assert_allclose(h.values, _matrix(P_NODES, 60.0, 30.0))


def test_antecedent_precipitation_window_and_gap() -> None:
    t = pd.date_range("2020-11-30 00:00", periods=4, freq="6h").append(
        pd.date_range("2021-05-01 06:00", periods=20, freq="6h"))
    p = xr.DataArray(np.ones(t.size), dims="time", coords={"time": t})
    a = compute_antecedent_precipitation(p, hours=72)
    r = compute_antecedent_precipitation(p, hours=24, exclude_current=False)
    # 2021: R24 valid from the 4th step; A72 needs 12 previous steps
    assert np.isnan(r.values[4:7]).all() and r.values[7] == 4.0
    assert np.isnan(a.values[4:16]).all() and a.values[16] == 12.0


# ------------------------------------------------------------------ viento


def test_wind_tail_normalization() -> None:
    h = normalize_wind_hazard_with_tail(
        np.array([90.0, 34.0, 34.0, 10.0, 2.0]), 90.0,
        np.array([True, True, False, False, False]), 2.0, h34=0.2,
    )
    np.testing.assert_allclose(h, [1.0, 0.2, 0.2, 0.2 * 8.0 / 32.0, 0.0])


def test_w06_tail_continuity_and_zero_at_attribution() -> None:
    r = np.linspace(0.0, 700.0, 7001)
    for r34_q in (150.0, 250.0):
        out = compute_w06_wind_hazard_with_tail(
            r, r34_q, 600.0, vmax_kt=90.0, lat=20.0, r34_mean_km=200.0, h34=0.2
        )
        h, v = out["H_W"], out["wind_kt"]
        i34 = np.searchsorted(r, r34_q)
        assert out["x1_calibrated"]
        assert abs(v[i34] - 34.0) < 0.2
        assert abs(h[i34] - 0.2) < 0.01 and abs(h[i34 + 1] - 0.2) < 0.01
        assert (h[np.searchsorted(r, 600.0):] < 1e-6).all()


def test_w06_tail_tropical_depression_stays_below_h34() -> None:
    r = np.linspace(0.0, 600.0, 601)
    out = compute_w06_wind_hazard_with_tail(r, np.nan, 500.0, vmax_kt=30.0, lat=18.0, h34=0.2)
    assert not out["x1_calibrated"]
    assert 0.0 < out["H_W"].max() < 0.2


# ------------------------------------------------------------------ componentes y pipeline


def test_indirect_component_union() -> None:
    h = compute_indirect_component(np.array([0.0, 0.5, 0.5]), indirect_wind_hazard=np.array([0.2, 0.0, 0.2]))
    np.testing.assert_allclose(h, [0.2, 0.5, 0.6])
    np.testing.assert_allclose(compute_indirect_component(np.array([0.3])), [0.3])


def test_components_v01_unchanged_without_indirect_wind() -> None:
    hp = np.array([0.2, 0.4, 0.6])
    hw = np.array([0.5, 0.5, 0.5])
    d = np.array([True, False, False])
    i = np.array([False, True, False])
    v01 = compute_hazard_components(hp, hw, d, i)
    assert set(v01) == {"H_Pdir", "H_Pind", "H_W", "H_dir", "H_ind"}
    v02 = compute_hazard_components(hp, hw, d, i, wind_hazard_indirect=hw)
    np.testing.assert_allclose(v02["H_W"], [0.5, 0.5, 0.0])
    np.testing.assert_allclose(v02["H_ind"], [0.0, 1 - 0.5 * 0.6, 0.0])


def _grid_case() -> dict:
    lon, lat = np.meshgrid(np.arange(-100.0, -90.0, 0.1), np.arange(15.0, 25.0, 0.1))
    return dict(
        lon=lon, lat=lat, center_lon=-95.0, center_lat=20.0,
        r34_by_quadrant={"RNE": 120, "RSE": 100, "RSW": 60, "RNW": 80}, r34_unit="nm",
        rocloud_by_quadrant={"RNE": 650, "RSE": 600, "RSW": 450, "RNW": 500}, rocloud_unit="km",
        vmax_kt=95.0, rmw_km=30.0, run_quality_control=True,
    )


def test_pipeline_precomputed_hazard_matches_v01() -> None:
    kw = _grid_case()
    p = np.random.default_rng(0).gamma(0.7, 12.0, kw["lon"].shape)
    v01 = compute_itchi_snapshot_from_grid(p, Q6["Q90"], Q6["Q95"], Q6["Q99"], **kw)
    hp = compute_precipitation_hazard(p, Q6["Q90"], Q6["Q95"], Q6["Q99"])
    same = compute_itchi_snapshot_from_grid(p, None, None, None, precipitation_hazard=hp, **kw)
    np.testing.assert_allclose(v01["ITCHI"], same["ITCHI"])


def test_pipeline_v02_w06_tail() -> None:
    kw = _grid_case()
    p = np.random.default_rng(0).gamma(0.7, 12.0, kw["lon"].shape)
    hp = _matrix(p, np.full(p.shape, 100.0), np.full(p.shape, 100.0))
    res = compute_itchi_snapshot_from_grid(
        p, None, None, None, precipitation_hazard=hp, wind_model="w06_tail", **kw
    )
    ind = res["M_indirect"]
    assert res["wind_hazard_source"] == "w06_tail" and res["w06_x1_calibrated"]
    assert (res["H_Wind"][ind] > 0.0).mean() > 0.9
    assert res["H_Wind"].max() <= 0.2 + 1e-9
    np.testing.assert_allclose(res["H_ind"], 1 - (1 - res["H_Wind"]) * (1 - res["H_Pind"]))
    assert (res["ITCHI"][res["M_exterior"]] == 0.0).all()


def test_pipeline_rejects_unknown_wind_model() -> None:
    kw = _grid_case()
    with pytest.raises(ValueError):
        compute_itchi_snapshot_from_grid(np.zeros(kw["lon"].shape), 1.0, 2.0, 3.0, wind_model="x", **kw)
