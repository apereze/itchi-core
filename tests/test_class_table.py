"""
Tests for the class table P6 x R24 x A72 (precipitation.compute_precipitation_class_table_hazard),
the Block 1 candidates, the 12 m/s wind-tail cut and the Block 1 experiment overlays.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from itchi.config import (
    KNOT_TO_MS,
    class_table_spec_from_config,
    itchi_v02_settings_from_config,
    load_config,
    load_experiment_config,
)
from itchi.pipeline import compute_itchi_snapshot_v02
from itchi.precipitation import (
    PRECIP_MATRIX_V02,
    PRECIP_TABLE_V02,
    PRECIP_TABLES_B1,
    build_precipitation_class_table,
    compute_percentile_coordinate,
    compute_precipitation_class_table_hazard,
    compute_precipitation_hazard_from_spec,
    compute_precipitation_hazard_ramp,
    compute_precipitation_matrix_hazard,
)
from itchi.wind_profiles import KNOT_TO_MS as KNOT_TO_MS_PROFILES

ROOT = Path(__file__).resolve().parents[1]
B1 = ROOT / "configs" / "experiments" / "b1"

Q6 = {"Q75": 8.0, "Q90": 18.0, "Q95": 28.0, "Q99": 50.0}
Q24 = {"Q75": 20.0, "Q90": 40.0, "Q95": 55.0}
Q72 = {"Q75": 30.0, "Q90": 60.0, "Q95": 80.0}
QESC = {"R24": Q24, "A72": Q72}


def _table(spec, p6, r24, a72, q6=Q6, qesc=QESC):
    return compute_precipitation_class_table_hazard(p6, q6, {"R24": r24, "A72": a72}, qesc, spec=spec)


def _random_percentiles(rng, shape, names, scale):
    steps = rng.gamma(2.0, scale, (len(names),) + shape)
    return dict(zip(names, np.cumsum(steps, axis=0), strict=True))


# ------------------------------------------------------------------ coordenada


def test_percentile_coordinate_nodes_and_nan() -> None:
    x = np.array([-1.0, 0.0, 4.0, 8.0, 13.0, 18.0, 28.0, 100.0, np.nan])
    u = compute_percentile_coordinate(x, Q6, ("Q75", "Q90", "Q95"))
    np.testing.assert_allclose(u[:-1], [0.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 3.0])
    assert np.isnan(u[-1])
    u_nan_q = compute_percentile_coordinate(np.array([10.0]), {**Q6, "Q90": np.nan}, ("Q75", "Q90"))
    assert np.isnan(u_nan_q[0])


# ------------------------------------------------------------------ tabla


def test_table_reproduces_v02_matrix() -> None:
    rng = np.random.default_rng(1)
    shape = (60, 50)
    q6 = _random_percentiles(rng, shape, ["Q75", "Q90", "Q95", "Q99"], 6.0)
    q24 = _random_percentiles(rng, shape, ["Q90", "Q95"], 15.0)
    q72 = _random_percentiles(rng, shape, ["Q75"], 25.0)
    p6 = rng.gamma(0.8, 15.0, shape)
    r24 = rng.gamma(0.8, 40.0, shape)
    a72 = rng.gamma(0.8, 40.0, shape)
    r24[:5] = np.nan                       # sin R24: sin escalada en ambas formas
    a72[:, :5] = np.nan
    p6[-3:, -3:] = np.nan
    esc, qesc = {"R24": r24, "A72": a72}, {"R24": q24, "A72": q72}
    ref = compute_precipitation_matrix_hazard(p6, q6, esc, qesc)
    out = compute_precipitation_class_table_hazard(p6, q6, esc, qesc, spec=PRECIP_TABLE_V02)
    np.testing.assert_allclose(out, ref, atol=1e-12)
    assert np.isnan(out[-3:, -3:]).all()


def test_m0_is_the_p6_ramp_to_p95() -> None:
    p = np.linspace(0.0, 60.0, 601)
    out = _table(PRECIP_TABLES_B1["M0"], p, 500.0, 500.0)
    ref = compute_precipitation_hazard_ramp(p, Q6, {"Q75": 0.0, "Q90": 0.5, "Q95": 1.0})
    np.testing.assert_allclose(out, ref, atol=1e-12)


@pytest.mark.parametrize("name", ["M0", "M1", "M2"])
def test_b1_candidates_p95_is_maximum_and_p75_is_onset(name: str) -> None:
    spec = PRECIP_TABLES_B1[name]
    r24 = np.linspace(0.0, 120.0, 25)[:, None, None]
    a72 = np.linspace(0.0, 150.0, 31)[None, :, None]
    p_high = np.array([28.0, 35.0, 80.0])[None, None, :]       # >= P95
    p_low = np.array([0.0, 3.0, 7.9])[None, None, :]            # < P75
    assert (_table(spec, p_high, r24, a72) == 1.0).all()
    assert (_table(spec, p_low, r24, a72) == 0.0).all()


@pytest.mark.parametrize("name", ["M0", "M1", "M2"])
def test_b1_candidates_monotone_and_bounded(name: str) -> None:
    p = np.linspace(0.0, 40.0, 41)[:, None, None]
    r24 = np.linspace(0.0, 80.0, 33)[None, :, None]
    a72 = np.linspace(0.0, 100.0, 26)[None, None, :]
    h = _table(PRECIP_TABLES_B1[name], p, r24, a72)
    for axis in range(3):
        assert (np.diff(h, axis=axis) >= -1e-12).all()
    assert h.min() >= 0.0 and h.max() <= 1.0


def test_m1_escalation_values() -> None:
    spec = PRECIP_TABLES_B1["M1"]
    # P6 = P90: 0.5 sin escalada, 1 con R24 >= P95 y A72 >= P75
    assert _table(spec, 18.0, 55.0, 30.0) == pytest.approx(1.0)
    assert _table(spec, 18.0, 54.0, 30.0) < 1.0
    assert _table(spec, 18.0, 55.0, 0.0) == pytest.approx(0.5)
    assert _table(spec, 18.0, 40.0, 100.0) == pytest.approx(0.5)
    # escalada parcial: R24 a la mitad entre P90 y P95, A72 a la mitad de P75
    assert _table(spec, 18.0, 47.5, 15.0) == pytest.approx(0.5 + 0.5 * 0.25)


def test_m2_gate_and_maximum_of_durations() -> None:
    spec = PRECIP_TABLES_B1["M2"]
    assert _table(spec, 10.0, 55.0, 0.0) == pytest.approx(1.0)     # P6 >= P75, R24 >= P95
    assert _table(spec, 10.0, 40.0, 0.0) == pytest.approx(0.5)     # R24 = P90
    assert _table(spec, 10.0, 0.0, 60.0) == pytest.approx(0.5)     # A72 = P90
    assert _table(spec, 7.0, 500.0, 500.0) == pytest.approx(0.0)   # P6 < P75: compuerta cerrada


def test_missing_escalation_is_no_escalation() -> None:
    spec = PRECIP_TABLES_B1["M1"]
    assert _table(spec, 18.0, np.nan, 100.0) == pytest.approx(0.5)
    out = _table(spec, 18.0, 100.0, 100.0, qesc={"R24": {**Q24, "Q95": np.nan}, "A72": Q72})
    assert out == pytest.approx(0.5)


def test_invalid_tables_raise() -> None:
    axes = {"P6": ("Q75", "Q90"), "R24": ("Q95",)}
    with pytest.raises(ValueError):
        build_precipitation_class_table({"axes": axes, "table": [[0.0, 0.0], [0.5, 0.4], [1.0, 1.0]]})
    with pytest.raises(ValueError):
        build_precipitation_class_table({"axes": axes, "p6_values": (0.0, 0.5)})
    with pytest.raises(KeyError):
        build_precipitation_class_table(
            {"axes": axes, "p6_values": (0.0, 0.0, 1.0), "rules": ({"when": {"A72": "Q75"}, "p6_values": (0, 0, 1)},)}
        )
    with pytest.raises(ValueError):
        build_precipitation_class_table({"axes": {"R24": ("Q95",)}, "p6_values": (0.0, 1.0)})


def test_explicit_table_matches_rules() -> None:
    spec = PRECIP_TABLES_B1["M1"]
    table = build_precipitation_class_table(spec)
    explicit = {"kind": "class_table", "axes": spec["axes"], "table": table.tolist()}
    p = np.linspace(0.0, 40.0, 81)
    np.testing.assert_allclose(_table(explicit, p, 50.0, 20.0), _table(spec, p, 50.0, 20.0))


def test_dispatch_and_xarray() -> None:
    p = np.linspace(0.0, 40.0, 41)
    np.testing.assert_allclose(
        compute_precipitation_hazard_from_spec(p, Q6, {"R24": 50.0, "A72": 20.0}, QESC, PRECIP_MATRIX_V02),
        compute_precipitation_matrix_hazard(p, Q6, {"R24": 50.0, "A72": 20.0}, QESC),
    )
    da = xr.DataArray(p, dims="x")
    out = _table(PRECIP_TABLES_B1["M1"], da, xr.full_like(da, 50.0), 20.0)
    assert isinstance(out, xr.DataArray) and out.dims == ("x",)
    np.testing.assert_allclose(out.values, _table(PRECIP_TABLES_B1["M1"], p, 50.0, 20.0))


# ------------------------------------------------------------------ configuración y pipeline


def test_knot_constant_matches_wind_profiles() -> None:
    assert KNOT_TO_MS == KNOT_TO_MS_PROFILES


def test_b1_overlays() -> None:
    settings = {c: itchi_v02_settings_from_config(load_experiment_config(B1 / f"{c}.yaml"))
                for c in ("p0_w0", "p1_w0", "p0_w1", "p1_w1")}
    assert settings["p0_w0"] == itchi_v02_settings_from_config()
    for c in ("p1_w0", "p1_w1"):
        spec = settings[c]["precip_matrix"]
        assert spec["kind"] == "class_table"
        np.testing.assert_array_equal(
            build_precipitation_class_table(spec), build_precipitation_class_table(PRECIP_TABLES_B1["M1"])
        )
    for c in ("p0_w1", "p1_w1"):
        assert settings[c]["wind_tail_end_kt"] == pytest.approx(12.0 / KNOT_TO_MS)
    assert settings["p1_w0"]["wind_tail_end_kt"] is None
    assert settings["p0_w1"]["precip_matrix"] == settings["p0_w0"]["precip_matrix"]


def test_class_table_spec_from_yaml_block() -> None:
    block = load_config(B1 / "p1_w0.yaml")["precipitation"]["class_table"]
    spec = class_table_spec_from_config(block)
    assert spec["rules"][0]["when"] == {"R24": "Q95", "A72": "Q75"}


def test_pipeline_with_class_table_and_12ms_cut() -> None:
    lon, lat = np.meshgrid(np.arange(-110.0, -80.0, 0.25), np.arange(5.0, 35.0, 0.25))
    rng = np.random.default_rng(0)
    p6 = rng.gamma(0.7, 12.0, lon.shape)
    r24, a72 = p6 + rng.gamma(0.7, 25.0, lon.shape), rng.gamma(0.8, 30.0, lon.shape)
    kw = dict(quantiles_6h=Q6, quantiles_24h=Q24, quantiles_72h=Q72, lon=lon, lat=lat,
              center_lon=-95.0, center_lat=20.0, vmax_kt=95.0, rmw_km=30.0,
              r34_by_quadrant={"RNE": 120, "RSE": 100, "RSW": 60, "RNW": 80})
    settings = itchi_v02_settings_from_config(load_experiment_config(B1 / "p1_w1.yaml"))
    res = compute_itchi_snapshot_v02(p6, r24, a72, **kw, **settings, run_quality_control=True)
    ref = compute_itchi_snapshot_v02(p6, r24, a72, **kw)
    assert res["precipitation_hazard_source"] == "class_table"
    outside = ~res["M_direct"]
    assert (res["H_W"][outside & (res["wind_kt"] <= 12.0 / KNOT_TO_MS)] < 1e-12).all()
    assert (res["H_W"] <= ref["H_W"] + 1e-12).all()
    np.testing.assert_allclose(res["H_W"][res["M_direct"]], ref["H_W"][ref["M_direct"]])
    assert (res["H_P"][p6 >= Q6["Q95"]] == 1.0).all()
