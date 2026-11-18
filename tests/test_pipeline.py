"""
Tests for pipeline.compute_itchi_snapshot (synthetic fields; no MSWEP or
IBTrACS) and its configuration (configs/default.yaml).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import xarray as xr

from itchi.climatology import quantile_dataarray_to_dict
from itchi.config import KNOT_TO_MS, itchi_settings_from_config, load_default_config, merge_config
from itchi.constants import SAFFIR_SIMPSON_HAZARD_NODES, TAIL_END_WIND_MS
from itchi.parametric_radii import (
    R_W06_DEFAULTS,
    compute_parametric_radii,
    compute_w06_reference_radii,
    compute_w06_wind_hazard_with_tail,
)
from itchi.pipeline import compute_itchi_snapshot
from itchi.precipitation import PRECIP_MATRIX, compute_precipitation_matrix_hazard
from itchi.radii import resolve_attribution_radius_with_w06
from itchi.wind_profiles import KNOT_TO_MS as KNOT_TO_MS_PROFILES

QUADS = ("RNE", "RSE", "RSW", "RNW")
Q6 = {"Q75": 8.0, "Q90": 18.0, "Q95": 28.0}
Q24 = {"Q90": 40.0, "Q95": 55.0}
Q72 = {"Q75": 30.0, "Q90": 60.0, "Q95": 80.0}
KT_12MS = TAIL_END_WIND_MS / KNOT_TO_MS_PROFILES

CASE = dict(
    center_lon=-95.0, center_lat=20.0, vmax_kt=95.0, rmw_km=30.0,
    r34_by_quadrant={"RNE": 120, "RSE": 100, "RSW": 60, "RNW": 80}, r34_unit="nm",
)


def _grid() -> tuple[np.ndarray, np.ndarray]:
    # +-15 degrees around the center: contains R_W06 in every quadrant
    return np.meshgrid(np.arange(-110.0, -80.0, 0.2), np.arange(5.0, 35.0, 0.2))


def _fields(shape: tuple[int, ...], seed: int = 0) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(seed)
    p6 = rng.gamma(0.7, 12.0, shape)
    r24 = p6 + rng.gamma(0.7, 25.0, shape)
    a72 = rng.gamma(0.8, 30.0, shape)
    return p6, r24, a72


def _snap(p6, r24, a72, lon, lat, **kw) -> dict:
    return compute_itchi_snapshot(
        p6, r24, a72, quantiles_6h=Q6, quantiles_24h=Q24, quantiles_72h=Q72,
        lon=lon, lat=lat, **{**CASE, **kw},
    )


def _r_w06() -> float:
    return compute_w06_reference_radii(CASE["vmax_kt"], CASE["center_lat"])["r_outer_km"]


# ------------------------------------------------------------------ radio de atribución

W06 = {q: 700.0 for q in QUADS}
DIRECT = {"RNE": 200.0, "RSE": 180.0, "RSW": 120.0, "RNW": 150.0}


def test_attribution_floor_by_quadrant() -> None:
    out = resolve_attribution_radius_with_w06(
        {"RNE": 800.0, "RSE": None, "RSW": 300.0, "RNW": 0.0}, W06, DIRECT
    )
    assert out.radii == {"RNE": 800.0, "RSE": 700.0, "RSW": 700.0, "RNW": 700.0}
    assert out.status == {
        "RNE": "rocloud", "RSE": "w06_missing_rocloud",
        "RSW": "w06_floor", "RNW": "w06_missing_rocloud",   # 0 = no detection
    }
    assert out.clipped_to_direct == () and out.mode == "floor"


def test_attribution_other_modes() -> None:
    roc = {"RNE": 800.0, "RSE": None, "RSW": 300.0, "RNW": 500.0}
    fallback = resolve_attribution_radius_with_w06(roc, W06, mode="fallback")
    assert fallback.radii == {"RNE": 800.0, "RSE": 700.0, "RSW": 300.0, "RNW": 500.0}
    always = resolve_attribution_radius_with_w06(roc, W06, mode="always")
    assert set(always.radii.values()) == {700.0}
    never = resolve_attribution_radius_with_w06(roc, W06, mode="never")
    assert never.radii["RSE"] == pytest.approx((800.0 + 300.0 + 500.0) / 3.0)
    assert never.status["RSE"] == "rocloud_filled"
    with pytest.raises(ValueError):
        resolve_attribution_radius_with_w06(None, W06, mode="never")
    with pytest.raises(ValueError):
        resolve_attribution_radius_with_w06(roc, W06, mode="unknown")
    with pytest.raises(ValueError):
        resolve_attribution_radius_with_w06(roc, {**W06, "RNE": np.nan})


def test_attribution_never_below_direct() -> None:
    out = resolve_attribution_radius_with_w06(None, {q: 100.0 for q in QUADS}, DIRECT)
    assert out.radii == DIRECT
    assert set(out.clipped_to_direct) == set(QUADS)


# ------------------------------------------------------------------ R_W06 y cola de viento


def test_r_w06_reference_is_pa21_outer_radius() -> None:
    ref = compute_w06_reference_radii(90.0, 20.0, rmw_km=30.0, r34_mean_km=200.0)
    pa21 = compute_parametric_radii(np.array([90.0]), np.array([20.0]), rmax_variant="pa21")
    assert ref["r_outer_km"] == pytest.approx(float(pa21["r_outer_km"][0]))
    assert ref["r34_km"] == pytest.approx(float(pa21["r34_km"][0]))
    assert ref["r_outer_km"] > ref["r34_km"] > ref["rmax_km"] > 0.0

def test_r_w06_reference_is_pa21_outer_radius() -> None:
    ref = compute_w06_reference_radii(90.0, 20.0, rmw_km=30.0, r34_mean_km=200.0)
    pa21 = compute_parametric_radii(np.array([90.0]), np.array([20.0]), rmax_variant="pa21")
    assert ref["r_outer_km"] == pytest.approx(float(pa21["r_outer_km"][0]))
    assert ref["r34_km"] == pytest.approx(float(pa21["r34_km"][0]))
    assert ref["r_outer_km"] > ref["r34_km"] > ref["rmax_km"] > 0.0


# ------------------------------------------------------------------ compute_itchi_snapshot


def test_snapshot_is_built_from_its_components() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    res = _snap(p6, r24, a72, lon, lat, rocloud_by_quadrant={q: 100.0 for q in QUADS}, run_quality_control=True)
    r_w06 = _r_w06()
    assert set(res["R_attr_status"].values()) == {"w06_floor"}
    assert res["R_attr_by_quadrant"] == pytest.approx({q: r_w06 for q in QUADS})
    hp = compute_precipitation_matrix_hazard(p6, Q6, {"R24": r24, "A72": a72}, {"R24": Q24, "A72": Q72})
    np.testing.assert_allclose(res["H_P"], hp)
    r34_mean = np.mean([120.0, 100.0, 60.0, 80.0]) * 1.852
    wind = compute_w06_wind_hazard_with_tail(
        res["radius_km"], res["R_direct_q"], res["R_W06_q"], vmax_kt=95.0, lat=20.0, rmw_km=30.0,
        r34_mean_km=r34_mean, tail_end_wind_kt=KT_12MS)
    region = res["M_direct"] | res["M_indirect"]
    np.testing.assert_allclose(res["H_W"][region], wind["H_W"][region], atol=1e-12)
    # ITCHI = unión de componentes en cada región
    d, i = res["M_direct"], res["M_indirect"]
    np.testing.assert_allclose(res["ITCHI"][d], 1 - (1 - res["H_W"][d]) * (1 - res["H_P"][d]), atol=1e-12)
    np.testing.assert_allclose(res["ITCHI"][i], 1 - (1 - res["H_W"][i]) * (1 - res["H_P"][i]), atol=1e-12)
    assert res["wind_tail_end_kt"] == pytest.approx(KT_12MS)
    assert res["wind_absolute_nodes"] == SAFFIR_SIMPSON_HAZARD_NODES


def test_snapshot_hurricane_classes() -> None:
    # Categoría 3 (100 kt): el ojo y la pared llegan a la clase 5 solo por viento
    lon, lat = _grid()
    p6 = np.zeros(lon.shape)
    res = _snap(p6, p6, p6, lon, lat, vmax_kt=100.0)
    assert float(np.max(res["H_W"])) > 0.8
    assert float(np.max(res["ITCHI"])) == pytest.approx(float(np.max(res["H_W"])))


def test_snapshot_wind_ends_at_r_w06_and_rain_extends_to_rocloud() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    r_w06 = _r_w06()
    roc = {"RNE": r_w06 + 200.0, "RSE": None, "RSW": 0.0, "RNW": r_w06 - 100.0}
    res = _snap(p6, r24, a72, lon, lat, rocloud_by_quadrant=roc, run_quality_control=True)

    assert res["R_attr_status"] == {
        "RNE": "rocloud", "RSE": "w06_missing_rocloud",
        "RSW": "w06_missing_rocloud", "RNW": "w06_floor",
    }
    assert res["R_attr_by_quadrant"]["RNE"] == pytest.approx(r_w06 + 200.0)

    beyond = res["radius_km"] >= res["R_W06_q"]
    assert (res["H_W"][beyond] < 1e-12).all()
    band = res["M_indirect"] & beyond          # indirect region beyond R_W06 (RNE)
    assert band.any()
    np.testing.assert_allclose(res["H_ind"][band], res["H_Pind"][band])
    assert (res["ITCHI"][res["M_exterior"]] == 0.0).all()


def test_snapshot_without_rocloud_uses_r_w06() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    res = _snap(p6, r24, a72, lon, lat)
    assert set(res["R_attr_status"].values()) == {"w06_missing_rocloud"}
    assert res["R_attr_by_quadrant"] == pytest.approx(res["R_W06_by_quadrant"])


def test_snapshot_missing_precipitation() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    p6[:5, :5] = np.nan
    res = _snap(p6, r24, a72, lon, lat)
    assert res["H_P_missing"][:5, :5].all() and not res["H_P_missing"][5:, 5:].any()
    assert (res["H_P"][:5, :5] == 0.0).all()
    kept = _snap(p6, r24, a72, lon, lat, precipitation_nan_value=None)
    assert np.isnan(kept["H_P"][:5, :5]).all()


def test_snapshot_tropical_depression_without_r34() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    res = _snap(p6, r24, a72, lon, lat, vmax_kt=30.0, rmw_km=None,
               r34_by_quadrant={q: None for q in QUADS})
    assert res["R_direct_source"] == "RMW_fallback_for_tropical_depression"
    assert 0.0 < float(np.max(res["H_W"])) < 0.2                   # depresión tropical: cola (clase 1)


def test_snapshot_xarray_matches_numpy() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    coords = {"lat": lat[:, 0], "lon": lon[0, :]}

    def da(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(values, dims=("lat", "lon"), coords=coords)

    roc = {"RNE": 900.0, "RSE": None, "RSW": 300.0, "RNW": 0.0}
    ref = _snap(p6, r24, a72, lon, lat, rocloud_by_quadrant=roc)
    res = _snap(da(p6), da(r24), da(a72), da(lon), da(lat), rocloud_by_quadrant=roc)
    for key in ("ITCHI", "H_P", "H_W", "H_ind", "R_attr_q", "R_W06_q"):
        assert isinstance(res[key], xr.DataArray)
        assert res[key].dims == ("lat", "lon")
        np.testing.assert_allclose(res[key].values, ref[key], atol=1e-9)


def test_snapshot_rejects_invalid_vmax() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    with pytest.raises(ValueError):
        _snap(p6, r24, a72, lon, lat, vmax_kt=np.nan)

# ------------------------------------------------------------------ configuración


def test_knot_constant_matches_wind_profiles() -> None:
    assert KNOT_TO_MS == KNOT_TO_MS_PROFILES


def test_default_config_equals_keyword_defaults() -> None:
    settings = itchi_settings_from_config(load_default_config())
    params = inspect.signature(compute_itchi_snapshot).parameters
    resolved = {"precip_matrix": PRECIP_MATRIX, "r_w06_kwargs": R_W06_DEFAULTS}
    for key, value in settings.items():
        default = params[key].default
        if default is None and key in resolved:
            default = resolved[key]
        assert value == default, key


def test_settings_can_be_passed_as_kwargs() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    settings = itchi_settings_from_config(load_default_config())
    np.testing.assert_allclose(_snap(p6, r24, a72, lon, lat, **settings)["ITCHI"], _snap(p6, r24, a72, lon, lat)["ITCHI"])


def test_experiment_overlay_changes_only_its_keys() -> None:
    cfg = merge_config(load_default_config(), {"wind": {"tail_end_wind_ms": None}})
    s = itchi_settings_from_config(cfg)
    assert s["wind_tail_end_kt"] is None
    assert s["precip_matrix"] == itchi_settings_from_config()["precip_matrix"]
    with pytest.raises(ValueError):
        itchi_settings_from_config(merge_config(load_default_config(), {"wind": {"tail_end_wind_kt": 20.0}}))


def test_quantile_names_from_percentile_file() -> None:
    da = xr.DataArray(
        np.arange(8.0).reshape(4, 2), dims=("quantile", "x"),
        coords={"quantile": [0.75, 0.9, 0.99, 0.999]},
    )
    out = quantile_dataarray_to_dict(da)
    assert list(out) == ["Q75", "Q90", "Q99", "Q99.9"]
    np.testing.assert_allclose(out["Q99.9"].values, [6.0, 7.0])
