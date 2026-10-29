"""
Tests for the ITCHI v0.2 entry point (pipeline.compute_itchi_snapshot_v02),
the attribution radius ROCLOUD with R_W06 floor, the R_W06 definition, the
optional cut of the wind tail and the consistency between configs/default.yaml
and the v0.2 defaults.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import xarray as xr

from itchi.climatology import quantile_dataarray_to_dict
from itchi.config import itchi_v02_settings_from_config, load_default_config
from itchi.parametric_radii import (
    R_W06_DEFAULTS,
    compute_parametric_radii,
    compute_w06_reference_radii,
    compute_w06_wind_hazard_with_tail,
)
from itchi.pipeline import compute_itchi_snapshot_from_grid, compute_itchi_snapshot_v02
from itchi.precipitation import PRECIP_MATRIX_V02, compute_precipitation_matrix_hazard
from itchi.radii import resolve_attribution_radius_with_w06

QUADS = ("RNE", "RSE", "RSW", "RNW")
Q6 = {"Q75": 8.0, "Q90": 18.0, "Q95": 28.0, "Q99": 50.0}
Q24 = {"Q90": 40.0, "Q95": 55.0}
Q72 = {"Q75": 30.0, "Q90": 60.0, "Q95": 80.0}

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


def _v02(p6, r24, a72, lon, lat, **kw) -> dict:
    return compute_itchi_snapshot_v02(
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


def test_w06_tail_cut_at_wind_speed() -> None:
    r = np.linspace(0.0, 900.0, 9001)
    kw = dict(vmax_kt=90.0, lat=20.0, r34_mean_km=200.0, h34=0.2)
    full = compute_w06_wind_hazard_with_tail(r, 200.0, 800.0, **kw)
    cut = compute_w06_wind_hazard_with_tail(r, 200.0, 800.0, tail_end_wind_kt=17.0, **kw)
    outside = r > 200.0
    assert (cut["H_W"][outside & (cut["wind_kt"] <= 17.0)] < 1e-12).all()
    np.testing.assert_allclose(cut["H_W"][~outside], full["H_W"][~outside])
    assert (cut["H_W"] <= full["H_W"] + 1e-12).all()
    with pytest.raises(ValueError):
        compute_w06_wind_hazard_with_tail(r, 200.0, 800.0, tail_end_wind_kt=40.0, **kw)


# ------------------------------------------------------------------ compute_itchi_snapshot_v02


def test_v02_reproduces_the_v02_study_setting() -> None:
    # ROCLOUD smaller than R_W06 in every quadrant -> R_attr = R_W06 = end of
    # the wind tail, the setting of notebooks 05-08 (attribution = R_W06)
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    r_w06 = _r_w06()
    res = _v02(p6, r24, a72, lon, lat, rocloud_by_quadrant={q: 100.0 for q in QUADS},
               run_quality_control=True)
    assert set(res["R_attr_status"].values()) == {"w06_floor"}
    assert res["R_attr_by_quadrant"] == pytest.approx({q: r_w06 for q in QUADS})

    hp = compute_precipitation_matrix_hazard(
        p6, Q6, {"R24": r24, "A72": a72}, {"R24": Q24, "A72": Q72}
    )
    ref = compute_itchi_snapshot_from_grid(
        p6, None, None, None, lon=lon, lat=lat,
        rocloud_by_quadrant={q: r_w06 for q in QUADS}, rocloud_unit="km",
        precipitation_hazard=hp, wind_model="w06_tail", **CASE,
    )
    for key in ("M_direct", "M_indirect", "M_exterior"):
        np.testing.assert_array_equal(res[key], ref[key])
    for key in ("H_P", "H_W", "H_Wind", "H_dir", "H_ind", "ITCHI"):
        np.testing.assert_allclose(res[key], ref[key], atol=1e-12)
    assert res["itchi_formulation"] == "v0.2"


def test_v02_wind_ends_at_r_w06_and_rain_extends_to_rocloud() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    r_w06 = _r_w06()
    roc = {"RNE": r_w06 + 200.0, "RSE": None, "RSW": 0.0, "RNW": r_w06 - 100.0}
    res = _v02(p6, r24, a72, lon, lat, rocloud_by_quadrant=roc, run_quality_control=True)

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


def test_v02_without_rocloud_uses_r_w06() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    res = _v02(p6, r24, a72, lon, lat)
    assert set(res["R_attr_status"].values()) == {"w06_missing_rocloud"}
    assert res["R_attr_by_quadrant"] == pytest.approx(res["R_W06_by_quadrant"])


def test_v02_missing_precipitation() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    p6[:5, :5] = np.nan
    res = _v02(p6, r24, a72, lon, lat)
    assert res["H_P_missing"][:5, :5].all() and not res["H_P_missing"][5:, 5:].any()
    assert (res["H_P"][:5, :5] == 0.0).all()
    kept = _v02(p6, r24, a72, lon, lat, precipitation_nan_value=None)
    assert np.isnan(kept["H_P"][:5, :5]).all()


def test_v02_tropical_depression_without_r34() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    res = _v02(p6, r24, a72, lon, lat, vmax_kt=30.0, rmw_km=None,
               r34_by_quadrant={q: None for q in QUADS})
    assert res["R_direct_source"] == "RMW_fallback_for_tropical_depression"
    assert 0.0 < float(np.max(res["H_W"])) < 0.2


def test_v02_xarray_matches_numpy() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    coords = {"lat": lat[:, 0], "lon": lon[0, :]}

    def da(values: np.ndarray) -> xr.DataArray:
        return xr.DataArray(values, dims=("lat", "lon"), coords=coords)

    roc = {"RNE": 900.0, "RSE": None, "RSW": 300.0, "RNW": 0.0}
    ref = _v02(p6, r24, a72, lon, lat, rocloud_by_quadrant=roc)
    res = _v02(da(p6), da(r24), da(a72), da(lon), da(lat), rocloud_by_quadrant=roc)
    for key in ("ITCHI", "H_P", "H_W", "H_ind", "R_attr_q", "R_W06_q"):
        assert isinstance(res[key], xr.DataArray)
        assert res[key].dims == ("lat", "lon")
        np.testing.assert_allclose(res[key].values, ref[key], atol=1e-9)


def test_v02_rejects_invalid_vmax() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    with pytest.raises(ValueError):
        _v02(p6, r24, a72, lon, lat, vmax_kt=np.nan)


# ------------------------------------------------------------------ configuración


def test_default_config_equals_v02_defaults() -> None:
    settings = itchi_v02_settings_from_config(load_default_config())
    params = inspect.signature(compute_itchi_snapshot_v02).parameters
    resolved = {"precip_matrix": PRECIP_MATRIX_V02, "r_w06_kwargs": R_W06_DEFAULTS}
    for key, value in settings.items():
        default = params[key].default
        if default is None and key in resolved:
            default = resolved[key]
        assert value == default, key


def test_v02_settings_can_be_passed_as_kwargs() -> None:
    lon, lat = _grid()
    p6, r24, a72 = _fields(lon.shape)
    settings = itchi_v02_settings_from_config(load_default_config())
    np.testing.assert_allclose(
        _v02(p6, r24, a72, lon, lat, **settings)["ITCHI"],
        _v02(p6, r24, a72, lon, lat)["ITCHI"],
    )


def test_quantile_names_from_percentile_file() -> None:
    da = xr.DataArray(
        np.arange(8.0).reshape(4, 2), dims=("quantile", "x"),
        coords={"quantile": [0.75, 0.9, 0.99, 0.999]},
    )
    out = quantile_dataarray_to_dict(da)
    assert list(out) == ["Q75", "Q90", "Q99", "Q99.9"]
    np.testing.assert_allclose(out["Q99.9"].values, [6.0, 7.0])
