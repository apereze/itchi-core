"""Tests for itchi.wind_profiles and itchi.parametric_radii."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from itchi.parametric_radii import (
    add_parametric_radii,
    build_parametric_rocloud_record,
    compute_parametric_radii,
    compute_w06_wind_hazard,
    fit_x1_to_radius,
    parametric_direct_fallback_km,
)
from itchi.wind_profiles import (
    KNOT_TO_MS,
    V_R34_MS,
    Willoughby2006Profile,
    radius_of_wind_km,
)

VMAX_MS = np.array([20.0, 35.0, 50.0, 70.0])
LAT = np.array([30.0, 15.0, 20.0, 22.0])


@pytest.mark.parametrize("idx", range(len(VMAX_MS)))
def test_maximum_at_rmax(idx: int) -> None:
    profile = Willoughby2006Profile(VMAX_MS[idx], LAT[idx])
    rmax = float(profile.rmax)
    radius = np.linspace(1.0, 500.0, 49901)  # dr = 0.01 km
    wind = profile(radius)

    assert float(profile(rmax)) == pytest.approx(VMAX_MS[idx], abs=1e-9)
    assert abs(radius[np.argmax(wind)] - rmax) < 0.5

    h = 1e-4  # centered difference: dV/dr = 0 at Rmax
    assert abs((profile(rmax + h) - profile(rmax - h)) / (2 * h)) < 1e-4


def test_threshold_is_hit() -> None:
    profile = Willoughby2006Profile(VMAX_MS, LAT)

    for threshold in (2.0, 5 * KNOT_TO_MS, V_R34_MS):
        radius = radius_of_wind_km(profile, threshold)
        valid = VMAX_MS > threshold
        np.testing.assert_allclose(profile(radius)[valid], threshold, atol=1e-8)


def test_single_exponential_limit() -> None:
    # vmax = 20 m s-1, lat = 30 -> A < 0 -> A = 0, V = Vmax exp(-(r - Rmax)/X1)
    profile = Willoughby2006Profile(20.0, 30.0)
    assert float(profile.a) == 0.0

    radius = float(radius_of_wind_km(profile, 2.0))
    expected = float(profile.rmax) + float(profile.x1) * math.log(20.0 / 2.0)
    assert radius == pytest.approx(expected, abs=1e-6)


def test_undefined_r34_is_nan() -> None:
    radii = compute_parametric_radii(vmax_kt=np.array([25.0]), lat=np.array([20.0]))
    assert np.isnan(radii["r34_km"][0])


def test_x1_calibration_roundtrip() -> None:
    x1_true = np.array([150.0, 300.0, 450.0])
    vmax, lat = np.array([30.0, 45.0, 60.0]), np.array([18.0, 25.0, 30.0])
    r34 = radius_of_wind_km(Willoughby2006Profile(vmax, lat, x1_km=x1_true), V_R34_MS)

    x1_fit = fit_x1_to_radius(vmax, lat, None, r34)
    np.testing.assert_allclose(x1_fit, x1_true, atol=1e-3)


def test_reproduces_pa21_cases() -> None:
    # PA21: Irma 2017-09-09 06 UTC -> 745 km; Willa 2018-10-24 00 UTC -> 812 km.
    # Inputs are approximate; 5 % tolerance.
    radii = compute_parametric_radii(
        vmax_kt=np.array([135.0, 50.0 / KNOT_TO_MS]),
        lat=np.array([22.1, 22.0]),
        rmax_variant="pa21",
    )
    np.testing.assert_allclose(radii["r_outer_km"], [745.0, 812.0], rtol=0.05)


def test_track_dataframe_adapter() -> None:
    track = pd.DataFrame(
        {
            "storm_id": ["A", "A", "B"],
            "time": pd.to_datetime(
                ["2017-09-09 06:00", "2017-09-09 12:00", "2018-10-24 00:00"]
            ),
            "lat": [22.1, 22.3, 22.0],
            "lon": [-78.0, -78.5, -106.0],
            "vmax_kt": [135.0, 130.0, 25.0],
            "rmw_km": [27.8, np.nan, np.nan],
            "r34_rne": [180.0, 180.0, np.nan],
            "r34_rse": [150.0, np.nan, np.nan],
            "r34_rsw": [110.0, 110.0, np.nan],
            "r34_rnw": [150.0, 150.0, np.nan],
        }
    )
    out = add_parametric_radii(track, use_observed_rmw=True, calibrate_to_r34=True)

    assert out["w06_x1_calibrated"].tolist() == [True, True, False]
    assert np.isnan(out.loc[2, "w06_r34_km"])  # tropical depression
    assert (out["w06_r_outer_km"] > out["w06_r34_km"].fillna(0)).all()

    # calibrated profile reproduces the observed mean R34 (nm -> km)
    r34_mean_km = np.mean([180.0, 150.0, 110.0, 150.0]) * 1.852
    assert out.loc[0, "w06_r34_km"] == pytest.approx(r34_mean_km, rel=1e-6)

    record = build_parametric_rocloud_record(out.iloc[0])
    r_outer = out.loc[0, "w06_r_outer_km"]
    assert record["rocloud_rne"] == record["rocloud_rnw"] == r_outer
    assert parametric_direct_fallback_km(out.iloc[2]) is None


def test_w06_wind_hazard_vanishes_at_r34() -> None:
    radii = compute_parametric_radii(np.array([100.0]), np.array([20.0]))
    r34 = float(radii["r34_km"][0])
    radius = np.array([r34 - 1.0, r34, r34 + 50.0])
    hazard = compute_w06_wind_hazard(radius, 100.0, 20.0)

    assert hazard[0] > 0.0
    assert hazard[1] == pytest.approx(0.0, abs=1e-6)
    assert hazard[2] == 0.0
