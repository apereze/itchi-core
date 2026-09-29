"""Tests for itchi.chavas_outer (E04 outer model constrained by R34)."""

from __future__ import annotations

import numpy as np
import pytest

from itchi.chavas_outer import (
    CD_DEFAULT,
    V34_MS,
    W_COOL_DEFAULT,
    coriolis,
    e04_outer_wind,
    e04_r0_from_r34,
    e04_radius_of_wind,
)


def test_r34_constraint_is_met() -> None:
    for r34, lat in ((100.0, 15.0), (250.0, 20.0), (400.0, 30.0)):
        r0 = e04_r0_from_r34(r34, lat)
        assert float(e04_outer_wind(np.array([r34]), r0, lat)[0]) == pytest.approx(
            V34_MS, abs=1e-3
        )


def test_ode_is_satisfied() -> None:
    lat, r0_km = 20.0, 1000.0
    f, r0 = coriolis(lat), r0_km * 1e3
    r_km = np.linspace(200.0, 900.0, 8)
    h = 1e-3  # km
    m = lambda rk: rk * 1e3 * e04_outer_wind(rk, r0_km, lat) + 0.5 * f * (rk * 1e3) ** 2
    dm_dr = (m(r_km + h) - m(r_km - h)) / (2 * h * 1e3)
    rv = r_km * 1e3 * e04_outer_wind(r_km, r0_km, lat)
    expected = 2 * CD_DEFAULT / W_COOL_DEFAULT * rv**2 / (r0**2 - (r_km * 1e3) ** 2)
    np.testing.assert_allclose(dm_dr, expected, rtol=1e-4)


def test_profile_shape() -> None:
    r0 = e04_r0_from_r34(200.0, 20.0)
    r = np.linspace(200.0, r0 * 0.999, 200)
    v = e04_outer_wind(r, r0, 20.0)
    assert np.all(np.diff(v) < 0) and v[-1] < 0.5
    assert float(e04_outer_wind(np.array([r0 * 1.01]), r0, 20.0)[0]) == 0.0


def test_monotonic_sensitivities() -> None:
    r0_r34 = [e04_r0_from_r34(r34, 20.0) for r34 in (100.0, 200.0, 300.0)]
    r0_lat = [e04_r0_from_r34(250.0, lat) for lat in (15.0, 25.0, 35.0)]
    assert np.all(np.diff(r0_r34) > 0)  # larger R34 -> larger r0
    assert np.all(np.diff(r0_lat) < 0)  # larger f -> smaller r0


def test_outer_radius_between_r34_and_r0() -> None:
    r0, r2 = e04_radius_of_wind(250.0, 20.0, 2.0)
    assert 250.0 < r2 < r0
    wind = float(e04_outer_wind(np.array([r2]), r0, 20.0)[0])
    assert wind == pytest.approx(2.0, abs=1e-3)


def test_missing_r34_gives_nan() -> None:
    assert np.isnan(e04_r0_from_r34(np.nan, 20.0))
    assert np.isnan(e04_r0_from_r34(0.0, 20.0))
