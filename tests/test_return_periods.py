"""
Tests for itchi.return_periods: series operations, L-moments, GEV and GPD fits.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from itchi.return_periods import (
    annual_return_period,
    declustered_peaks,
    empirical_pot_level,
    fit_gev_lmoments,
    fit_gpd_top_k,
    gev_return_level,
    gev_return_period,
    pot_recurrence_interval,
    pot_return_level,
    recurrence_from_annual,
    running_sum,
    sample_lmoments,
    seasonal_maximum,
    update_top_k,
)


def _gev_sample(rng, n, xi, sigma, mu):
    u = rng.random(n)
    y = -np.log(u)
    return mu + sigma / xi * (y ** (-xi) - 1.0) if abs(xi) > 1e-9 else mu - sigma * np.log(y)


# ------------------------------------------------------------------ series


def test_running_sum_and_nan() -> None:
    x = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    np.testing.assert_allclose(running_sum(x, 2), [np.nan, 3.0, 5.0, np.nan, np.nan, 11.0])
    np.testing.assert_allclose(running_sum(x, 1), x)
    assert np.isnan(running_sum(np.ones(2), 4)).all()


def test_declustered_peaks_separation_ties_and_threshold() -> None:
    x = np.array([0, 5, 9, 7, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 3, 3, 0, 0.5, 0], dtype=float)
    p = declustered_peaks(x, min_separation=3, threshold=1.0)
    assert np.nonzero(np.isfinite(p))[0].tolist() == [2, 8, 15]     # 9; 8; first of the tie; 0.5 < 1 mm
    p2 = declustered_peaks(x, min_separation=6, threshold=1.0)
    assert np.nonzero(np.isfinite(p2))[0].tolist() == [2, 15]       # 8 is within 6 steps of 9
    xn = x.copy()
    xn[2] = np.nan
    assert np.isfinite(declustered_peaks(xn, 3)[3])                  # NaN does not block a neighbour


def test_declustered_peaks_two_dimensional() -> None:
    rng = np.random.default_rng(0)
    x = rng.gamma(0.3, 5.0, (200, 7))
    p = declustered_peaks(x, 4)
    for j in range(7):
        idx = np.nonzero(np.isfinite(p[:, j]))[0]
        assert (np.diff(idx) > 4).all()
        np.testing.assert_allclose(p[idx, j], x[idx, j])


def test_update_top_k_equals_full_sort() -> None:
    rng = np.random.default_rng(1)
    chunks = [np.where(rng.random((50, 4)) < 0.3, rng.gamma(1.0, 10.0, (50, 4)), np.nan) for _ in range(5)]
    top = None
    for c in chunks:
        top = update_top_k(top, c, 12)
    allv = np.concatenate(chunks)
    for j in range(4):
        ref = np.sort(allv[:, j][np.isfinite(allv[:, j])])[::-1][:12]
        np.testing.assert_allclose(top[: ref.size, j], ref)
    few = update_top_k(None, np.array([[3.0], [np.nan], [1.0]]), 5)
    assert few[:2, 0].tolist() == [3.0, 1.0] and np.isnan(few[2:, 0]).all()


def test_seasonal_maximum_requires_valid_fraction() -> None:
    x = np.ones((10, 2))
    x[:, 1] = np.nan
    x[0, 1] = 5.0
    np.testing.assert_allclose(seasonal_maximum(x), [1.0, np.nan])


# ------------------------------------------------------------------ L-momentos


def test_sample_lmoments_against_definition() -> None:
    rng = np.random.default_rng(2)
    x = rng.gamma(2.0, 3.0, 9)
    l1, l2, l3, n = sample_lmoments(x[:, None])
    # definición directa por combinaciones (Hosking, 1990)
    s = np.sort(x)
    c2 = [b - a for a, b in itertools.combinations(s, 2)]
    c3 = [c - 2 * b + a for a, b, c in itertools.combinations(s, 3)]
    assert l1[0] == pytest.approx(x.mean())
    assert l2[0] == pytest.approx(np.mean(c2) / 2)
    assert l3[0] == pytest.approx(np.mean(c3) / 3)
    xn = np.r_[x, np.nan, np.nan][:, None]
    assert sample_lmoments(xn)[1][0] == pytest.approx(l2[0]) and sample_lmoments(xn)[3][0] == 9


# ------------------------------------------------------------------ GEV


@pytest.mark.parametrize("xi", [-0.15, 0.0, 0.15])
def test_gev_fit_recovers_parameters(xi: float) -> None:
    rng = np.random.default_rng(3)
    sample = _gev_sample(rng, 20000, xi, 20.0, 80.0)[:, None]
    fit = fit_gev_lmoments(sample)
    assert fit["xi"][0] == pytest.approx(xi, abs=0.03)
    assert fit["sigma"][0] == pytest.approx(20.0, rel=0.04)
    assert fit["mu"][0] == pytest.approx(80.0, rel=0.02)
    true10 = gev_return_level(10.0, xi, 20.0, 80.0)
    assert gev_return_level(10.0, fit["xi"], fit["sigma"], fit["mu"])[0] == pytest.approx(true10, rel=0.03)


def test_gev_return_level_and_period_are_inverse() -> None:
    T = np.array([1.5, 2.0, 5.0, 10.0, 50.0, 100.0])
    for xi in (-0.2, 0.0, 0.2):
        x = gev_return_level(T, xi, 15.0, 60.0)
        np.testing.assert_allclose(gev_return_period(x, xi, 15.0, 60.0), T, rtol=1e-9)
    assert gev_return_period(1e6, -0.2, 15.0, 60.0) == np.inf       # above the upper bound
    assert gev_return_period(-1e6, 0.2, 15.0, 60.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        gev_return_level(1.0, 0.0, 1.0, 0.0)


def test_gev_min_n_and_clipping() -> None:
    rng = np.random.default_rng(4)
    am = _gev_sample(rng, 26 * 400, 0.1, 20.0, 80.0).reshape(26, 400)
    am[12:, 0] = np.nan                       # 12 seasons < min_n
    fit = fit_gev_lmoments(am)
    assert np.isnan(fit["xi"][0]) and np.isfinite(fit["xi"][1:]).all()
    assert (fit["xi"][1:] >= -0.5).all() and (fit["xi"][1:] <= 0.5).all()
    heavy = _gev_sample(np.random.default_rng(5), 5000, 0.9, 20.0, 80.0)[:, None]
    f2 = fit_gev_lmoments(heavy)
    assert f2["xi"][0] == pytest.approx(0.5) and f2["xi_clipped"][0]


# ------------------------------------------------------------------ GPD (POT)


def _pot_record(rng, n_seasons, rate, xi, sigma, u, k):
    """Peaks of a Poisson process above u with GPD excesses; returns the top k+1."""
    n = rng.poisson(rate * n_seasons)
    v = rng.random(n)
    exc = sigma / xi * (v ** (-xi) - 1.0) if abs(xi) > 1e-9 else -sigma * np.log(v)
    peaks = np.sort(u + exc)[::-1]
    return update_top_k(None, peaks[:, None], k + 1)


@pytest.mark.parametrize("xi", [-0.1, 0.0, 0.2])
def test_gpd_fit_and_return_levels(xi: float) -> None:
    rng = np.random.default_rng(6)
    n_seasons, rate, sigma, u, k = 2000, 3.0, 12.0, 30.0, 6000
    top = _pot_record(rng, n_seasons, rate, xi, sigma, u, k)
    fit = fit_gpd_top_k(top, n_seasons)
    assert fit["xi"][0] == pytest.approx(xi, abs=0.04)
    for R in (1.0, 10.0):
        lam = 1.0 / R
        true = u + (sigma / xi * ((rate / lam) ** xi - 1.0) if xi else sigma * np.log(rate / lam))
        est = pot_return_level(R, fit["u"], fit["sigma"], fit["xi"], fit["rate"])[0]
        assert est == pytest.approx(true, rel=0.05)
        emp = empirical_pot_level(top, n_seasons, [R])[0, 0]
        assert emp == pytest.approx(true, rel=0.05)


def test_pot_level_and_interval_are_inverse() -> None:
    R = np.array([0.5, 1.0, 2.0, 10.0, 50.0])
    for xi in (-0.2, 0.0, 0.25):
        x = pot_return_level(R, 25.0, 10.0, xi, 3.0)
        np.testing.assert_allclose(pot_recurrence_interval(x, 25.0, 10.0, xi, 3.0), R, rtol=1e-9)
    assert np.isnan(pot_return_level(0.2, 25.0, 10.0, 0.1, 3.0))        # more frequent than u
    assert np.isnan(pot_recurrence_interval(20.0, 25.0, 10.0, 0.1, 3.0))  # below u
    assert pot_recurrence_interval(1e5, 25.0, 10.0, -0.2, 3.0) == np.inf


def test_gpd_needs_enough_peaks() -> None:
    top = np.full((79, 2), np.nan)
    top[:10, 0] = np.linspace(50.0, 10.0, 10)
    top[:, 1] = np.linspace(80.0, 5.0, 79)
    fit = fit_gpd_top_k(top, 26)
    assert np.isnan(fit["u"][0]) and fit["u"][1] == pytest.approx(5.0)
    assert fit["rate"][1] == pytest.approx(78 / 26)


def test_empirical_pot_level_ranks() -> None:
    top = np.arange(100.0, 0.0, -1.0)[:, None]             # 100, 99, ..., 1
    lv = empirical_pot_level(top, 10.0, [10.0, 1.0, 0.5, 0.05])[:, 0]
    assert lv[0] == 100.0 and lv[1] == 91.0 and lv[2] == 81.0 and np.isnan(lv[3])


# ------------------------------------------------------------------ conversiones


def test_annual_return_period_conversion() -> None:
    assert annual_return_period(1.0) == pytest.approx(1.0 / (1.0 - math.exp(-1.0)))
    T = np.array([1.1, 2.0, 10.0, 100.0])
    np.testing.assert_allclose(annual_return_period(recurrence_from_annual(T)), T)
    assert annual_return_period(100.0) == pytest.approx(100.5, abs=0.01)
