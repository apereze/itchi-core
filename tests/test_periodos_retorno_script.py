"""
Tests for scripts/periodos_retorno.py with a synthetic reader (no MSWEP files).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from itchi.return_periods import declustered_peaks, gev_return_period, running_sum

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "periodos_retorno.py"
_spec = importlib.util.spec_from_file_location("periodos_retorno", SCRIPT)
pr = importlib.util.module_from_spec(_spec)
sys.modules["periodos_retorno"] = pr
_spec.loader.exec_module(pr)

LAT, LON = np.array([20.0, 20.1, 20.2]), np.array([-100.0, -99.9])


def _season(year: int) -> np.ndarray:
    return np.arange(np.datetime64(f"{year}-05-01T06"), np.datetime64(f"{year}-12-01T00"),
                     np.timedelta64(6, "h"))


def _reader(seed: int = 0, missing_year: int | None = None, gap: bool = False):
    rng = np.random.default_rng(seed)
    data = {}

    def read(year: int):
        if year == missing_year:
            return None
        if year not in data:
            t = _season(year)
            x = rng.gamma(0.25, 6.0, (t.size, LAT.size, LON.size)).astype(np.float32)
            x *= (rng.random(x.shape) < 0.3)
            x[:, 0, 0] *= 2.0                               # píxel más lluvioso
            if gap:
                keep = np.ones(t.size, bool)
                keep[100:104] = False
                t, x = t[keep], x[keep]
            data[year] = (t, LAT, LON, x)
        return data[year]

    return read, data


def _cfg(**kw) -> "pr.ConfigRP":
    return pr.ConfigRP(start_year=2000, end_year=2025, filas_por_banda=2, **kw)


def test_contiguous_fills_gaps() -> None:
    t = _season(2001)
    x = np.arange(t.size, dtype=float)[:, None]
    keep = np.ones(t.size, bool)
    keep[[5, 6, 50]] = False
    out = pr.contiguous(t[keep], x[keep])
    assert out.shape == x.shape and np.isnan(out[[5, 6, 50], 0]).all()
    np.testing.assert_allclose(out[keep], x[keep])
    with pytest.raises(ValueError):
        pr.contiguous(t[[0, 0, 1]], x[:3])


def test_compute_matches_brute_force() -> None:
    read, data = _reader()
    cfg = _cfg()
    res = pr.compute(cfg, read, verbose=False)
    st = res["state"]
    for d in cfg.durations:
        allpeaks = []
        for iy, year in enumerate(cfg.years):
            r = running_sum(data[year][3].astype(float), d // 6)
            np.testing.assert_allclose(st["am"][d][iy], np.nanmax(r, axis=0), rtol=1e-6)
            allpeaks.append(declustered_peaks(r, cfg.separacion_h[d] // 6, cfg.wet))
        allpeaks = np.concatenate(allpeaks)
        for j in range(LAT.size):
            for i in range(LON.size):
                v = np.sort(allpeaks[:, j, i][np.isfinite(allpeaks[:, j, i])])[::-1][: cfg.k + 1]
                np.testing.assert_allclose(st["top"][d][: v.size, j, i], v, rtol=1e-6)
        f = res["fits"][d]
        assert (f["n_seasons"] == 26).all()
        assert np.isfinite(f["gev"]["xi"]).all() and np.isfinite(f["gpd"]["xi"]).all()
        # niveles crecientes con T y R; píxel más lluvioso con niveles mayores
        assert (np.diff(f["rl_gev"], axis=0) > 0).all()
        assert (np.diff(f["rl_pot"], axis=0)[np.isfinite(np.diff(f["rl_pot"], axis=0))] > 0).all()
        assert f["rl_gev"][2, 0, 0] > f["rl_gev"][2, 1, 1]
        # el máximo observado en 26 temporadas tiene T de decenas de años
        T_max = gev_return_period(np.nanmax(st["am"][d], axis=0), f["gev"]["xi"], f["gev"]["sigma"],
                                  f["gev"]["mu"])
        assert (T_max > 5.0).all()


def test_compute_missing_year_and_gaps() -> None:
    read, _ = _reader(seed=1, missing_year=2003, gap=True)
    res = pr.compute(_cfg(durations=(24,)), read, verbose=False)
    assert 2003 not in res["leidos"] and len(res["leidos"]) == 25
    assert (res["fits"][24]["n_seasons"] == 25).all()
    assert np.isnan(res["state"]["am"][24][3]).all()


def test_compute_without_files_raises() -> None:
    with pytest.raises(FileNotFoundError):
        pr.compute(_cfg(), lambda year: None, verbose=False)


def test_compute_counts_events_above_thresholds() -> None:
    read, data = _reader(seed=2)
    cfg = _cfg(durations=(6, 24))
    rng = np.random.default_rng(3)
    thr = {d: rng.uniform(1.0, 30.0, (3, LAT.size, LON.size)).astype(np.float32) for d in cfg.durations}
    thr[6][1, 0, 1] = np.nan                                # umbral faltante -> NaN
    res = pr.compute(cfg, read, verbose=False, thresholds=lambda lat, lon: thr)
    for d in cfg.durations:
        cnt = np.zeros(thr[d].shape)
        for year in cfg.years:
            pk = declustered_peaks(running_sum(data[year][3].astype(float), d // 6), cfg.separacion_h[d] // 6,
                                   cfg.wet)
            cnt += (pk[None] >= thr[d][:, None]).sum(axis=1)
        expected = np.where(np.isfinite(thr[d]), cnt / 26.0, np.nan)
        np.testing.assert_allclose(res["fits"][d]["events_above_pct"], expected, rtol=1e-6)
        assert res["fits"][d]["full_threshold"].all()


def test_full_threshold_flag_arid_pixel() -> None:
    read, data = _reader(seed=4)

    def arid(year):
        t, la, lo, x = read(year)
        x = x.copy()
        x[:, 2, 1] = 0.0
        x[[10, 400], 2, 1] = 5.0                            # dos eventos por temporada < K+1 = 79
        return t, la, lo, x

    res = pr.compute(_cfg(durations=(6,)), arid, verbose=False)
    full = res["fits"][6]["full_threshold"]
    assert not full[2, 1] and full[0, 0]
    assert res["fits"][6]["gpd"]["rate"][2, 1] < 3.0 or np.isnan(res["fits"][6]["gpd"]["rate"][2, 1])
