"""
Tests for the absolute wind hazard homologated with Saffir-Simpson and the W06
wind with its tail (parametric_radii.compute_w06_wind_hazard_with_tail).
"""

from __future__ import annotations

import numpy as np
import pytest

from itchi.constants import SAFFIR_SIMPSON_HAZARD_NODES, SAFFIR_SIMPSON_KT, TAIL_END_WIND_MS
from itchi.parametric_radii import compute_w06_wind_hazard_with_tail
from itchi.plotting import classify_itchi_to_siat_levels
from itchi.wind import normalize_wind_hazard_absolute
from itchi.wind_profiles import KNOT_TO_MS

KT_12MS = TAIL_END_WIND_MS / KNOT_TO_MS


# ------------------------------------------------------------------ escala absoluta


def test_absolute_nodes_values() -> None:
    v = np.array([34.0, 49.0, 64.0, 83.0, 96.0, 137.0, 185.0])
    h = normalize_wind_hazard_absolute(v, outer_wind_kt=KT_12MS)
    np.testing.assert_allclose(h, [0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 1.0])


def test_absolute_tail_below_34_kt() -> None:
    v = np.array([10.0, KT_12MS, 0.5 * (KT_12MS + 34.0), 33.999])
    h = normalize_wind_hazard_absolute(v, outer_wind_kt=KT_12MS)
    np.testing.assert_allclose(h, [0.0, 0.0, 0.1, 0.2], atol=1e-4)


def test_saffir_simpson_categories_fall_in_their_classes() -> None:
    # Centro de cada categoría -> clase: moderado = H1, alto = H2, máximo desde H3
    centers = {"cola": 28.0, "TS": 49.0, "H1": 73.0, "H2": 89.0, "H3": 104.0, "H4": 125.0, "H5": 150.0}
    expected = {"cola": 1, "TS": 2, "H1": 3, "H2": 4, "H3": 5, "H4": 5, "H5": 5}
    h = normalize_wind_hazard_absolute(np.array(list(centers.values())), outer_wind_kt=KT_12MS)
    cls = classify_itchi_to_siat_levels(h)
    assert dict(zip(centers, cls.astype(int).tolist(), strict=True)) == expected


def test_saffir_simpson_constants_consistent() -> None:
    winds = [v for v, _ in SAFFIR_SIMPSON_HAZARD_NODES]
    for key in ("TS", "H1", "H2", "H3", "H5"):
        assert SAFFIR_SIMPSON_KT[key] in winds


def test_absolute_rejects_bad_nodes() -> None:
    with pytest.raises(ValueError):
        normalize_wind_hazard_absolute(np.array([50.0]), 20.0, nodes=((34.0, 0.2), (30.0, 0.4)))
    with pytest.raises(ValueError):
        normalize_wind_hazard_absolute(np.array([50.0]), 20.0, nodes=((34.0, 0.5), (64.0, 0.4)))


# ------------------------------------------------------------------ W06 con cola


def test_absolute_scale_does_not_depend_on_vmax() -> None:
    # Mismo viento local, mismo peligro, en una tormenta de 45 kt y en una de 140 kt
    r = np.linspace(5.0, 600.0, 400)
    out = {
        vmax: compute_w06_wind_hazard_with_tail(
            radius_km=r, direct_radius_km=150.0, attribution_radius_km=600.0,
            vmax_kt=vmax, lat=20.0, rmw_km=30.0, tail_end_wind_kt=KT_12MS,
        )
        for vmax in (45.0, 140.0)
    }
    for vmax, res in out.items():
        expected = normalize_wind_hazard_absolute(res["wind_kt"], res["outer_wind_kt"])
        outside_eye = (res["wind_kt"] < 0.999 * vmax) & (r > 30.0)
        np.testing.assert_allclose(res["H_W"][outside_eye], expected[outside_eye])
        # ojo: dentro del RMW el peligro es el de Vmax
        h_vmax = normalize_wind_hazard_absolute(np.array([vmax]), KT_12MS)[0]
        np.testing.assert_allclose(res["H_W"][r <= 30.0], h_vmax)
    assert np.nanmax(out[45.0]["H_W"]) < 0.4                        # tormenta tropical: clase 2
    assert np.nanmax(out[140.0]["H_W"]) == pytest.approx(1.0)


def test_w06_tail_continuity_and_zero_at_attribution() -> None:
    r = np.linspace(0.0, 700.0, 7001)
    for r34_q in (150.0, 250.0):
        out = compute_w06_wind_hazard_with_tail(r, r34_q, 600.0, vmax_kt=90.0, lat=20.0, r34_mean_km=200.0)
        h, v = out["H_W"], out["wind_kt"]
        i34 = np.searchsorted(r, r34_q)
        assert out["x1_calibrated"]
        assert abs(v[i34] - 34.0) < 0.2
        assert abs(h[i34] - 0.2) < 0.01 and abs(h[i34 + 1] - 0.2) < 0.01
        assert (h[np.searchsorted(r, 600.0):] < 1e-6).all()


def test_w06_tail_cut_at_wind_speed() -> None:
    r = np.linspace(0.0, 900.0, 9001)
    kw = dict(vmax_kt=90.0, lat=20.0, r34_mean_km=200.0)
    full = compute_w06_wind_hazard_with_tail(r, 200.0, 800.0, **kw)
    cut = compute_w06_wind_hazard_with_tail(r, 200.0, 800.0, tail_end_wind_kt=KT_12MS, **kw)
    outside = r > 200.0
    assert (cut["H_W"][outside & (cut["wind_kt"] <= KT_12MS)] < 1e-12).all()
    np.testing.assert_allclose(cut["H_W"][~outside], full["H_W"][~outside])
    assert (cut["H_W"] <= full["H_W"] + 1e-12).all()
    with pytest.raises(ValueError):
        compute_w06_wind_hazard_with_tail(r, 200.0, 800.0, tail_end_wind_kt=40.0, **kw)


def test_w06_tail_tropical_depression_stays_below_class_2() -> None:
    r = np.linspace(0.0, 600.0, 601)
    out = compute_w06_wind_hazard_with_tail(r, np.nan, 500.0, vmax_kt=30.0, lat=18.0)
    assert not out["x1_calibrated"]
    assert 0.0 < out["H_W"].max() < 0.2
