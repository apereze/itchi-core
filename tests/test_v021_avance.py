"""
Tests for the v0.2.1 progress-report options: maximum of the precipitation
matrix at the 6-h P95 and absolute wind hazard homologated with the
Saffir-Simpson scale (configs/experiments/v021/).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from itchi.config import itchi_v02_settings_from_config, load_experiment_config
from itchi.constants import SAFFIR_SIMPSON_HAZARD_NODES, SAFFIR_SIMPSON_KT
from itchi.parametric_radii import compute_w06_wind_hazard_with_tail
from itchi.pipeline import compute_itchi_snapshot_v02
from itchi.plotting import classify_itchi_to_siat_levels
from itchi.precipitation import (
    PRECIP_MATRIX_V02,
    PRECIP_MATRIX_V02_P95,
    compute_precipitation_matrix_hazard,
)
from itchi.wind import normalize_wind_hazard_absolute

ROOT = Path(__file__).resolve().parents[1]
V021 = ROOT / "configs" / "experiments" / "v021"
KT_12MS = 12.0 / 0.514444

Q6 = {"Q75": 8.0, "Q90": 18.0, "Q95": 28.0, "Q99": 50.0}
Q24 = {"Q90": 40.0, "Q95": 55.0}
Q72 = {"Q75": 30.0}


# ------------------------------------------------------------------ viento absoluto


def test_absolute_nodes_values() -> None:
    v = np.array([34.0, 49.0, 64.0, 80.0, 96.0, 113.0, 137.0, 185.0])
    h = normalize_wind_hazard_absolute(v, outer_wind_kt=KT_12MS)
    np.testing.assert_allclose(h, [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.0])


def test_absolute_tail_below_34_kt() -> None:
    v = np.array([10.0, KT_12MS, 0.5 * (KT_12MS + 34.0), 33.999])
    h = normalize_wind_hazard_absolute(v, outer_wind_kt=KT_12MS)
    np.testing.assert_allclose(h, [0.0, 0.0, 0.1, 0.2], atol=1e-4)


def test_saffir_simpson_categories_fall_in_their_classes() -> None:
    # Centro de cada categoría -> clase SIAT esperada
    centers = {"cola": 28.0, "TS": 49.0, "H1": 73.0, "H2": 89.0, "H3": 104.0, "H4": 125.0, "H5": 150.0}
    expected = {"cola": 1, "TS": 2, "H1": 3, "H2": 3, "H3": 4, "H4": 5, "H5": 5}
    h = normalize_wind_hazard_absolute(np.array(list(centers.values())), outer_wind_kt=KT_12MS)
    cls = classify_itchi_to_siat_levels(h)
    assert dict(zip(centers, cls.astype(int).tolist(), strict=True)) == expected


def test_saffir_simpson_constants_consistent() -> None:
    winds = [v for v, _ in SAFFIR_SIMPSON_HAZARD_NODES]
    for key in ("TS", "H1", "H3", "H4", "H5"):
        assert SAFFIR_SIMPSON_KT[key] in winds


def test_absolute_rejects_bad_nodes() -> None:
    with pytest.raises(ValueError):
        normalize_wind_hazard_absolute(np.array([50.0]), 20.0, nodes=((34.0, 0.2), (30.0, 0.4)))
    with pytest.raises(ValueError):
        normalize_wind_hazard_absolute(np.array([50.0]), 20.0, nodes=((34.0, 0.5), (64.0, 0.4)))


def test_absolute_scale_does_not_depend_on_vmax() -> None:
    # Mismo viento local en una tormenta de 45 kt y en una de 140 kt
    r = np.linspace(5.0, 600.0, 400)
    out = {}
    for vmax in (45.0, 140.0):
        out[vmax] = {
            scale: compute_w06_wind_hazard_with_tail(
                radius_km=r, direct_radius_km=150.0, attribution_radius_km=600.0,
                vmax_kt=vmax, lat=20.0, rmw_km=30.0, tail_end_wind_kt=KT_12MS,
                hazard_scale=scale,
            )
            for scale in ("vmax", "saffir_simpson")
        }
    for vmax, res in out.items():
        ss = res["saffir_simpson"]
        expected = normalize_wind_hazard_absolute(ss["wind_kt"], ss["outer_wind_kt"])
        outside_eye = ss["wind_kt"] < 0.999 * vmax
        outside_eye &= r > 30.0
        np.testing.assert_allclose(ss["H_W"][outside_eye], expected[outside_eye])
        # ojo: dentro del RMW el peligro es el de Vmax
        h_vmax = normalize_wind_hazard_absolute(np.array([vmax]), KT_12MS)[0]
        np.testing.assert_allclose(ss["H_W"][r <= 30.0], h_vmax)
    # v0.2: la tormenta de 45 kt llega a H_W ≈ 1; con Saffir-Simpson se queda en TT (< 0.4)
    assert np.nanmax(out[45.0]["vmax"]["H_W"]) > 0.95          # clase 5 (malla radial finita)
    assert np.nanmax(out[45.0]["saffir_simpson"]["H_W"]) < 0.4
    assert np.nanmax(out[140.0]["saffir_simpson"]["H_W"]) == pytest.approx(1.0)


def test_pipeline_saffir_simpson_from_overlay() -> None:
    lon, lat = np.meshgrid(np.arange(-110.0, -80.0, 0.2), np.arange(5.0, 35.0, 0.2))
    rng = np.random.default_rng(1)
    p6 = rng.gamma(0.7, 12.0, lon.shape)
    r24 = p6 + rng.gamma(0.7, 25.0, lon.shape)
    a72 = rng.gamma(0.8, 30.0, lon.shape)
    case = dict(center_lon=-95.0, center_lat=20.0, vmax_kt=50.0, rmw_km=40.0,
                r34_by_quadrant={"RNE": 90, "RSE": 80, "RSW": 50, "RNW": 60}, r34_unit="nm")
    res = {}
    for name in ("v020", "w12", "ss"):
        settings = itchi_v02_settings_from_config(load_experiment_config(V021 / f"{name}.yaml"))
        res[name] = compute_itchi_snapshot_v02(
            p6, r24, a72, quantiles_6h=Q6, quantiles_24h=Q24, quantiles_72h=Q72,
            lon=lon, lat=lat, **case, **settings,
        )
    assert res["ss"]["wind_hazard_scale"] == "saffir_simpson"
    assert res["v020"]["wind_hazard_scale"] == "vmax"
    # Tormenta tropical de 50 kt: v0.2 alcanza H_W ≈ 1; Saffir-Simpson no pasa de la clase 2
    assert np.nanmax(res["v020"]["H_W"]) > 0.95              # clase 5 en el RMW
    assert 0.2 < np.nanmax(res["ss"]["H_W"]) < 0.4
    # El corte en 12 m/s solo reduce el área de la cola
    assert (np.asarray(res["w12"]["H_W"]) > 0).sum() < (np.asarray(res["v020"]["H_W"]) > 0).sum()
    np.testing.assert_array_equal(np.asarray(res["w12"]["H_W"]) > 0, np.asarray(res["ss"]["H_W"]) > 0)
    # El ojo (centro del ciclón) conserva el peligro de Vmax
    j, i = np.unravel_index(np.argmin(np.asarray(res["ss"]["radius_km"])), lon.shape)
    assert res["ss"]["H_W"][j, i] == pytest.approx(0.2 + 0.2 * (50.0 - 34.0) / 30.0)
    # Las regiones no cambian entre experimentos
    np.testing.assert_array_equal(res["v020"]["M_indirect"], res["ss"]["M_indirect"])


# ------------------------------------------------------------------ lluvia P95


def test_p95_matrix_definition() -> None:
    assert PRECIP_MATRIX_V02_P95["p6_nodes"] == ("Q75", "Q90", "Q95")
    assert PRECIP_MATRIX_V02_P95["base"] == (0.0, 0.0, 0.25, 1.0)
    assert PRECIP_MATRIX_V02_P95["escalated"] == (0.0, 0.0, 0.8, 1.0)
    assert PRECIP_MATRIX_V02_P95["escalation"] == PRECIP_MATRIX_V02["escalation"]
    assert PRECIP_MATRIX_V02["p6_nodes"][-1] == "Q99"          # el original no cambia


def test_p95_hazard_values_and_dominance() -> None:
    p6 = np.linspace(0.0, 60.0, 601)
    zero = {"R24": np.zeros_like(p6), "A72": np.zeros_like(p6)}
    sat = {"R24": np.full_like(p6, 60.0), "A72": np.full_like(p6, 40.0)}   # R24 >= P95 y A72 >= P75
    qs = {"R24": Q24, "A72": Q72}
    for esc, h90 in ((zero, 0.25), (sat, 0.8)):
        h99 = compute_precipitation_matrix_hazard(p6, Q6, esc, qs, matrix=PRECIP_MATRIX_V02)
        h95 = compute_precipitation_matrix_hazard(p6, Q6, esc, qs, matrix=PRECIP_MATRIX_V02_P95)
        assert np.all(h95 >= h99 - 1e-12)                       # P95 nunca baja el peligro
        assert np.all(np.diff(h95) >= -1e-12)
        np.testing.assert_allclose(np.interp([18.0, 28.0, 50.0], p6, h95), [h90, 1.0, 1.0], atol=1e-9)
        same = p6 <= Q6["Q75"]
        np.testing.assert_allclose(h95[same], h99[same])


def test_p95_overlay_settings() -> None:
    settings = itchi_v02_settings_from_config(load_experiment_config(V021 / "p95.yaml"))
    assert settings["precip_matrix"]["p6_nodes"] == PRECIP_MATRIX_V02_P95["p6_nodes"]
    assert settings["precip_matrix"]["base"] == PRECIP_MATRIX_V02_P95["base"]
    assert settings["precip_matrix"]["escalated"] == PRECIP_MATRIX_V02_P95["escalated"]
    assert settings["wind_hazard_scale"] == "vmax"
    both = itchi_v02_settings_from_config(load_experiment_config(V021 / "p95_ss.yaml"))
    assert both["precip_matrix"]["p6_nodes"][-1] == "Q95"
    assert both["wind_hazard_scale"] == "saffir_simpson"
    assert both["wind_tail_end_kt"] == pytest.approx(KT_12MS)
    assert both["wind_absolute_nodes"] == SAFFIR_SIMPSON_HAZARD_NODES
