"""
Tests for scripts/experimento_b1.py: scores, M3 calibration and an end-to-end run on
synthetic municipality–cyclone pairs (percentiles patched, no climatology files).
"""

from __future__ import annotations

import importlib.util
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itchi.precipitation import build_precipitation_class_table

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "experimento_b1.py"
_spec = importlib.util.spec_from_file_location("experimento_b1", SCRIPT)
b1 = importlib.util.module_from_spec(_spec)
sys.modules["experimento_b1"] = b1          # dataclasses needs the module registered
_spec.loader.exec_module(b1)

QS = {
    6: {"Q75": 8.0, "Q90": 18.0, "Q95": 28.0, "Q99": 50.0, "Q99.9": 80.0},
    24: {"Q75": 20.0, "Q90": 40.0, "Q95": 55.0, "Q99": 90.0, "Q99.9": 140.0},
    72: {"Q75": 30.0, "Q90": 60.0, "Q95": 80.0, "Q99": 130.0, "Q99.9": 200.0},
}


# ------------------------------------------------------------------ puntajes


def test_auc_matches_pairwise_definition() -> None:
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, 300).astype(float)
    o = rng.random(300) < 0.3
    brute = np.mean([(a > b) + 0.5 * (a == b) for a, b in itertools.product(s[o], s[~o])])
    assert b1.auc(s, o) == pytest.approx(brute)


def test_scores_and_sedi() -> None:
    s = b1.scores_from_counts(30.0, 70.0, 10.0, 890.0)
    assert s["POD"] == pytest.approx(0.75) and s["POFD"] == pytest.approx(70 / 960)
    assert s["PSS"] == pytest.approx(0.75 - 70 / 960)
    assert s["sesgo"] == pytest.approx(100 / 40)
    assert b1.sedi(0.5, 0.5) == pytest.approx(0.0)
    assert b1.sedi(0.9, 0.1) > 0.0 > b1.sedi(0.1, 0.9)


def test_pod_at_pofd_on_own_curve() -> None:
    rng = np.random.default_rng(1)
    s = rng.integers(0, 6, 2000).astype(float)
    o = rng.random(2000) < 0.1 + 0.05 * s
    for k in range(1, 6):
        c = b1.contingency(s >= k, o)
        assert b1.pod_at_pofd(s, o, c["POFD"]) == pytest.approx(c["POD"])


def test_bootstrap_counts_sum_to_table() -> None:
    rng = np.random.default_rng(2)
    pred, obs = rng.random(500) < 0.3, rng.random(500) < 0.2
    g = rng.integers(0, 17, 500)
    counts = b1.bootstrap_counts(pred, obs, g)
    c = b1.contingency(pred, obs)
    np.testing.assert_allclose(counts.sum(0), [c["a"], c["b"], c["c"], c["d"]])
    w = b1.bootstrap_weights(counts.shape[0], 50, 0)
    assert w.shape == (50, counts.shape[0]) and (w.sum(1) == counts.shape[0]).all()


def test_wind_hazard_matches_library_for_one_storm() -> None:
    from itchi.wind import normalize_wind_hazard_with_tail
    v = np.linspace(0.0, 100.0, 201)
    for v_out_ms in (2.0, 12.0):
        ours = b1.wind_hazard(v, np.full(v.size, 100.0), 0.2, v_out_ms)
        lib = normalize_wind_hazard_with_tail(v, 100.0, v >= 34.0, v_out_ms / b1.KNOT_TO_MS, h34=0.2)
        np.testing.assert_allclose(ours, lib, atol=1e-12)


# ------------------------------------------------------------------ esquemas y M3


def test_fixed_schemes_are_valid_tables() -> None:
    for name, spec in b1.precip_schemes().items():
        if spec.get("kind") == "class_table":
            build_precipitation_class_table(spec)
        else:
            assert name == "v0.2"


def test_fit_m3_table_properties() -> None:
    rng = np.random.default_rng(3)
    cells = rng.integers(0, 4, (60000, 3))
    logit = -5.0 + 0.8 * cells[:, 0] + 0.6 * (cells[:, 1] == 3) * (cells[:, 2] >= 1)
    d = rng.random(len(cells)) < 1.0 / (1.0 + np.exp(-logit))
    h = b1.fit_m3_table(cells, d)
    assert h.shape == (4, 4, 4)
    for axis in range(3):
        assert (np.diff(h, axis=axis) >= 0.0).all()
    assert (h[3] == 1.0).all() and (h[0] == 0.0).all() and h[1, 0, 0] == 0.0
    assert (h >= np.array([0.0, 0.0, 0.5, 1.0])[:, None, None]).all()
    assert h[2, 3, 3] > h[2, 0, 0]          # la escalada sube P90-P95 con R24 y A72 altos
    build_precipitation_class_table(b1.m3_spec(h))


# ------------------------------------------------------------------ corrida sintética


def _synthetic(tmp: Path, n_pairs: int = 3000, with_steps: bool = False) -> b1.ConfigB1:
    rng = np.random.default_rng(4)
    p6 = rng.gamma(0.8, 12.0, n_pairs)
    r24 = p6 + rng.gamma(0.8, 20.0, n_pairs)
    a72 = rng.gamma(0.8, 30.0, n_pairs)
    risk = 0.01 + 0.08 * (p6 > 28) + 0.05 * ((r24 > 55) & (a72 > 30))
    pares = pd.DataFrame({
        "CVEGEO": [f"{i:05d}" for i in range(n_pairs)], "CT": rng.integers(0, 40, n_pairs).astype(str),
        "t_pico": pd.Timestamp("2005-06-01") + pd.to_timedelta(rng.integers(0, 20, n_pairs) * 365, "D"),
        "nivel_pico": 0, "P_evento": p6, "R24_en_pico6": r24, "A72_en_pico6": a72,
        "dist_pico_km": rng.uniform(0, 700, n_pairs), "vmax_pico_kt": rng.uniform(25, 120, n_pairs),
        "lat_pmax": 20.0, "lon_pmax": -100.0, "D": (rng.random(n_pairs) < risk).astype(int),
    })
    pares.to_csv(tmp / "pares.csv.gz", index=False)
    if with_steps:
        reps = pares.loc[pares.index.repeat(3)].reset_index(drop=True)
        scale = np.tile([0.5, 1.0, 0.8], n_pairs)
        pd.DataFrame({
            "CVEGEO": reps["CVEGEO"], "CT": reps["CT"], "time": "2005-06-01",
            "dist_km": reps["dist_pico_km"] * (2.0 - scale), "r_w06_km": 800.0, "vmax_kt": reps["vmax_pico_kt"],
            "lat_ct": 20.0, "P6_max": reps["P_evento"] * scale, "R24_en_pico6": reps["R24_en_pico6"],
            "A72_en_pico6": reps["A72_en_pico6"], "lat_pmax": 20.0, "lon_pmax": -100.0,
        }).to_csv(tmp / "pasos.csv.gz", index=False)
    return b1.ConfigB1(diag_dir=str(tmp), pares_file="pares.csv.gz", pasos_file="pasos.csv.gz",
                       out_dir=str(tmp / "out"), n_boot=50)


@pytest.mark.parametrize("with_steps", [False, True])
def test_run_end_to_end_synthetic(tmp_path: Path, with_steps: bool) -> None:
    cfg = _synthetic(tmp_path, with_steps=with_steps)
    original = b1.percentiles_at
    b1.percentiles_at = lambda cfg, d, lat, lon: {k: np.full(lat.shape, v) for k, v in QS[d].items()}
    try:
        out = b1.run(cfg)
    finally:
        b1.percentiles_at = original
    for f in ("celdas_P6_R24_A72.csv", "lluvia_candidatos.csv", "diseno_2x2.csv", "tabla_m3.yaml", "resumen_b1.md"):
        assert (tmp_path / "out" / f).exists()
    dis = out["diseno"].set_index(["lluvia", "viento"])
    ref = dis.loc[("v0.2", "W0 cola hasta R_W06")]
    assert ref["dPSS>=5"] == pytest.approx(0.0) and abs(ref["dPOD@POFD>=5"]) < 1e-9
    assert len(dis) == 5 * 4
    lluvia = out["lluvia"].set_index("lluvia")
    # P95 como máximo: más pares en clase 5 que v0.2
    assert lluvia.loc["M0", "POFD>=5"] >= lluvia.loc["v0.2", "POFD>=5"]
    assert out["celdas"]["n"].sum() == len(out["pares"])
