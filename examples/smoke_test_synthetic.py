"""
Synthetic smoke test for ITCHI.

Builds a synthetic grid, 6-h precipitation (P6, R24, A72), percentile fields
and a hurricane, computes ITCHI with pipeline.compute_itchi_snapshot (scheme
of configs/default.yaml) and checks the basic properties of the result.

Run from the repository root:

    python examples/smoke_test_synthetic.py
    python examples/smoke_test_synthetic.py --plot --figure-path outputs/figures/smoke_test_synthetic.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from itchi.config import itchi_settings_from_config, load_default_config
from itchi.pipeline import compute_itchi_snapshot
from itchi.plotting import classify_itchi_to_siat_levels


def synthetic_case(seed: int = 0) -> dict:
    """Grid, rain fields, percentiles and a 100-kt hurricane (category 3)."""
    lon, lat = np.meshgrid(np.arange(-110.0, -85.0, 0.1), np.arange(10.0, 30.0, 0.1))
    rng = np.random.default_rng(seed)
    center = (-100.0, 18.0)
    d2 = (lon - center[0]) ** 2 + (lat - center[1]) ** 2
    p6 = rng.gamma(0.6, 4.0, lon.shape) + 45.0 * np.exp(-d2 / 1.0)
    r24 = 3.0 * p6
    a72 = rng.gamma(0.8, 20.0, lon.shape)
    q6 = {"Q75": np.full(lon.shape, 8.0), "Q90": np.full(lon.shape, 18.0), "Q95": np.full(lon.shape, 28.0)}
    q24 = {"Q90": np.full(lon.shape, 40.0), "Q95": np.full(lon.shape, 55.0)}
    q72 = {"Q75": np.full(lon.shape, 30.0)}
    storm = dict(center_lon=center[0], center_lat=center[1], vmax_kt=100.0, rmw_km=25.0,
                 r34_by_quadrant={"RNE": 120, "RSE": 100, "RSW": 70, "RNW": 90}, r34_unit="nm",
                 rocloud_by_quadrant={"RNE": 900.0, "RSE": 700.0, "RSW": None, "RNW": 400.0})
    return dict(p6=p6, r24=r24, a72=a72, quantiles_6h=q6, quantiles_24h=q24, quantiles_72h=q72,
                lon=lon, lat=lat, **storm)


def run() -> dict:
    case = synthetic_case()
    p6, r24, a72 = case.pop("p6"), case.pop("r24"), case.pop("a72")
    settings = itchi_settings_from_config(load_default_config())
    res = compute_itchi_snapshot(p6, r24, a72, **case, **settings, run_quality_control=True)

    itchi = np.asarray(res["ITCHI"])
    assert np.nanmin(itchi) >= 0.0 and np.nanmax(itchi) <= 1.0
    assert (itchi[np.asarray(res["M_exterior"])] == 0.0).all()
    assert float(np.max(res["H_W"])) > 0.8            # huracán categoría 3: clase 5 por viento
    classes = classify_itchi_to_siat_levels(itchi)
    counts = {k: int((classes == k).sum()) for k in range(1, 6)}
    print(f"ITCHI {res['itchi_version']}: max = {itchi.max():.3f}; celdas por clase: {counts}")
    print("R_attr por cuadrante (km):", {q: round(v) for q, v in res["R_attr_by_quadrant"].items()})
    return res


def plot(res: dict, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for ax, key in zip(axes, ("H_P", "H_W", "ITCHI"), strict=True):
        im = ax.imshow(np.asarray(res[key]), origin="lower", vmin=0, vmax=1, cmap="viridis")
        ax.set_title(key)
        fig.colorbar(im, ax=ax, shrink=0.8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    print(f"figura: {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Synthetic smoke test for ITCHI")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--figure-path", type=Path, default=Path("outputs/figures/smoke_test_synthetic.png"))
    args = ap.parse_args()
    result = run()
    if args.plot:
        plot(result, args.figure_path)
