"""
ITCHI-CORE · Prueba rápida del acumulado de 6 h MSWEP V3 (pocos días)

Procesa unos días (por defecto Cristóbal, 1–4 jun 2020) sin escribir el
producto anual y verifica:

  1. ventanas completas, NaN y rango de valores;
  2. ubicación y fecha del máximo de 6 h;
  3. figura de control en ../outputs/diagnostics/.

Ejecutar desde scripts/:
    python prueba_rapida_6h.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mswep_v3_6h_accum import (
    Config,
    accumulate_windows,
    build_index,
    check_out_dir,
    detect_time_label,
    make_reader,
    region_indices,
)

# ============================================================================
# CONFIGURACIÓN DE LA PRUEBA
# ============================================================================

DAYS = ("2020-06-01", "2020-06-04")          # días completos (UTC) a evaluar
TIME_LABEL = "start"   # PROVISIONAL: MSWEP V3 no trae time.bounds; verificar antes de usar con AIFS/eventos
FIG = "../outputs/diagnostics/prueba_mswep_v3_6h.png"


def main() -> None:

    cfg = Config(workers=1)
    check_out_dir(cfg)

    print("\n[1/3] Índice")
    index = build_index(cfg)

    label = TIME_LABEL if TIME_LABEL != "auto" else detect_time_label(index["path"].iloc[0])
    if label is None:
        raise SystemExit("Sin time.bounds: fijar TIME_LABEL = 'start' o 'end'.")
    print(f"time_label = {label}")

    d0, d1 = pd.Timestamp(DAYS[0]), pd.Timestamp(DAYS[1])
    ends = pd.date_range(d0 + pd.Timedelta(hours=6), d1 + pd.Timedelta(days=1), freq="6h")

    sub = index[(index["time"] >= d0 - pd.Timedelta(hours=1)) & (index["time"] <= ends[-1])]
    if sub.empty:
        raise SystemExit(f"El índice no tiene horas entre {d0} y {ends[-1]}.")
    lookup = {t: (p, int(i)) for t, p, i in zip(sub["time"], sub["path"], sub["tidx"])}

    print("\n[2/3] Acumulación")
    reg = region_indices(sub["path"].iloc[0], cfg)
    out, missing = accumulate_windows(
        ends, lookup, make_reader(cfg, reg), (reg["lat"].size, reg["lon"].size), label, progress=True,
    )

    valid = out[np.isfinite(out)]
    print(f"Ventanas: {len(ends)} | faltantes: {len(missing)} | malla: {out.shape[1]} x {out.shape[2]}")
    print(f"NaN: {np.isnan(out).mean():.2%} | máx: {valid.max():.1f} mm/6h | "
          f"ventanas-píxel >= 1 mm: {(valid >= 1).mean():.1%}")
    print(f"lat {reg['lat'][0]:.2f}..{reg['lat'][-1]:.2f}  lon {reg['lon'][0]:.2f}..{reg['lon'][-1]:.2f}")

    k, j, i = np.unravel_index(np.nanargmax(out), out.shape)
    print(f"Máximo de 6 h: {out[k, j, i]:.1f} mm en ventana que termina {ends[k]:%Y-%m-%d %H} UTC, "
          f"lat {reg['lat'][j]:.2f}, lon {reg['lon'][i]:.2f}")

    print("\n[3/3] Figura")
    lon, lat = reg["lon"], reg["lat"]
    panels = [("V3: máximo de 6 h (mm)", np.nanmax(out, axis=0)),
              ("V3: total del periodo (mm)", np.nansum(out, axis=0))]

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.2 * len(panels)), constrained_layout=True)
    for ax, (title, field) in zip(np.atleast_1d(axes), panels):
        m = ax.pcolormesh(lon, lat, field, cmap="viridis", vmin=0, vmax=np.nanpercentile(field, 99.5))
        ax.plot(reg["lon"][i], reg["lat"][j], "r+", ms=10)          # posición del máximo de 6 h
        ax.set_title(title)
        ax.set_aspect("equal")
        fig.colorbar(m, ax=ax, shrink=0.9)

    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=120)
    print(f"Figura: {FIG}")


if __name__ == "__main__":
    main()
