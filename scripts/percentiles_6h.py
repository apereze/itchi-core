"""
ITCHI-CORE · Percentiles de precipitación de 6 h MSWEP V3

Mismo método que scripts/percentiles.py (se reutilizan sus funciones),
aplicado a los acumulados de 6 h generados por mswep_v3_6h_accum.py.

Productos (mismos nombres que la climatología diaria V2.8):

    p_wet             percentiles sobre ventanas húmedas (P >= 1 mm (6 h)-1),
                      con filtro de suficiencia muestral
    p_wet_raw         idem, sin filtro
    n_wet             número de ventanas húmedas
    frac_above_p_wet  fracción de la precipitación húmeda aportada por
                      ventanas que superan cada percentil
    p_all             percentiles sobre todas las ventanas

Nuevos:

    n_valid           número de ventanas válidas
    wet_freq          n_wet / n_valid
    freq_above_abs    fracción de ventanas que superan umbrales absolutos
                      (10, 20, 35, 60 mm (6 h)-1)
    *_hour            (opcional) lo anterior estratificado por hora final
                      de la ventana: 00, 06, 12, 18 UTC

Nota: P75 sobre todas las ventanas es ~0 donde la frecuencia húmeda es
< 25 %. P75 solo tiene sentido como p_wet, idealmente con un piso absoluto.

Ejecutar desde scripts/:
    python percentiles_6h.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import xarray as xr
from dask.diagnostics import ProgressBar

# Funciones de la climatología diaria: mismo método de cuantil y mismo
# criterio de suficiencia, para que V2.8 diario y V3 6 h sean comparables.
from percentiles import (
    all_day_percentiles,
    contribution_above,
    wet_day_percentiles,
)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================


@dataclass
class Config:

    directory: str = "D:/MSWEP_V316_6H"

    start_year: int = 2000
    end_year: int = 2025

    # Ventana húmeda
    wet: float = 1.0  # mm (6 h)-1

    qs: tuple = (0.75, 0.90, 0.95, 0.99, 0.999)

    # Excedencias esperadas mínimas: n_wet >= k_exc / (1 - q)
    #   P75 -> 40, P90 -> 100, P99 -> 1000, P99.9 -> 10000 ventanas húmedas
    k_exc: int = 10

    # Hyndman & Fan tipo 8, igual que percentiles.py
    qmethod: str = "median_unbiased"

    # Umbrales absolutos de referencia (mm (6 h)-1)
    abs_thresholds: tuple = (10.0, 20.0, 35.0, 60.0)

    # Estratificar por hora final de ventana (ciclo diurno)
    by_hour: bool = False

    chunks: dict = field(default_factory=lambda: {"time": -1, "lat": 25, "lon": 25})

    out_file: str = "../data/processed/mswep_v3_6h_percentiles_may_nov.nc"


# ============================================================================
# APERTURA
# ============================================================================


def open_precip_6h(cfg: Config) -> xr.DataArray:

    files = [
        os.path.join(cfg.directory, f"mswep_v3_6h_{y}.nc")
        for y in range(cfg.start_year, cfg.end_year + 1)
    ]

    missing = [os.path.basename(f) for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Faltan acumulados de 6 h: {missing}")

    print(f"\nArchivos anuales: {len(files)}")

    ds = xr.open_mfdataset(files, combine="by_coords", chunks=cfg.chunks, parallel=False)
    pr = ds["precipitation"].chunk({"time": -1})

    print("Dimensiones:", dict(pr.sizes))

    return pr


# ============================================================================
# DIAGNÓSTICOS ADICIONALES
# ============================================================================


def frequency_fields(pr: xr.DataArray, cfg: Config) -> xr.Dataset:

    valid = pr.notnull()
    n_valid = valid.sum("time")

    wet_freq = (pr >= cfg.wet).sum("time") / n_valid.where(n_valid > 0)

    freq_abs = xr.concat(
        [(pr > t).sum("time") / n_valid.where(n_valid > 0) for t in cfg.abs_thresholds],
        dim=xr.DataArray(list(cfg.abs_thresholds), dims="threshold", name="threshold"),
    )

    return xr.Dataset({"n_valid": n_valid, "wet_freq": wet_freq, "freq_above_abs": freq_abs})


def percentile_fields(pr: xr.DataArray, cfg: Config) -> xr.Dataset:

    q_raw, q_wet, n_wet, wet = wet_day_percentiles(pr, cfg)

    return xr.Dataset({
        "p_wet": q_wet,
        "p_wet_raw": q_raw,
        "n_wet": n_wet,
        "frac_above_p_wet": contribution_above(wet, q_raw),
        "p_all": all_day_percentiles(pr, cfg),
    })


def by_hour_fields(pr: xr.DataArray, cfg: Config) -> xr.Dataset:

    hours = (0, 6, 12, 18)
    parts = []

    for h in hours:
        sub = pr.sel(time=pr["time"].dt.hour == h)
        f = percentile_fields(sub, cfg)[["p_wet", "p_wet_raw", "n_wet", "p_all"]]
        parts.append(f)

    out = xr.concat(parts, dim=xr.DataArray(list(hours), dims="hour", name="hour"))
    out["hour"].attrs["long_name"] = "end hour (UTC) of the 6-h window"

    return out.rename({v: f"{v}_hour" for v in out.data_vars})


# ============================================================================
# METADATOS
# ============================================================================


def add_metadata(out: xr.Dataset, cfg: Config) -> xr.Dataset:

    for v in [v for v in out.data_vars if v.startswith(("p_wet", "p_all"))]:
        out[v].attrs.update(units="mm (6 h)-1", long_name=f"6-h precipitation percentile ({v})")

    for v in [v for v in out.data_vars if v.startswith(("n_wet", "n_valid"))]:
        out[v].attrs.update(units="1", long_name=f"number of 6-h windows ({v})")

    out["frac_above_p_wet"].attrs.update(
        units="1", long_name="Fraction of wet-window precipitation from windows exceeding the percentile")
    out["wet_freq"].attrs.update(units="1", long_name="wet-window frequency")
    out["freq_above_abs"].attrs.update(units="1", long_name="fraction of valid windows above threshold")
    out["threshold"].attrs.update(units="mm (6 h)-1")

    out.attrs.update(
        framework="ITCHI-CORE",
        source="MSWEP V3 hourly -> 6-h accumulations (scripts/mswep_v3_6h_accum.py)",
        period=f"{cfg.start_year}-{cfg.end_year}",
        months="5,6,7,8,9,10,11",
        accumulation_window="(T-6h, T], T in 00/06/12/18 UTC",
        wet_window_threshold_mm=cfg.wet,
        quantiles=",".join(map(str, cfg.qs)),
        quantile_method=cfg.qmethod,
        min_expected_exceedances=cfg.k_exc,
        description="6-h MSWEP V3 precipitation percentiles for May-November.",
    )

    return out


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================


def run(cfg: Config = Config()) -> xr.Dataset:

    t0 = time.time()

    print("\n" + "=" * 75)
    print("ITCHI-CORE · MSWEP V3 6-h precipitation percentiles")
    print("=" * 75)

    print("\n[1/3] Lectura de acumulados de 6 h")
    pr = open_precip_6h(cfg)

    print("\n[2/3] Construcción del grafo de cálculo")
    out = xr.merge([percentile_fields(pr, cfg), frequency_fields(pr, cfg)])
    if cfg.by_hour:
        out = xr.merge([out, by_hour_fields(pr, cfg)])
    out = add_metadata(out, cfg)

    print("\n[3/3] Ejecutando cálculo Dask y escribiendo...")
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out_file)), exist_ok=True)

    encoding = {v: {"zlib": True, "complevel": 4, "dtype": "float32"} for v in out.data_vars}
    for v in [v for v in out.data_vars if v.startswith(("n_wet", "n_valid"))]:
        encoding[v] = {"zlib": True, "complevel": 4, "dtype": "int32"}  # int16 se desborda con > 32767 ventanas

    with ProgressBar():
        out.to_netcdf(cfg.out_file, encoding=encoding)

    print("\n" + "=" * 75)
    print(f"PROCESO TERMINADO · {(time.time() - t0) / 60:.1f} minutos · {cfg.out_file}")
    print("=" * 75)

    return out


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    # Prueba con el año generado por mswep_v3_6h_accum.py; luego ampliar el periodo.
    out = run(Config(start_year=2020, end_year=2020, by_hour=True))
