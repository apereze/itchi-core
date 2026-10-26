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

Multiduración (duration_h = 6, 24, 72):

    R_d(T) = suma de los acumulados de 6 h en (T-d, T], d múltiplo de 6 h,
    calculada al vuelo dentro de cada temporada (no cruza dic -> may).
    Una ventana con algún paso de 6 h faltante -> NaN. Al inicio de cada
    temporada las primeras d/6 - 1 ventanas son NaN.

    Ventana húmeda: R_d >= wet (1 mm por ventana, igual que 6 h).

    El antecedente exclusivo A_d(T) = R_d(T - 6 h), en (T-d-6h, T-6h], tiene
    la misma distribución que R_d: usa este mismo archivo de percentiles.

    Las ventanas móviles se traslapan (d/6 por paso), así que el número
    efectivo de muestras independientes es ~n/(d/6). El filtro de
    suficiencia se aplica sobre ese número efectivo:
        n_wet >= k_exc * (d/6) / (1 - q)
    En 72 h, P99.9 queda fuera del filtro casi en todo el dominio
    (p_wet = NaN); p_wet_raw lo conserva pero no es robusto.

Salidas:
    ../data/processed/mswep_v3_{d}h_percentiles_may_nov.nc
    (d = 6 conserva el nombre original)

Ejecutar desde scripts/:
    python percentiles_6h.py                      # 24 y 72 h
    python percentiles_6h.py --durations 6 --by-hour
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass, field, replace

import numpy as np
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

    # Duración de la acumulación (h): múltiplo de 6
    duration_h: int = 6

    # Ventana húmeda
    wet: float = 1.0  # mm por ventana de duration_h

    qs: tuple = (0.75, 0.90, 0.95, 0.99, 0.999)

    # Excedencias esperadas mínimas: n_wet >= k_exc / (1 - q)
    #   P75 -> 40, P90 -> 100, P99 -> 1000, P99.9 -> 10000 ventanas húmedas
    k_exc: int = 10

    # Hyndman & Fan tipo 8, igual que percentiles.py
    qmethod: str = "median_unbiased"

    # Umbrales absolutos de referencia (mm por ventana); None -> ABS_THRESHOLDS[duration_h]
    abs_thresholds: tuple | None = None

    # Estratificar por hora final de ventana (ciclo diurno); solo útil en 6 h
    by_hour: bool = False

    chunks: dict = field(default_factory=lambda: {"time": -1, "lat": 25, "lon": 25})

    # None -> ../data/processed/mswep_v3_{duration_h}h_percentiles_may_nov.nc
    out_file: str | None = None

    def __post_init__(self):
        if self.duration_h % 6 or self.duration_h < 6:
            raise ValueError(f"duration_h debe ser múltiplo de 6 h: {self.duration_h}")
        if self.abs_thresholds is None:
            self.abs_thresholds = ABS_THRESHOLDS.get(self.duration_h, ABS_THRESHOLDS[6])
        if self.out_file is None:
            self.out_file = f"../data/processed/mswep_v3_{self.duration_h}h_percentiles_may_nov.nc"

    @property
    def n_overlap(self) -> int:
        # Ventanas de 6 h por acumulado: factor de traslape entre ventanas consecutivas
        return self.duration_h // 6


# Umbrales absolutos por duración (mm por ventana)
ABS_THRESHOLDS = {
    6: (10.0, 20.0, 35.0, 60.0),
    24: (25.0, 50.0, 100.0, 150.0),
    72: (50.0, 100.0, 200.0, 300.0),
}


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


def rolling_accum(pr: xr.DataArray, hours: int) -> xr.DataArray:
    """
    R_d(T): suma de n = d/6 acumulados de 6 h en (T-d, T].
    NaN si falta algún paso o si la ventana cruza el hueco dic -> may entre temporadas.
    """

    n = hours // 6
    if n == 1:
        return pr

    acc = pr.rolling(time=n, min_periods=n).sum()

    # Contigüidad: el paso más antiguo de la ventana está exactamente (n-1)·6 h antes
    tv = pr["time"].values
    ok = np.zeros(tv.size, dtype=bool)
    ok[n - 1:] = (tv[n - 1:] - tv[:tv.size - (n - 1)]) == np.timedelta64(6 * (n - 1), "h")
    contigua = xr.DataArray(ok, dims="time", coords={"time": pr["time"]})

    return acc.where(contigua)


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

    # Suficiencia sobre el número efectivo de ventanas independientes (~n / (d/6))
    q_raw, q_wet, n_wet, wet = wet_day_percentiles(pr, replace(cfg, k_exc=cfg.k_exc * cfg.n_overlap))

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

    d = cfg.duration_h

    for v in [v for v in out.data_vars if v.startswith(("p_wet", "p_all"))]:
        out[v].attrs.update(units=f"mm ({d} h)-1", long_name=f"{d}-h precipitation percentile ({v})")

    for v in [v for v in out.data_vars if v.startswith(("n_wet", "n_valid"))]:
        out[v].attrs.update(units="1", long_name=f"number of {d}-h windows ({v})")

    out["frac_above_p_wet"].attrs.update(
        units="1", long_name="Fraction of wet-window precipitation from windows exceeding the percentile")
    out["wet_freq"].attrs.update(units="1", long_name="wet-window frequency")
    out["freq_above_abs"].attrs.update(units="1", long_name="fraction of valid windows above threshold")
    out["threshold"].attrs.update(units=f"mm ({d} h)-1")

    out.attrs.update(
        framework="ITCHI-CORE",
        source="MSWEP V3 hourly -> 6-h accumulations (scripts/mswep_v3_6h_accum.py)"
               + ("" if d == 6 else f" -> {d}-h running sums of 6-h accumulations"),
        period=f"{cfg.start_year}-{cfg.end_year}",
        months="5,6,7,8,9,10,11",
        duration_h=d,
        accumulation_window=f"(T-{d}h, T], T in 00/06/12/18 UTC",
        antecedent_note=("" if d == 6 else
                         f"exclusive antecedent A{d}(T) = R{d}(T-6h) uses these same percentiles"),
        wet_window_threshold_mm=cfg.wet,
        quantiles=",".join(map(str, cfg.qs)),
        quantile_method=cfg.qmethod,
        min_expected_exceedances=cfg.k_exc,
        overlap_factor=cfg.n_overlap,
        sufficiency_rule="n_wet >= k_exc * overlap_factor / (1 - q)",
        description=f"{d}-h MSWEP V3 precipitation percentiles for May-November.",
    )

    return out


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================


def run(cfg: Config = Config()) -> xr.Dataset:

    t0 = time.time()

    print("\n" + "=" * 75)
    print(f"ITCHI-CORE · MSWEP V3 {cfg.duration_h}-h precipitation percentiles")
    print("=" * 75)

    print(f"\n[1/3] Lectura de acumulados de 6 h -> {cfg.duration_h} h")
    pr = rolling_accum(open_precip_6h(cfg), cfg.duration_h)

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

    ap = argparse.ArgumentParser(description="Percentiles MSWEP V3 de 6 h y acumulados móviles")
    ap.add_argument("--durations", type=int, nargs="+", default=[24, 72],
                    help="duraciones en h, múltiplos de 6 (6 ya existe con by_hour=True)")
    ap.add_argument("--by-hour", action="store_true",
                    help="estratificar por hora final (duplica el tiempo; solo útil en 6 h)")
    args = ap.parse_args()

    # Climatología completa 2000–2025, una corrida por duración
    for d in args.durations:
        run(Config(start_year=2000, end_year=2025, duration_h=d, by_hour=args.by_hour))
