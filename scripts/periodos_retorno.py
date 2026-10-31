"""
ITCHI-CORE · Periodos de retorno de la lluvia (climatología de rareza, paso 1)

Para cada píxel y duración d (6, 24, 72 h) de los acumulados de 6 h MSWEP V3
(mayo–noviembre, 2000–2025, todas las causas de lluvia, no solo ciclones):

    1. Máximos estacionales M_y = max R_d en la temporada y -> GEV por L-momentos.
       T(x) = periodo de retorno en años (probabilidad 1/T de superarse en una temporada).
    2. Picos independientes (eventos separados por al menos separacion_h) -> los
       K = picos_por_temporada x n_temporadas mayores -> GPD sobre el (K+1)-ésimo.
       R(x) = intervalo medio de recurrencia en años; sirve también por debajo de
       un año (R = 0.33: tres eventos por temporada).

    R_d(T) = suma de los acumulados de 6 h en (T-d, T], dentro de cada temporada
    (cada archivo anual es una temporada; no cruza dic -> may).

Productos (data/processed/mswep_v3_return_periods_may_nov.nc):

    seasonal_max      (duration, season, lat, lon)   máximos estacionales
    peaks             (duration, rank, lat, lon)     K+1 picos mayores (opcional)
    n_seasons         (duration, lat, lon)           temporadas válidas
    gev_xi, gev_sigma, gev_mu, gev_xi_clipped
    gpd_u, gpd_sigma, gpd_xi, gpd_rate, gpd_xi_clipped
    rl_gev            (duration, T, lat, lon)        nivel con periodo de retorno T (GEV)
    rl_pot            (duration, R, lat, lon)        nivel con recurrencia R (GPD)
    rl_pot_emp        (duration, R, lat, lon)        idem, empírico (rango de los picos)

Uso posterior: itchi.return_periods.gev_return_period / pot_recurrence_interval
convierten cualquier lluvia en su periodo de retorno con estos parámetros.

Ejecutar desde scripts/ (Anaconda Prompt, entorno itchi):
    python periodos_retorno.py
    python periodos_retorno.py --durations 24 --sin-picos
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import itchi  # noqa: F401
except ModuleNotFoundError:   # paquete no instalado: usar src/ del repo
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itchi.return_periods import (
    declustered_peaks,
    empirical_pot_level,
    fit_gev_lmoments,
    fit_gpd_top_k,
    gev_return_level,
    pot_return_level,
    running_sum,
    seasonal_maximum,
    update_top_k,
)

REPO = Path(__file__).resolve().parents[1]


@dataclass
class ConfigRP:

    directory: str = "D:/MSWEP_V316_6H"
    pattern: str = "mswep_v3_6h_{year}.nc"
    variable: str = "precipitation"
    start_year: int = 2000
    end_year: int = 2025

    durations: tuple = (6, 24, 72)
    # Separación mínima entre picos independientes (h): mayor que la duración y
    # que la vida típica de un sistema de lluvia sobre un punto
    separacion_h: dict = field(default_factory=lambda: {6: 48, 24: 48, 72: 96})
    wet: float = 1.0                     # mm por ventana: un pico debe ser lluvia
    picos_por_temporada: float = 3.0     # K = 3 x n_temporadas (umbral GPD ~ P95-P99 de 6 h)
    min_valid_fraction: float = 0.9      # temporada válida si >= 90 % de pasos válidos
    min_temporadas: int = 15             # GEV
    min_excesos: int = 20                # GPD
    xi_bounds: tuple = (-0.5, 0.5)

    T_gev: tuple = (2.0, 5.0, 10.0, 25.0, 50.0, 100.0)
    R_pot: tuple = (0.25, 1.0 / 3.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0)

    filas_por_banda: int = 30            # memoria: filas de latitud procesadas a la vez
    guardar_picos: bool = True
    out_file: str = str(REPO / "data/processed/mswep_v3_return_periods_may_nov.nc")

    @property
    def years(self) -> list[int]:
        return list(range(self.start_year, self.end_year + 1))

    @property
    def k(self) -> int:
        return int(round(self.picos_por_temporada * len(self.years)))


def local_path(path: str | Path) -> Path:
    """'D:/...' -> '/mnt/d/...' en WSL/Linux; sin cambios en Windows."""
    s = str(path).replace("\\", "/")
    if os.name != "nt" and re.match(r"^[A-Za-z]:/", s):
        return Path(f"/mnt/{s[0].lower()}") / s[3:]
    return Path(s)


# ============================================================================
# LECTURA
# ============================================================================


def read_year_xarray(cfg: ConfigRP) -> Callable[[int], tuple]:
    """Lector de un año: (tiempos, lat, lon, P6 (time, lat, lon) float32)."""

    import xarray as xr

    def read(year: int):
        path = local_path(cfg.directory) / cfg.pattern.format(year=year)
        if not path.exists():
            return None
        with xr.open_dataset(path) as ds:
            da = ds[cfg.variable].transpose("time", "lat", "lon")
            return (da["time"].values, da["lat"].values, da["lon"].values,
                    da.values.astype(np.float32))

    return read


def contiguous(times: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Coloca los pasos en una malla de 6 h sin huecos (pasos faltantes = NaN)."""

    t = times.astype("datetime64[h]").astype(np.int64)
    pos = (t - t[0]) // 6
    if np.any((t - t[0]) % 6) or np.any(np.diff(pos) <= 0):
        raise ValueError("Los tiempos no son múltiplos crecientes de 6 h.")
    if pos[-1] + 1 == len(pos):
        return values
    out = np.full((int(pos[-1]) + 1,) + values.shape[1:], np.nan, dtype=values.dtype)
    out[pos] = values
    return out


# ============================================================================
# CÁLCULO
# ============================================================================


def compute(cfg: ConfigRP, read: Callable[[int], tuple | None], verbose: bool = True) -> dict:
    """
    Recorre las temporadas una vez y acumula, por duración, los máximos
    estacionales y los K+1 picos independientes mayores de cada píxel.
    """

    years = cfg.years
    k1 = cfg.k + 1
    state = None
    leidos = []
    t0 = time.time()

    for iy, year in enumerate(years):
        rec = read(year)
        if rec is None:
            print(f"  {year}: sin archivo, se omite")
            continue
        times, lat, lon, p6 = rec
        p6 = contiguous(np.asarray(times), p6)
        ny, nx = p6.shape[1:]
        if state is None:
            state = {
                "lat": lat, "lon": lon,
                "am": {d: np.full((len(years), ny, nx), np.nan, np.float32) for d in cfg.durations},
                "top": {d: np.full((k1, ny, nx), np.nan, np.float32) for d in cfg.durations},
            }
        elif (ny, nx) != state["am"][cfg.durations[0]].shape[1:]:
            raise ValueError(f"{year}: la malla no coincide con la de los años anteriores.")

        for j0 in range(0, ny, cfg.filas_por_banda):
            band = slice(j0, min(j0 + cfg.filas_por_banda, ny))
            x6 = p6[:, band].astype(float)
            for d in cfg.durations:
                r = running_sum(x6, d // 6)
                state["am"][d][iy, band] = seasonal_maximum(r, cfg.min_valid_fraction)
                peaks = declustered_peaks(r, cfg.separacion_h[d] // 6, cfg.wet)
                state["top"][d][:, band] = update_top_k(state["top"][d][:, band], peaks, k1)
        leidos.append(year)
        if verbose:
            print(f"  {year}: {p6.shape[0]} pasos · {(time.time() - t0) / 60:.1f} min")

    if state is None:
        raise FileNotFoundError(
            f"No se leyó ninguna temporada en {local_path(cfg.directory)} ({cfg.pattern}).")

    fits = {}
    for d in cfg.durations:
        am = state["am"][d]
        n_seasons = np.isfinite(am).sum(axis=0)
        gev = fit_gev_lmoments(am, min_n=cfg.min_temporadas, xi_bounds=cfg.xi_bounds)
        gpd = fit_gpd_top_k(state["top"][d], np.maximum(n_seasons, 1), xi_bounds=cfg.xi_bounds,
                            min_excesses=cfg.min_excesos)
        T = np.asarray(cfg.T_gev).reshape(-1, 1, 1)
        R = np.asarray(cfg.R_pot).reshape(-1, 1, 1)
        fits[d] = {
            "n_seasons": n_seasons, "gev": gev, "gpd": gpd,
            "rl_gev": gev_return_level(T, gev["xi"], gev["sigma"], gev["mu"]),
            "rl_pot": pot_return_level(R, gpd["u"], gpd["sigma"], gpd["xi"], gpd["rate"]),
            "rl_pot_emp": empirical_pot_level(state["top"][d][:-1], float(np.median(n_seasons)),
                                              cfg.R_pot),
        }
    return {"state": state, "fits": fits, "years": years, "leidos": leidos}


# ============================================================================
# SALIDA
# ============================================================================


def to_dataset(res: dict, cfg: ConfigRP):
    import xarray as xr

    st, durs = res["state"], list(cfg.durations)
    lat, lon = st["lat"], st["lon"]
    sp = ("lat", "lon")

    def stack(fn) -> np.ndarray:
        return np.stack([fn(d) for d in durs]).astype(np.float32)

    ds = xr.Dataset(coords={"duration": ("duration", durs), "lat": lat, "lon": lon,
                            "season": ("season", res["years"]),
                            "T": ("T", list(cfg.T_gev)), "R": ("R", list(cfg.R_pot))})
    ds["seasonal_max"] = (("duration", "season") + sp, stack(lambda d: st["am"][d]))
    if cfg.guardar_picos:
        ds = ds.assign_coords(rank=("rank", np.arange(1, cfg.k + 2)))
        ds["peaks"] = (("duration", "rank") + sp, stack(lambda d: st["top"][d]))
    ds["n_seasons"] = (("duration",) + sp, stack(lambda d: res["fits"][d]["n_seasons"]).astype(np.int16))
    for fam in ("gev", "gpd"):
        keys = ("xi", "sigma", "mu") if fam == "gev" else ("u", "sigma", "xi", "rate")
        for key in keys:
            ds[f"{fam}_{key}"] = (("duration",) + sp, stack(lambda d, f=fam, k=key: res["fits"][d][f][k]))
        ds[f"{fam}_xi_clipped"] = (("duration",) + sp,
                                   stack(lambda d, f=fam: res["fits"][d][f]["xi_clipped"]).astype(np.int8))
    ds["rl_gev"] = (("duration", "T") + sp, stack(lambda d: res["fits"][d]["rl_gev"]))
    ds["rl_pot"] = (("duration", "R") + sp, stack(lambda d: res["fits"][d]["rl_pot"]))
    ds["rl_pot_emp"] = (("duration", "R") + sp, stack(lambda d: res["fits"][d]["rl_pot_emp"]))

    mm = "mm per accumulation window of `duration` hours"
    attrs = {
        "duration": dict(units="h", long_name="accumulation length d: R_d(T) = precipitation in (T-d, T]"),
        "T": dict(units="year", long_name="return period of the seasonal maximum (GEV)"),
        "R": dict(units="year", long_name="mean recurrence interval of independent events (POT)"),
        "seasonal_max": dict(units=mm, long_name="May-November maximum of R_d"),
        "n_seasons": dict(units="1", long_name="seasons with >= min_valid_fraction valid steps"),
        "gev_xi": dict(units="1", long_name="GEV shape (Coles sign: > 0 heavy tail), L-moments"),
        "gev_sigma": dict(units=mm, long_name="GEV scale"),
        "gev_mu": dict(units=mm, long_name="GEV location"),
        "gpd_u": dict(units=mm, long_name="POT threshold: (K+1)-th largest independent peak"),
        "gpd_sigma": dict(units=mm, long_name="GPD scale of the excesses over gpd_u"),
        "gpd_xi": dict(units="1", long_name="GPD shape (Coles sign), L-moments"),
        "gpd_rate": dict(units="events per season", long_name="independent events above gpd_u"),
        "rl_gev": dict(units=mm, long_name="return level of the seasonal maximum for return period T"),
        "rl_pot": dict(units=mm, long_name="value reached once every R years on average (GPD)"),
        "rl_pot_emp": dict(units=mm, long_name="same, empirical (rank R^-1 x n_seasons of the peaks)"),
    }
    if cfg.guardar_picos:
        attrs["peaks"] = dict(units=mm, long_name="largest independent peaks of R_d, descending")
    for name, a in attrs.items():
        ds[name].attrs.update(a)
    ds.attrs.update(
        framework="ITCHI-CORE",
        title="Return periods of MSWEP V3 precipitation, May-November",
        source="MSWEP V3 6-h accumulations (scripts/mswep_v3_6h_accum.py)",
        period=f"{cfg.start_year}-{cfg.end_year}",
        seasons_read=",".join(map(str, res["leidos"])),
        causes="all rainfall (not only tropical cyclones)",
        peak_separation_h=",".join(f"{d}:{cfg.separacion_h[d]}" for d in durs),
        peak_min_mm=cfg.wet,
        peaks_per_season_K=cfg.picos_por_temporada,
        xi_bounds=str(cfg.xi_bounds),
        estimators="GEV and GPD by L-moments (Hosking 1990; Hosking and Wallis 1987)",
        conversion="T = 1 / (1 - exp(-1/R)) for Poisson events",
        code="scripts/periodos_retorno.py, itchi.return_periods",
    )
    return ds


def write(ds, cfg: ConfigRP) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(cfg.out_file)), exist_ok=True)
    enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(cfg.out_file, encoding=enc)


def run(cfg: ConfigRP = ConfigRP()):
    t0 = time.time()
    print("\n" + "=" * 75)
    print("ITCHI-CORE · Periodos de retorno de la lluvia (MSWEP V3, mayo-noviembre)")
    print("=" * 75)
    print(f"durations {cfg.durations} h · temporadas {cfg.start_year}-{cfg.end_year} · "
          f"K = {cfg.k} picos por píxel · {local_path(cfg.directory)}")
    res = compute(cfg, read_year_xarray(cfg))
    ds = to_dataset(res, cfg)
    write(ds, cfg)
    for d in cfg.durations:
        f = res["fits"][d]
        print(f"  {d:>2} h: GEV válido en {np.isfinite(f['gev']['xi']).mean():.1%} de los píxeles, "
              f"forma acotada en {f['gev']['xi_clipped'].mean():.1%}; "
              f"GPD válido en {np.isfinite(f['gpd']['xi']).mean():.1%}")
    print(f"\nPROCESO TERMINADO · {(time.time() - t0) / 60:.1f} min · {cfg.out_file}")
    return ds


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Periodos de retorno de la lluvia MSWEP V3")
    ap.add_argument("--durations", type=int, nargs="+", default=list(ConfigRP.durations))
    ap.add_argument("--filas-por-banda", type=int, default=ConfigRP.filas_por_banda,
                    help="baja este número si falta memoria")
    ap.add_argument("--sin-picos", action="store_true", help="no guarda los picos (archivo menor)")
    ap.add_argument("--out", default=ConfigRP.out_file)
    a = ap.parse_args()
    run(ConfigRP(durations=tuple(a.durations), filas_por_banda=a.filas_por_banda,
                 guardar_picos=not a.sin_picos, out_file=a.out))

