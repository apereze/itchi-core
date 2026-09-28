"""
ITCHI-CORE · Percentiles de precipitación extrema MSWEP V2.8

Calcula, por píxel, percentiles de precipitación diaria MSWEP V2.8
para la temporada mayo–noviembre.

Productos:

(A) p_wet
    P90, P95 y P99 sobre días húmedos (P >= 1 mm d-1).
    Producto principal.

(B) p_all
    P90, P95 y P99 sobre todos los días.

(C) n_wet
    Número de días húmedos por píxel.

(D) p_wet_raw
    Percentiles sobre días húmedos sin filtro de suficiencia muestral.

(E) frac_above_p_wet
    Fracción de la precipitación total de días húmedos aportada por
    los días que superan cada percentil.

Entrada:
    Archivos diarios MSWEP:
        YYYYDDD.nc

Salida:
    NetCDF único con:
        p_wet
        p_wet_raw
        n_wet
        frac_above_p_wet
        p_all

Referencias
  Zhang et al. (2011)  doi:10.1002/wcc.147
  Schär et al. (2016)  doi:10.1007/s10584-016-1669-2
  Hyndman & Fan (1996) doi:10.1080/00031305.1996.10473566
  Beck et al. (2019)   doi:10.1175/BAMS-D-17-0138.1
"""

from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass, field

import pandas as pd
import xarray as xr
from dask.diagnostics import ProgressBar
from tqdm.auto import tqdm

# ============================================================================
# CONFIGURACIÓN
# ============================================================================


@dataclass
class Config:

    # ------------------------------------------------------------------------
    # Entrada
    # ------------------------------------------------------------------------

    # Ruta desde WSL.
    #
    # Si los datos están realmente en:
    # D:\MSWEP_V280\merged\Daily
    #
    # desde WSL corresponde a:
    # /mnt/d/MSWEP_V280/merged/Daily
    #
    directory: str = "D:/MSWEP_V280/merged/Daily"

    start_year: int = 2000
    end_year: int = 2024

    # Mayo-noviembre
    months: tuple = (5, 6, 7, 8, 9, 10, 11)

    # ------------------------------------------------------------------------
    # Región
    # ------------------------------------------------------------------------

    # MSWEP normalmente utiliza latitud descendente.
    lat_bounds: tuple = (35, 5)
    lon_bounds: tuple = (-140, -60)

    # ------------------------------------------------------------------------
    # Definición de día húmedo
    # ------------------------------------------------------------------------

    wet: float = 1.0  # mm d-1

    # ------------------------------------------------------------------------
    # Percentiles
    # ------------------------------------------------------------------------

    qs: tuple = (0.90, 0.95, 0.99)

    # ------------------------------------------------------------------------
    # Control de suficiencia muestral
    # ------------------------------------------------------------------------

    # Número mínimo esperado de excedencias sobre el percentil.
    #
    # Para:
    #   P90 -> 10 / 0.10 = 100 días húmedos
    #   P95 -> 10 / 0.05 = 200 días húmedos
    #   P99 -> 10 / 0.01 = 1000 días húmedos
    #
    k_exc: int = 10

    # Hyndman & Fan type 8
    qmethod: str = "median_unbiased"

    # ------------------------------------------------------------------------
    # Chunks Dask
    # ------------------------------------------------------------------------

    # El tiempo completo debe estar en un solo chunk porque quantile()
    # opera sobre la dimensión temporal.
    #
    # Los bloques espaciales moderados evitan crear chunks gigantes.
    chunks: dict = field(
        default_factory=lambda: {
            "time": -1,
            "lat": 25,
            "lon": 25,
        }
    )

    # ------------------------------------------------------------------------
    # Salida
    # ------------------------------------------------------------------------

    out_file: str = "../data/processed/mswep_percentiles_may_nov.nc"


# ============================================================================
# UTILIDADES
# ============================================================================


def time_from_name(path: str) -> pd.Timestamp:
    """
    Convierte:
        YYYYDDD.nc

    a Timestamp.
    """

    name = os.path.splitext(os.path.basename(path))[0]

    return pd.to_datetime(
        name,
        format="%Y%j",
    )


# ============================================================================
# LISTADO DE ARCHIVOS
# ============================================================================


def list_files(cfg: Config) -> list[str]:
    """
    Busca archivos diarios MSWEP y conserva únicamente
    los correspondientes a mayo-noviembre.
    """

    files = []

    years = range(
        cfg.start_year,
        cfg.end_year + 1,
    )

    print("\nBuscando archivos MSWEP...")

    for year in tqdm(
        years,
        desc="Años",
        unit="año",
    ):

        pattern = os.path.join(
            cfg.directory,
            f"{year}???.nc",
        )

        year_files = glob.glob(pattern)

        files.extend(year_files)

    files = sorted(files)

    # Filtrar por mes.
    files = [f for f in files if time_from_name(f).month in cfg.months]

    return files


# ============================================================================
# PREPROCESS
# ============================================================================


def _make_preprocess(cfg: Config):
    """
    Recorta espacialmente cada archivo y asigna
    la fecha correspondiente al nombre del archivo.
    """

    def prep(ds: xr.Dataset) -> xr.Dataset:

        # Fecha a partir del nombre YYYYDDD.nc
        source = ds.encoding["source"]

        t = time_from_name(source)

        # Recorte espacial
        ds = ds.sel(
            lat=slice(*cfg.lat_bounds),
            lon=slice(*cfg.lon_bounds),
        )

        # Asegurar una dimensión temporal de longitud 1.
        if "time" in ds.dims:

            return ds.assign_coords(time=[t])

        return ds.expand_dims(time=[t])

    return prep


# ============================================================================
# APERTURA DE PRECIPITACIÓN
# ============================================================================


def open_precip(cfg: Config) -> xr.DataArray:
    """
    Abre la precipitación como DataArray Dask.

    Los archivos se leen respetando sus chunks originales.
    Posteriormente se reorganiza únicamente la dimensión temporal
    para facilitar el cálculo de cuantiles.
    """

    files = list_files(cfg)

    print(f"\nArchivos encontrados: {len(files):,}")

    if not files:
        raise FileNotFoundError(
            "\nNo se encontraron archivos MSWEP.\n"
            f"Directorio buscado:\n{cfg.directory}\n"
        )

    print("\nPrimeros archivos:")

    for f in files[:3]:
        print(f"  {f}")

    print("\nÚltimos archivos:")

    for f in files[-3:]:
        print(f"  {f}")

    print("\nAbriendo dataset...")

    ds = xr.open_mfdataset(
        files,
        preprocess=_make_preprocess(cfg),
        combine="nested",
        concat_dim="time",
        parallel=False,
        chunks=None,
        data_vars="minimal",
        coords="minimal",
        compat="override",
    )

    if "precipitation" not in ds:
        raise KeyError(
            "No se encontró la variable " "'precipitation' en los archivos MSWEP."
        )

    pr = ds["precipitation"]

    print("\nDataset abierto correctamente.")

    print("\nDimensiones:")
    for dim, size in pr.sizes.items():
        print(f"  {dim:>6}: {size:,}")

    print("\nChunks originales:")
    print(pr.chunks)

    # ------------------------------------------------------------
    # Reorganización Dask
    # ------------------------------------------------------------

    print("\nConfigurando chunks para el cálculo...")

    # Mantener los chunks espaciales originales de MSWEP.
    # Solo consolidamos toda la serie temporal para que
    # xarray pueda calcular los cuantiles sobre time.

    pr = pr.chunk(
        {
            "time": -1,
        }
    )

    print("\nChunks de trabajo:")
    print(pr.chunks)

    return pr


# ============================================================================
# PERCENTILES DE DÍAS HÚMEDOS
# ============================================================================


def wet_day_percentiles(
    pr: xr.DataArray,
    cfg: Config,
):
    """
    Calcula P90, P95 y P99 sobre días húmedos.

    Días húmedos:
        precipitation >= cfg.wet

    Devuelve:

        q_raw
            Percentiles sin filtro de suficiencia.

        q_wet
            Percentiles con filtro de suficiencia muestral.

        n_wet
            Número de días húmedos.

        wet
            Serie de precipitación únicamente para días húmedos.
    """

    print("\nAplicando umbral de día húmedo...")

    wet = pr.where(pr >= cfg.wet)

    print("Calculando número de días húmedos...")

    n_wet = wet.count(dim="time")

    print("Calculando P90/P95/P99 sobre días húmedos...")

    q_raw = wet.quantile(
        list(cfg.qs),
        dim="time",
        method=cfg.qmethod,
        skipna=True,
    )

    # ------------------------------------------------------------
    # Suficiencia muestral
    # ------------------------------------------------------------

    n_min = xr.DataArray(
        [cfg.k_exc / (1 - q) for q in cfg.qs],
        dims="quantile",
        coords={"quantile": list(cfg.qs)},
    )

    q_wet = q_raw.where(n_wet >= n_min)

    return (
        q_raw,
        q_wet,
        n_wet,
        wet,
    )


# ============================================================================
# CONTRIBUCIÓN DE EVENTOS EXTREMOS
# ============================================================================


def contribution_above(
    wet: xr.DataArray,
    q: xr.DataArray,
) -> xr.DataArray:
    """
    Fracción de la precipitación total de días húmedos
    aportada por eventos que exceden cada percentil.
    """

    total = wet.sum(dim="time")

    extreme_total = wet.where(wet > q).sum(dim="time")

    return extreme_total / total.where(total > 0)


# ============================================================================
# PERCENTILES DE TODOS LOS DÍAS
# ============================================================================


def all_day_percentiles(
    pr: xr.DataArray,
    cfg: Config,
) -> xr.DataArray:
    """
    P90/P95/P99 sobre todos los días,
    incluidos los días con precipitación < 1 mm.
    """

    print("Calculando P90/P95/P99 sobre todos los días...")

    return pr.quantile(
        list(cfg.qs),
        dim="time",
        method=cfg.qmethod,
        skipna=True,
    )


# ============================================================================
# METADATOS
# ============================================================================


def add_metadata(
    out: xr.Dataset,
    cfg: Config,
) -> xr.Dataset:

    for var in (
        "p_wet",
        "p_wet_raw",
        "p_all",
    ):
        out[var].attrs.update(
            units="mm d-1",
            long_name=(
                "Precipitation percentile " "for wet days"
                if var != "p_all"
                else "Precipitation percentile " "for all days"
            ),
        )

    out["n_wet"].attrs.update(
        units="days",
        long_name=("Number of wet days"),
    )

    out["frac_above_p_wet"].attrs.update(
        units="1",
        long_name=(
            "Fraction of wet-day precipitation " "from events exceeding the percentile"
        ),
    )

    out.attrs.update(
        framework="ITCHI-CORE",
        source=("MSWEP V2.8 daily merged"),
        period=(f"{cfg.start_year}-" f"{cfg.end_year}"),
        months=(",".join(map(str, cfg.months))),
        wet_day_threshold_mm=cfg.wet,
        quantiles=(",".join(map(str, cfg.qs))),
        quantile_method=cfg.qmethod,
        min_expected_exceedances=cfg.k_exc,
        day_definition=("UTC 00-00"),
        description=("Daily MSWEP precipitation " "percentiles for May-November."),
    )

    return out


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================


def run(
    cfg: Config = Config(),
) -> xr.Dataset:

    t0 = time.time()

    print("\n" + "=" * 75)
    print("ITCHI-CORE · MSWEP precipitation percentiles")
    print("=" * 75)

    # ========================================================================
    # 1. LECTURA
    # ========================================================================

    print("\n[1/5] Lectura de MSWEP")

    pr = open_precip(cfg)

    # ========================================================================
    # 2. PERCENTILES DÍAS HÚMEDOS
    # ========================================================================

    print("\n[2/5] Percentiles de días húmedos")

    q_raw, q_wet, n_wet, wet = wet_day_percentiles(
        pr,
        cfg,
    )

    # ========================================================================
    # 3. PERCENTILES TODOS LOS DÍAS
    # ========================================================================

    print("\n[3/5] Percentiles de todos los días")

    p_all = all_day_percentiles(
        pr,
        cfg,
    )

    # ========================================================================
    # 4. CONTRIBUCIÓN DE EXTREMOS
    # ========================================================================

    print("\n[4/5] Contribución de eventos extremos")

    frac = contribution_above(
        wet,
        q_raw,
    )

    # ========================================================================
    # CONSTRUIR DATASET
    # ========================================================================

    out = xr.Dataset(
        {
            "p_wet": q_wet,
            "p_wet_raw": q_raw,
            "n_wet": n_wet,
            "frac_above_p_wet": frac,
            "p_all": p_all,
        }
    )

    out = add_metadata(
        out,
        cfg,
    )

    # ========================================================================
    # 5. CÁLCULO DASK + ESCRITURA
    # ========================================================================

    print("\n[5/5] Ejecutando cálculo Dask...")

    print("Los datos se están calculando ahora " "y escribiendo al NetCDF.")

    # ------------------------------------------------------------
    # Crear directorio de salida si no existe
    # ------------------------------------------------------------

    output_dir = os.path.dirname(os.path.abspath(cfg.out_file))

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Compresión
    # ------------------------------------------------------------

    encoding = {}

    for var in out.data_vars:

        encoding[var] = {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }

    # n_wet es un conteo.
    encoding["n_wet"] = {
        "zlib": True,
        "complevel": 4,
        "dtype": "int16",
    }

    # Fracción entre 0 y 1.
    encoding["frac_above_p_wet"] = {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
    }

    # ------------------------------------------------------------
    # Escribir
    # ------------------------------------------------------------

    with ProgressBar():

        out.to_netcdf(
            cfg.out_file,
            encoding=encoding,
        )

    elapsed = time.time() - t0

    print("\n" + "=" * 75)
    print("PROCESO TERMINADO")
    print("=" * 75)

    print(f"Archivo: {cfg.out_file}")

    print(f"Tiempo total: " f"{elapsed / 60:.2f} minutos")

    print("=" * 75)

    return out


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    out = run(Config(out_file=("../data/processed/" "mswep_percentiles_may_nov.nc")))
