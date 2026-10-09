"""
ITCHI-CORE · Acumulados de 6 h MSWEP V3 (preprocesamiento)

Convierte los archivos horarios MSWEP V3 en un producto de precipitación
acumulada de 6 h, recortado a la región ITCHI y a mayo–noviembre.

Definición del producto (contrato explícito, ver docs/data_contracts.md):

    precipitation(T) = suma de la precipitación horaria en (T-6h, T]
    T ∈ {00, 06, 12, 18} UTC

    Es la misma convención que la precipitación total de 6 h de AIFS/IFS
    en el tiempo válido T. El núcleo de ITCHI lee este campo como un
    snapshot en valid_time = T; no vuelve a sumar en el tiempo.

    Temporada: T ∈ [1 may 06 UTC, 1 dic 00 UTC] → cubre exactamente
    1 may 00 UTC – 1 dic 00 UTC (856 ventanas por año).
    Ventana con alguna hora faltante → NaN (no se rellena).

Entrada:
    Carpeta con TODOS los archivos horarios MSWEP V3 (un nivel, sin subcarpetas).

Salida:
    ../data/interim/mswep_v3_6h/mswep_v3_6h_YYYY.nc   (un archivo por año)
    ../data/interim/mswep_v3_6h/index_hourly.csv       (índice tiempo -> archivo)

Ejecutar desde scripts/:
    python mswep_v3_6h_accum.py
"""

from __future__ import annotations

import os
import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

# ============================================================================
# CONFIGURACIÓN
# ============================================================================


@dataclass
class Config:

    # Desde WSL: /mnt/d/MSWEP_V316_HOURLY
    directory: str = "D:/MSWEP_V316_HOURLY"

    start_year: int = 2000
    end_year: int = 2025

    lat_bounds: tuple = (5, 35)
    lon_bounds: tuple = (-140, -60)

    variable: str = "precipitation"

    # Convención de la etiqueta horaria de MSWEP V3:
    #   "start": la etiqueta h cubre [h, h+1)
    #   "end"  : la etiqueta h cubre (h-1, h]
    #   "auto" : se infiere de time.bounds; si no existe, el script se detiene.
    # VERIFICAR con la documentación de MSWEP V3 antes de la corrida completa.
    time_label: str = "auto"

    # Archivos usados para validar tiempo del nombre vs tiempo interno.
    n_validate: int = 30

    # Procesos en paralelo (uno por año). Con disco mecánico, no subir de 2.
    workers: int = 2

    overwrite: bool = False

    # Carpeta HERMANA de la horaria en D:, nunca dentro de ella
    # (el índice lista todos los .nc de cfg.directory).
    out_dir: str = "D:/MSWEP_V316_6H"


WINDOW_H = 6
NC_EXT = (".nc", ".nc4")

# ============================================================================
# TIEMPO
# ============================================================================

_NAME_PATTERNS = (
    # 2000123.06.nc (estilo MSWEP: año + día juliano + hora)
    ("yjh", re.compile(r"(?<!\d)(?P<y>\d{4})(?P<j>\d{3})\.(?P<h>\d{2})(?!\d)")),
    # 2000050306 / 20000503.06 / 20000503_06 / 20000503T06
    ("ymdh", re.compile(r"(?<!\d)(?P<y>\d{4})(?P<m>\d{2})(?P<d>\d{2})[._T-]?(?P<h>\d{2})(?!\d)")),
)


def time_from_name(path: str) -> pd.Timestamp | None:
    """Tiempo nominal UTC a partir del nombre; None si no se reconoce."""

    name = os.path.basename(path)

    for kind, pat in _NAME_PATTERNS:
        m = pat.search(name)
        if not m:
            continue
        g = {k: int(v) for k, v in m.groupdict().items()}
        try:
            if kind == "yjh" and 1 <= g["j"] <= 366 and g["h"] <= 23:
                return pd.Timestamp(datetime(g["y"], 1, 1) + timedelta(days=g["j"] - 1, hours=g["h"]))
            if kind == "ymdh" and g["h"] <= 23:
                return pd.Timestamp(datetime(g["y"], g["m"], g["d"], g["h"]))
        except ValueError:
            continue

    return None


def window_ends(year: int) -> pd.DatetimeIndex:
    """Tiempos finales T de las ventanas de 6 h de mayo–noviembre."""

    return pd.date_range(f"{year}-05-01 06:00", f"{year}-12-01 00:00", freq=f"{WINDOW_H}h")


def hours_in_window(T: pd.Timestamp, time_label: str) -> list[pd.Timestamp]:
    """
    Etiquetas horarias que componen (T-6h, T].

        start: T-6, ..., T-1
        end  : T-5, ..., T
    """

    if time_label == "start":
        ks = range(WINDOW_H, 0, -1)
    elif time_label == "end":
        ks = range(WINDOW_H - 1, -1, -1)
    else:
        raise ValueError(f"time_label inválido: {time_label}")

    return [T - pd.Timedelta(hours=k) for k in ks]


def detect_time_label(path: str) -> str | None:
    """
    Infere la convención a partir de time.bounds, si existe.

    El tiempo de MSWEP V3 viene en float32 (días desde 1900, resolución
    ~337 s), por lo que no se compara con np.isclose (rtol=1e-5 sobre ~44 000
    días equivale a ~10 h). Se elige el límite más cercano y se exige que la
    diferencia sea < 1/4 del intervalo.
    """

    with xr.open_dataset(path, decode_times=False) as ds:
        tname = "time" if "time" in ds else None
        bname = ds[tname].attrs.get("bounds") if tname else None
        if not bname or bname not in ds:
            return None
        t0 = float(ds[tname].values.ravel()[0])
        b0, b1 = (float(b) for b in np.asarray(ds[bname].values).reshape(-1, 2)[0])

    d0, d1, width = abs(t0 - b0), abs(t0 - b1), abs(b1 - b0)

    if min(d0, d1) > 0.25 * width:
        print(f"  ADVERTENCIA: tiempo {t0} no está en un extremo de bounds ({b0}, {b1}).")
        return None

    return "start" if d0 < d1 else "end"


# ============================================================================
# ÍNDICE DE ARCHIVOS
# ============================================================================


def list_hourly_files(cfg: Config) -> list[str]:

    with os.scandir(cfg.directory) as it:
        return sorted(e.path for e in it if e.is_file() and e.name.lower().endswith(NC_EXT))


def _internal_times(path: str) -> pd.DatetimeIndex:
    """
    Tiempo interno redondeado a la hora.

    MSWEP V3 guarda time en float32: al decodificar aparecen errores de
    ±112.5 s (p. ej. 04:01:52.5 en lugar de 04:00). Sin redondear, 2/3 de
    las horas no coincidirían con las ventanas.
    """

    with xr.open_dataset(path) as ds:
        return pd.DatetimeIndex(ds["time"].values).round("h")


def build_index(cfg: Config) -> pd.DataFrame:
    """
    Índice (time, path, tidx). Se guarda en CSV y se reutiliza.

    Ruta rápida: un paso por archivo y tiempo legible en el nombre,
    validado contra el tiempo interno en una muestra aleatoria.
    Ruta lenta: lee el tiempo interno de cada archivo.
    """

    index_csv = os.path.join(cfg.out_dir, "index_hourly.csv")

    if os.path.exists(index_csv):
        print(f"Usando índice existente: {index_csv}")
        return pd.read_csv(index_csv, parse_dates=["time"])

    files = list_hourly_files(cfg)

    if not files:
        raise FileNotFoundError(f"No hay archivos netCDF en {cfg.directory}")

    print(f"Archivos encontrados: {len(files):,}")
    print(f"  primero: {os.path.basename(files[0])}")
    print(f"  último : {os.path.basename(files[-1])}")

    parsed = [time_from_name(f) for f in files]
    one_step = len(_internal_times(files[0])) == 1
    fast = one_step and all(p is not None for p in parsed)

    if fast:
        sample = random.Random(0).sample(range(len(files)), min(cfg.n_validate, len(files)))
        bad = [i for i in sample if _internal_times(files[i])[0] != parsed[i]]
        for i in bad[:5]:
            print(f"  ¡Discrepancia! {os.path.basename(files[i])}: nombre {parsed[i]} "
                  f"vs interno {_internal_times(files[i])[0]}")
        print(f"Validación nombre vs tiempo interno: {len(bad)}/{len(sample)} discrepancias")
        fast = not bad

    if fast:
        df = pd.DataFrame({"time": parsed, "path": files, "tidx": 0})
    else:
        print("Leyendo el tiempo interno de cada archivo (lento; se hace una sola vez)...")
        rows = []
        for f in tqdm(files, desc="Índice", unit="arch"):
            rows += [(t, f, i) for i, t in enumerate(_internal_times(f))]
        df = pd.DataFrame(rows, columns=["time", "path", "tidx"])

    df = df.sort_values("time").drop_duplicates("time", keep="first")

    full = pd.date_range(df["time"].min(), df["time"].max(), freq="1h")
    print(f"Índice: {len(df):,} pasos | {full[0]} – {full[-1]} | "
          f"horas faltantes: {len(full.difference(df['time'])):,}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    df.to_csv(index_csv, index=False)

    return df


# ============================================================================
# LECTURA HORARIA
# ============================================================================

_UNIT_SCALE = {  # factor -> mm por paso horario
    "mm": 1.0, "mm/h": 1.0, "mm h-1": 1.0, "mm hr-1": 1.0, "mm/hr": 1.0, "mm/hour": 1.0,
    "kg m-2": 1.0, "kg m**-2": 1.0,
    "kg m-2 s-1": 3600.0, "kg m**-2 s**-1": 3600.0, "mm/s": 3600.0, "mm s-1": 3600.0,
}


def _dim(da: xr.DataArray, keys: tuple[str, ...]) -> str:

    for d in da.dims:
        if d.lower().startswith(keys):
            return d

    raise KeyError(f"No se encontró dimensión {keys} en {da.dims}")


def region_indices(path: str, cfg: Config) -> dict:
    """Índices enteros del recorte, latitud de salida ascendente y factor de unidades."""

    with xr.open_dataset(path) as ds:
        da = ds[cfg.variable]
        ylat, xlon = _dim(da, ("lat", "y")), _dim(da, ("lon", "x"))
        lat, lon = ds[ylat].values, ds[xlon].values
        units = da.attrs.get("units", "")

    lon_min, lon_max = cfg.lon_bounds
    if lon.max() > 180.0:  # malla 0..360
        lon_min, lon_max = lon_min % 360.0, lon_max % 360.0

    jl = np.where((lat >= cfg.lat_bounds[0]) & (lat <= cfg.lat_bounds[1]))[0]
    il = np.where((lon >= lon_min) & (lon <= lon_max))[0]

    lat_sub = lat[jl]
    flip = bool(lat_sub[0] > lat_sub[-1])  # MSWEP: latitud descendente

    scale = _UNIT_SCALE.get(units.strip().lower())
    if scale is None:
        print(f"  ADVERTENCIA: unidades '{units}' no reconocidas; se asume mm por hora.")
        scale = 1.0

    return dict(
        ylat=ylat, xlon=xlon, jl=jl, il=il, flip=flip, scale=scale,
        lat=(lat_sub[::-1] if flip else lat_sub).astype("float64"),
        lon=np.where(lon[il] > 180.0, lon[il] - 360.0, lon[il]).astype("float64"),
    )


def make_reader(cfg: Config, reg: dict):
    """Devuelve read(path, tidx) -> ndarray (lat asc, lon) en mm."""

    def read(path: str, tidx: int) -> np.ndarray:

        with xr.open_dataset(path, cache=False) as ds:
            da = ds[cfg.variable]
            sel = {reg["ylat"]: reg["jl"], reg["xlon"]: reg["il"]}
            if "time" in da.dims:
                sel["time"] = tidx
            a = da.isel(sel).transpose(reg["ylat"], reg["xlon"]).values.astype("float32")

        if reg["flip"]:
            a = a[::-1, :]

        a[a < 0] = np.nan  # valores negativos = no físicos

        return a * np.float32(reg["scale"])

    return read


# ============================================================================
# ACUMULACIÓN
# ============================================================================


def accumulate_windows(ends, lookup: dict, read_fn, shape: tuple, time_label: str, progress: bool = False):
    """
    Suma 6 campos horarios por ventana.
    Ventana incompleta -> NaN. NaN de un píxel se propaga a la ventana.
    """

    out = np.full((len(ends), *shape), np.nan, dtype="float32")
    missing = []

    for k, T in tqdm(enumerate(ends), total=len(ends), desc="Ventanas 6 h", unit="ven", disable=not progress):
        locs = [lookup.get(h) for h in hours_in_window(T, time_label)]
        if any(loc is None for loc in locs):
            missing.append(T)
            continue
        acc = np.zeros(shape, dtype="float32")
        for path, tidx in locs:
            acc += read_fn(path, tidx)
        out[k] = acc

    return out, missing


def write_year(path: str, data, ends, lat, lon, attrs: dict) -> None:

    da = xr.DataArray(
        data, dims=("time", "lat", "lon"),
        coords={"time": ends, "lat": lat, "lon": lon},
        name="precipitation",
        attrs={
            "units": "mm",
            "long_name": "6-hour accumulated precipitation",
            "cell_methods": "time: sum (interval: 1 hour)",
            "accumulation_window": "(T-6h, T], T = valid_time",
        },
    )
    da["time"].attrs["long_name"] = "end of 6-h accumulation window (valid time)"

    ds = da.to_dataset()
    ds.attrs.update(attrs)

    encoding = {
        "precipitation": {
            "zlib": True, "complevel": 4, "dtype": "float32",
            "least_significant_digit": 2,                      # 0.01 mm
            "chunksizes": (len(ends), min(25, lat.size), min(25, lon.size)),
        },
    }

    tmp = path.replace(".nc", ".tmp.nc")
    ds.to_netcdf(tmp, encoding=encoding)
    os.replace(tmp, path)                                      # escritura atómica


def accumulate_year(year: int, cfg: Config, index_year: pd.DataFrame, time_label: str) -> str:

    out = os.path.join(cfg.out_dir, f"mswep_v3_6h_{year}.nc")

    if os.path.exists(out) and not cfg.overwrite:
        return f"{year}: ya existe, se omite"

    if index_year.empty:
        return f"{year}: sin archivos horarios"

    lookup = {t: (p, int(i)) for t, p, i in zip(index_year["time"], index_year["path"], index_year["tidx"])}

    reg = region_indices(index_year["path"].iloc[0], cfg)
    ends = window_ends(year)

    data, missing = accumulate_windows(
        ends, lookup, make_reader(cfg, reg), (reg["lat"].size, reg["lon"].size), time_label,
        progress=cfg.workers <= 1,
    )

    write_year(out, data, ends, reg["lat"], reg["lon"], {
        "framework": "ITCHI-CORE",
        "source": "MSWEP V3 hourly",
        "hourly_time_label": time_label,
        "season": "May-Nov",
        "n_windows": len(ends),
        "n_missing_windows": len(missing),
        "history": f"scripts/mswep_v3_6h_accum.py {datetime.now(timezone.utc):%Y-%m-%dT%H:%MZ}",
    })

    return f"{year}: {len(ends) - len(missing)}/{len(ends)} ventanas completas"


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================


def check_out_dir(cfg: Config) -> None:
    """Evita escribir el producto de 6 h dentro de la carpeta horaria."""

    src = os.path.normcase(os.path.abspath(cfg.directory))
    out = os.path.normcase(os.path.abspath(cfg.out_dir))
    if os.path.commonpath([src, out]) == src:
        raise SystemExit(
            f"out_dir ({cfg.out_dir}) está dentro de la carpeta horaria ({cfg.directory}).\n"
            "Usar una carpeta hermana, p. ej. D:/MSWEP_V316_6H."
        )


def run(cfg: Config = Config()) -> None:

    t0 = time.time()

    print("\n" + "=" * 75)
    print("ITCHI-CORE · MSWEP V3 horario -> acumulados de 6 h")
    print("=" * 75)

    check_out_dir(cfg)
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("\n[1/3] Índice de archivos horarios")

    index = build_index(cfg)

    print("\n[2/3] Convención temporal")

    label = cfg.time_label
    if label == "auto":
        label = detect_time_label(index["path"].iloc[0])
        if label is None:
            raise SystemExit(
                "No hay time.bounds para inferir la convención horaria.\n"
                "Definir Config.time_label = 'start' o 'end' según la documentación de MSWEP V3."
            )
    print(f"time_label = {label}")

    print("\n[3/3] Acumulados por año")

    jobs = {}
    for year in range(cfg.start_year, cfg.end_year + 1):
        e = window_ends(year)
        t0y, t1y = e[0] - pd.Timedelta(hours=WINDOW_H), e[-1]
        jobs[year] = index[(index["time"] >= t0y) & (index["time"] <= t1y)]

    if cfg.workers <= 1:
        for year, idx in tqdm(jobs.items(), desc="Años", unit="año"):
            print(accumulate_year(year, cfg, idx, label))
    else:
        with ProcessPoolExecutor(max_workers=cfg.workers) as ex:
            futs = [ex.submit(accumulate_year, y, cfg, idx, label) for y, idx in jobs.items()]
            for f in tqdm(as_completed(futs), total=len(futs), desc="Años", unit="año"):
                print(f.result())

    print("\n" + "=" * 75)
    print(f"PROCESO TERMINADO · {(time.time() - t0) / 60:.1f} minutos · salida: {cfg.out_dir}")
    print("=" * 75)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    run(Config(start_year=2000, end_year=2025, time_label="start", workers=2))
