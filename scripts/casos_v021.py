"""
ITCHI-CORE · Avance v0.2.1: casos de estudio en malla y por municipio

Calcula ITCHI con compute_itchi_snapshot_v02 en cada posición de la
trayectoria que puede afectar a México, para los experimentos de
configs/experiments/v021/, y lo agrega por municipio. Lo usan los notebooks
12 (lluvia: P99 contra P95), 13 (viento: v0.2.0 contra Saffir-Simpson) y
14 (deslizamientos y anticipación) de notebooks/third test/.

Experimentos (overlays sobre configs/default.yaml):

    v020    v0.2.0: matriz v0.2 (máximo en P99), cola hasta R_W06, viento / Vmax
    p95     máximo de lluvia en el P95 de 6 h (escalada R24, A72 recorrida a P90)
    w12     cola de viento cortada en 12 m/s (viento todavía / Vmax)
    ss      propuesta v0.2.1: cola en 12 m/s y viento absoluto Saffir-Simpson
    p95ss   P95 y Saffir-Simpson juntos

Unidad de cálculo. Malla MSWEP de 0.1° recortada a MEXICO_BOX. Se conservan
las posiciones cuya región de atribución (ROCLOUD con piso R_W06, el radio
mayor de los cuatro cuadrantes) alcanza algún píxel terrestre de México.

Valor municipal. Estadístico de ITCHI sobre los píxeles cuyo centro cae en el
municipio (itchi.declaratorias_6h.municipio_pixels; un municipio sin centros
toma el píxel más cercano a su punto representativo). Por defecto el máximo,
como en los notebooks 05–08 y el Bloque 1.

Salida (caché): outputs/diagnostics/avance_v021/<caso>.nc, que la segunda
corrida lee sin recalcular.

Ejecutar desde scripts/ (Anaconda Prompt, entorno itchi):
    python casos_v021.py                       # todos los casos
    python casos_v021.py --casos ODILE2014 MANUEL2013 --forzar
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

try:
    import itchi  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itchi.climatology import load_percentile_climatology
from itchi.config import get_project_root, itchi_v02_settings_from_config, load_experiment_config
from itchi.constants import QUADRANTS
from itchi.plotting import classify_itchi_to_siat_levels
from itchi.precipitation import compute_antecedent_precipitation, compute_precipitation_hazard_from_spec
from itchi.rocloud import DEFAULT_ROCLOUD_COLUMN_MAP
from itchi.tracks import DEFAULT_R34_COLUMN_MAP

ROOT = get_project_root()
EARTH_RADIUS_KM = 6371.0

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# atcf: identificador del ciclón en ROCLOUD y en las declaratorias (CT_pico)
CASES: dict[str, dict] = {
    "EMILY2005": dict(name="EMILY", season=2005, basin="NA", atcf="AL052005"),
    "ALEX2010": dict(name="ALEX", season=2010, basin="NA", atcf="AL012010"),
    "ODILE2014": dict(name="ODILE", season=2014, basin="EP", atcf="EP152014"),
    "PATRICIA2015": dict(name="PATRICIA", season=2015, basin="EP", atcf="EP202015"),
    "CRISTOBAL2020": dict(name="CRISTOBAL", season=2020, basin="NA", atcf="AL032020"),
    # caso 3 (deslizamientos)
    "MANUEL2013": dict(name="MANUEL", season=2013, basin="EP", atcf="EP132013"),
    "EARL2016": dict(name="EARL", season=2016, basin="NA", atcf="AL052016"),
}
CASOS_1_2 = ("EMILY2005", "ALEX2010", "PATRICIA2015", "CRISTOBAL2020", "ODILE2014")

# Caso 3: deslizamientos de ladera inducidos por ciclones, a nivel municipio.
# Hora local -> UTC: en 2013 y 2016 el centro de México tenía horario de verano
# (UTC-5) de abril a octubre.
DESLIZAMIENTOS: dict[str, dict] = {
    "MANUEL2013": dict(
        evento="La Pintada", municipio="Atoyac de Álvarez", cve_ent="12",
        t_local="2013-09-16 15:30", utc_offset_h=-5, incertidumbre_h=1.0,
        nota=("Alud que sepultó parte de La Pintada (Sierra de Atoyac, Guerrero) hacia las 15:30 del lunes "
              "16 de septiembre de 2013 (La Jornada, 3 nov 2013; Proceso, 21 sep 2013). El centro de Manuel se "
              "disipó sobre México hacia las 06 UTC del 16 y se regeneró en el golfo de California el 17 "
              "(NHC TCR EP132013), así que no hay posición del ciclón al ocurrir el deslizamiento. Ingrid "
              "estaba activa en el golfo de México al mismo tiempo."),
    ),
    "EARL2016": dict(
        evento="Huauchinango (colonias La Joya, Guadalupe, Aurora, La Cumbre; Xaltepec)",
        municipio="Huauchinango", cve_ent="21",
        t_local="2016-08-07 03:00", utc_offset_h=-5, incertidumbre_h=3.0,
        nota=("Deslizamientos en la madrugada del domingo 7 de agosto de 2016 con remanentes de Earl; "
              "la hora exacta no se reporta (se toma 03:00 ± 3 h). 24 muertos en Huauchinango "
              "(ERN, boletín del 8 de agosto de 2016)."),
    ),
}


def t_event_utc(case: str) -> pd.Timestamp:
    d = DESLIZAMIENTOS[case]
    return pd.Timestamp(d["t_local"]) - pd.Timedelta(hours=d["utc_offset_h"])


# etiqueta corta (NetCDF) -> (nombre en figuras, overlay)
EXPERIMENTS: dict[str, tuple[str, str]] = {
    "v020": ("v0.2.0", "v020.yaml"),
    "p95": ("P95 máximo", "p95.yaml"),
    "w12": ("corte 12 m/s", "w12.yaml"),
    "ss": ("v0.2.1 Saffir-Simpson", "ss.yaml"),
    "p95ss": ("P95 + Saffir-Simpson", "p95_ss.yaml"),
}
# variantes de lluvia (para H_P continuo, sin necesidad de posición del ciclón)
PRECIP_VARIANTS = {"P99": "v020", "P95": "p95"}

# Recorte de la malla: México con 1° de margen
MEXICO_BOX = dict(lon_min=-119.0, lon_max=-85.0, lat_min=13.5, lat_max=33.5)


def local_path(path) -> Path:
    # "D:/..." -> "/mnt/d/..." en WSL/Linux; sin cambios en Windows
    s = str(path).replace("\\", "/")
    if os.name != "nt" and re.match(r"^[A-Za-z]:/", s):
        return Path(f"/mnt/{s[0].lower()}") / s[3:]
    return Path(s)


@dataclass
class ConfigV021:
    mswep_6h_dir: Path = local_path("D:/MSWEP_V316_6H")
    clim: dict = field(default_factory=lambda: {
        d: ROOT / "data" / "processed" / f"mswep_v3_{d}h_percentiles_may_nov.nc" for d in (6, 24, 72)})
    clim_var: str = "p_wet_raw"
    tracks_dir: Path = ROOT / "data" / "raw" / "tracks"
    rocloud_dir: Path = ROOT / "data" / "external" / "Rocloud"
    mun_shp: Path = ROOT / "data" / "external" / "mun23gw" / "mun23gw.shp"
    decl_file: Path = ROOT / "outputs" / "diagnostics" / "declaratorias_6h" / "declaratorias_ct_eventos.csv"
    experiments_dir: Path = ROOT / "configs" / "experiments" / "v021"
    cache_dir: Path = ROOT / "outputs" / "diagnostics" / "avance_v021"
    fig_dir: Path = ROOT / "outputs" / "figures" / "avance_v021"
    box: dict = field(default_factory=lambda: dict(MEXICO_BOX))
    # H_P continuo: pasos de 6 h antes de la primera y después de la última posición
    hp_pad_before_h: int = 48
    hp_pad_after_h: int = 72
    municipal_stat: str = "max"            # "max" | "mean" | "p90"


# ============================================================================
# EXPERIMENTOS
# ============================================================================


def experiment_settings(cfg: ConfigV021 = ConfigV021()) -> dict[str, dict]:
    """Argumentos de compute_itchi_snapshot_v02 de cada experimento."""
    return {tag: itchi_v02_settings_from_config(load_experiment_config(cfg.experiments_dir / f))
            for tag, (_, f) in EXPERIMENTS.items()}


def label(tag: str) -> str:
    return EXPERIMENTS[tag][0]


# ============================================================================
# TRAYECTORIA Y SELECCIÓN DE POSICIONES
# ============================================================================


def load_track(case: str, cfg: ConfigV021 = ConfigV021()) -> pd.DataFrame:
    """Trayectoria sinóptica IBTrACS con R34, radios W06 PA21 y ROCLOUD (notebooks 05–08)."""
    import itchi.itchi_size_helpers as H

    c = CASES[case]
    ibtracs = cfg.tracks_dir / f"IBTrACS.{c['basin']}.v04r01.nc"
    rocloud = cfg.rocloud_dir / f"ROCLOUD_{c['basin']}.csv"
    track = H.load_case_track(ibtracs, rocloud, c["name"], c["season"], c["atcf"])
    return track.dropna(subset=["lat", "lon", "vmax_kt"]).reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dlam = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def max_attribution_radius_km(row: pd.Series) -> float:
    """Radio de atribución mayor de los cuadrantes: max(ROCLOUD_q, R_W06) (regla 'floor')."""
    roc = [row.get(DEFAULT_ROCLOUD_COLUMN_MAP[q], np.nan) for q in QUADRANTS]
    roc = [float(v) for v in roc if v is not None and np.isfinite(v) and v > 0]
    r_w06 = float(row.get("pa21_r2_km", np.nan))
    cands = roc + ([r_w06] if np.isfinite(r_w06) else [])
    return max(cands) if cands else np.nan


def select_positions(track: pd.DataFrame, land_lat: np.ndarray, land_lon: np.ndarray) -> pd.DataFrame:
    """
    Posiciones cuya región de atribución alcanza algún píxel terrestre de México.
    Agrega 'r_attr_max_km' y 'dist_tierra_km' (distancia del centro al píxel
    terrestre más cercano).
    """
    track = track.copy()
    track["r_attr_max_km"] = track.apply(max_attribution_radius_km, axis=1)
    dist = np.array([haversine_km(r["lat"], r["lon"], land_lat, land_lon).min() for _, r in track.iterrows()])
    track["dist_tierra_km"] = dist
    keep = np.isfinite(track["r_attr_max_km"]) & (dist <= track["r_attr_max_km"])
    return track[keep].reset_index(drop=True)


# ============================================================================
# LLUVIA Y PERCENTILES
# ============================================================================


def read_precip(times: pd.DatetimeIndex, cfg: ConfigV021 = ConfigV021()):
    """
    P6, R24 y A72 en todos los tiempos de 6 h entre min(times) - pad y
    max(times) + pad (sin huecos), recortados a cfg.box.
    """
    from itchi.mswep import subset_mswep_domain

    t0 = times.min() - pd.Timedelta(hours=cfg.hp_pad_before_h)
    t1 = times.max() + pd.Timedelta(hours=cfg.hp_pad_after_h)
    years = sorted({t0.year, t1.year})
    parts = []
    for y in years:
        path = cfg.mswep_6h_dir / f"mswep_v3_6h_{y}.nc"
        if not path.exists():
            raise FileNotFoundError(f"no existe {path}")
        with xr.open_dataset(path) as ds:
            p = subset_mswep_domain(ds["precipitation"], **cfg.box)
            parts.append(p.sel(time=slice(t0 - pd.Timedelta(hours=84), t1)).load())
    p6 = xr.concat(parts, dim="time") if len(parts) > 1 else parts[0]
    missing = times.difference(pd.DatetimeIndex(p6["time"].values))
    if not missing.empty:
        raise ValueError(f"tiempos de la trayectoria sin acumulado de 6 h: {list(missing)[:5]} ...")
    r24 = compute_antecedent_precipitation(p6, hours=24, exclude_current=False)
    a72 = compute_antecedent_precipitation(p6, hours=72, exclude_current=True)
    keep = (p6["time"] >= np.datetime64(t0)) & (p6["time"] <= np.datetime64(t1))
    return tuple(da.sel(time=keep).transpose("time", "lat", "lon") for da in (p6, r24, a72))


def read_climatologies(template: xr.DataArray, cfg: ConfigV021 = ConfigV021()) -> tuple[dict, dict, dict]:
    """Percentiles de 6, 24 y 72 h en la malla de template."""
    return tuple(load_percentile_climatology(cfg.clim[d], variable=cfg.clim_var, template=template) for d in (6, 24, 72))


def precipitation_hazard(p6, r24, a72, q6, q24, q72, spec: dict, nan_value: float | None = 0.0):
    hp = compute_precipitation_hazard_from_spec(
        precipitation=p6, quantiles_6h=q6,
        escalation_values={"R24": r24, "A72": a72},
        escalation_quantiles={"R24": q24, "A72": q72}, spec=spec)
    if nan_value is None:
        return hp
    return hp.fillna(nan_value) if isinstance(hp, xr.DataArray) else np.where(np.isfinite(hp), hp, nan_value)


# ============================================================================
# MUNICIPIOS
# ============================================================================


def load_municipalities(lat: np.ndarray, lon: np.ndarray, cfg: ConfigV021 = ConfigV021()):
    """
    Municipios cuyo punto representativo está en la malla y sus píxeles.
    Devuelve (gdf, pix) con pix[CVEGEO] = (índices planos, n de centros dentro).
    """
    import geopandas as gpd

    from itchi.declaratorias_6h import municipio_pixels

    gdf = gpd.read_file(cfg.mun_shp)
    gdf["CVEGEO"] = gdf["CVEGEO"].astype(str).str.zfill(5)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    rp = gdf.geometry.representative_point()
    sub = gdf[rp.x.between(lon.min(), lon.max()) & rp.y.between(lat.min(), lat.max())].reset_index(drop=True)
    pix = municipio_pixels(sub, lat, lon, sub["CVEGEO"])
    return sub, pix


def land_mask(pix: dict, shape: tuple[int, int]) -> np.ndarray:
    m = np.zeros(int(np.prod(shape)), bool)
    for flat, _ in pix.values():
        m[flat] = True
    return m.reshape(shape)


def municipal_values(field: np.ndarray, pix: dict, codes, stat: str = "max") -> np.ndarray:
    """
    (T, ny, nx) o (ny, nx) -> (municipios, T) o (municipios,): estadístico de
    los píxeles de cada municipio. NaN cuenta como 0 (fuera de la región).
    """
    squeeze = field.ndim == 2
    f = np.nan_to_num(np.asarray(field, dtype=float)[None] if squeeze else np.asarray(field, dtype=float))
    flat = f.reshape(f.shape[0], -1)
    flats = [pix[c][0] for c in codes]
    if stat in ("max", "mean"):
        sizes = np.array([len(x) for x in flats])
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
        sel = flat[:, np.concatenate(flats)]
        if stat == "max":
            out = np.maximum.reduceat(sel, starts, axis=1).T
        else:
            out = (np.add.reduceat(sel, starts, axis=1) / sizes).T
    elif stat == "p90":
        out = np.stack([np.percentile(flat[:, x], 90, axis=1) for x in flats])
    else:
        raise ValueError(f"stat desconocido: {stat!r}")
    return out[:, 0] if squeeze else out


def paint_municipal(values: np.ndarray, pix: dict, codes, shape: tuple[int, int]) -> np.ndarray:
    """
    Valor municipal pintado en la malla (cada píxel toma el valor de su
    municipio). Los municipios con centros propios (n > 0) tienen prioridad
    sobre los que tomaron el píxel más cercano (n = 0).
    """
    out = np.full(int(np.prod(shape)), np.nan)
    order = sorted(range(len(codes)), key=lambda i: pix[codes[i]][1] > 0)   # n = 0 primero, se sobrescriben
    for i in order:
        out[pix[codes[i]][0]] = values[i]
    return out.reshape(shape)


def siat_class(x) -> np.ndarray:
    """Clase SIAT 0–5 en tramos de 0.2, (inferior, superior]; NaN -> 0."""
    return np.asarray(classify_itchi_to_siat_levels(np.nan_to_num(np.asarray(x, dtype=float)))).astype(np.int8)


# ============================================================================
# CÁLCULO POR CASO
# ============================================================================


def compute_case_fields(track: pd.DataFrame, p6: xr.DataArray, r24: xr.DataArray, a72: xr.DataArray,
                        q6: dict, q24: dict, q72: dict, settings: dict[str, dict],
                        verbose: bool = True) -> xr.Dataset:
    """
    ITCHI de cada experimento en las posiciones de track (tiempos dentro de
    p6) y H_P continuo de cada variante de lluvia en todos los tiempos de p6.

    No lee archivos: track es la trayectoria aumentada (load_track +
    select_positions) y p6, r24, a72 son DataArrays (time, lat, lon).
    """
    from itchi.pipeline import compute_itchi_snapshot_v02

    lat, lon = p6["lat"].values, p6["lon"].values
    lon2d, lat2d = np.meshgrid(lon, lat)
    t_all = pd.DatetimeIndex(p6["time"].values)
    t_idx = t_all.get_indexer(pd.DatetimeIndex(track["time"]))
    if np.any(t_idx < 0):
        raise ValueError("hay posiciones fuera del intervalo de lluvia leído")

    q_np = [{k: np.asarray(v) for k, v in q.items()} for q in (q6, q24, q72)]
    tags = list(settings)
    nt, shape = len(track), lon2d.shape
    store = {k: np.zeros((len(tags), nt, *shape), np.float32) for k in ("ITCHI", "H_P", "H_W")}
    m_dir = np.zeros((nt, *shape), np.int8)
    m_ind = np.zeros((nt, *shape), np.int8)
    wind = np.zeros((nt, *shape), np.float32)
    meta = {k: np.full(nt, np.nan) for k in ("r_w06_max_km", "r_attr_max_km_pipeline", "r_direct_max_km")}
    ok = np.ones(nt, bool)

    t0 = time.time()
    for i, (_, row) in enumerate(track.iterrows()):
        j = t_idx[i]
        rmw = float(row["rmw_km"]) if np.isfinite(row.get("rmw_km", np.nan)) else None
        roc = {q: row.get(DEFAULT_ROCLOUD_COLUMN_MAP[q], np.nan) for q in QUADRANTS}
        common = dict(
            quantiles_6h=q_np[0], quantiles_24h=q_np[1], quantiles_72h=q_np[2],
            lon=lon2d, lat=lat2d, center_lon=float(row["lon"]), center_lat=float(row["lat"]),
            vmax_kt=float(row["vmax_kt"]), rmw_km=rmw,
            r34_by_quadrant={q: row.get(DEFAULT_R34_COLUMN_MAP[q], np.nan) for q in QUADRANTS},
            r34_unit="nm", rocloud_by_quadrant=roc, rocloud_unit="km",
        )
        for e, tag in enumerate(tags):
            try:
                res = compute_itchi_snapshot_v02(p6.values[j], r24.values[j], a72.values[j], **common, **settings[tag])
            except ValueError as exc:
                warnings.warn(f"posición {row['time']}: {exc}; se omite", stacklevel=2)
                ok[i] = False
                break
            for k in store:
                store[k][e, i] = np.nan_to_num(np.asarray(res[k], dtype=np.float32))
            if e == 0:
                m_dir[i] = np.asarray(res["M_direct"], bool)
                m_ind[i] = np.asarray(res["M_indirect"], bool)
                wind[i] = np.asarray(res["wind_kt"], np.float32)
                meta["r_w06_max_km"][i] = max(res["R_W06_by_quadrant"].values())
                meta["r_attr_max_km_pipeline"][i] = max(res["R_attr_by_quadrant"].values())
                meta["r_direct_max_km"][i] = max(res["R_direct_by_quadrant"].values())
        if verbose:
            print(f"\r  {i + 1}/{nt} posiciones | {time.time() - t0:.0f} s", end="", flush=True)
    if verbose:
        print()

    # H_P continuo (no depende de la posición del ciclón)
    hp_cont = {}
    for name, tag in PRECIP_VARIANTS.items():
        if tag in settings:
            hp = precipitation_hazard(p6.values, r24.values, a72.values, *q_np,
                                      spec=settings[tag]["precip_matrix"],
                                      nan_value=settings[tag]["precipitation_nan_value"])
            hp_cont[name] = np.asarray(hp, np.float32)

    keep = np.flatnonzero(ok)
    tr = track.iloc[keep].reset_index(drop=True)
    data = {
        **{k: (("exp", "time", "lat", "lon"), v[:, keep]) for k, v in store.items()},
        "M_direct": (("time", "lat", "lon"), m_dir[keep]),
        "M_indirect": (("time", "lat", "lon"), m_ind[keep]),
        "wind_kt": (("time", "lat", "lon"), wind[keep]),
        "P6": (("time_all", "lat", "lon"), p6.values.astype(np.float32)),
        "R24": (("time_all", "lat", "lon"), r24.values.astype(np.float32)),
        "A72": (("time_all", "lat", "lon"), a72.values.astype(np.float32)),
        "track_lon": ("time", tr["lon"].to_numpy(float)),
        "track_lat": ("time", tr["lat"].to_numpy(float)),
        "track_vmax_kt": ("time", tr["vmax_kt"].to_numpy(float)),
        **{k: ("time", v[keep]) for k, v in meta.items()},
    }
    for name, v in hp_cont.items():
        data[f"H_P_cont_{name}"] = (("time_all", "lat", "lon"), v)
    for k, v in zip((6, 24, 72), q_np, strict=True):
        for name, f in v.items():
            data[f"Q{k}h_{name}"] = (("lat", "lon"), np.asarray(f, np.float32))
    ds = xr.Dataset(data, coords={"exp": tags, "time": tr["time"].values, "time_all": t_all.values,
                                  "lat": lat, "lon": lon})
    return ds


def case_grid(season: int, cfg: ConfigV021 = ConfigV021()) -> tuple[np.ndarray, np.ndarray]:
    """lat y lon del recorte cfg.box en el archivo de 6 h de la temporada."""
    from itchi.mswep import subset_mswep_domain

    with xr.open_dataset(cfg.mswep_6h_dir / f"mswep_v3_6h_{season}.nc") as ds:
        p = subset_mswep_domain(ds["precipitation"].isel(time=0, drop=True), **cfg.box)
        return p["lat"].values.copy(), p["lon"].values.copy()


def load_or_compute(case: str, cfg: ConfigV021 = ConfigV021(), force: bool = False, verbose: bool = True):
    """
    Campos del caso (caché en cfg.cache_dir), trayectoria completa, municipios.

    Devuelve dict con: ds (xr.Dataset), track (todas las posiciones),
    gdf, pix, codes y land (máscara terrestre).
    """
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    f = cfg.cache_dir / f"{case.lower()}.nc"
    track_all = load_track(case, cfg)

    # Malla del recorte: la del archivo de 6 h de la temporada (la misma de P6)
    lat, lon = case_grid(CASES[case]["season"], cfg)
    gdf, pix = load_municipalities(lat, lon, cfg)
    codes = list(pix)

    ds = None
    if f.exists() and not force:
        ds = xr.load_dataset(f)
        if verbose:
            print(f"{case}: leyendo {f.name} ({ds.sizes['time']} posiciones)")
    if ds is None:
        land = land_mask(pix, (lat.size, lon.size))
        lon2d, lat2d = np.meshgrid(lon, lat)
        track = select_positions(track_all, lat2d[land], lon2d[land])
        if track.empty:
            raise ValueError(f"{case}: ninguna posición alcanza México")
        if verbose:
            print(f"{case}: {len(track)} de {len(track_all)} posiciones alcanzan México; calculando", flush=True)
        p6, r24, a72 = read_precip(pd.DatetimeIndex(track["time"]), cfg)
        template = p6.isel(time=0, drop=True)
        q6, q24, q72 = read_climatologies(template, cfg)
        ds = compute_case_fields(track, p6, r24, a72, q6, q24, q72, experiment_settings(cfg), verbose)
        ds.attrs.update(case=case, municipal_stat=cfg.municipal_stat,
                        experiments="; ".join(f"{k}: {v[0]} ({v[1]})" for k, v in EXPERIMENTS.items()))
        ds.to_netcdf(f, encoding={v: {"zlib": True, "complevel": 2} for v in ds.data_vars})
        if verbose:
            print(f"  guardado {f} ({f.stat().st_size / 1e6:.0f} MB)")

    if not (np.allclose(ds["lat"].values, lat) and np.allclose(ds["lon"].values, lon)):
        warnings.warn("la malla del caché no coincide con la de los municipios; usa force=True", stacklevel=2)
    return dict(case=case, ds=ds, track=track_all, gdf=gdf, pix=pix, codes=codes,
                land=land_mask(pix, (lat.size, lon.size)))


# ============================================================================
# RESÚMENES
# ============================================================================


def municipal_event(C: dict, field: str = "ITCHI", stat: str | None = None) -> xr.DataArray:
    """
    Valor municipal por experimento y tiempo: DataArray (exp, mun, time).
    """
    ds = C["ds"]
    stat = stat or ds.attrs.get("municipal_stat", "max")
    vals = np.stack([municipal_values(ds[field].sel(exp=e).values, C["pix"], C["codes"], stat)
                     for e in ds["exp"].values])
    return xr.DataArray(vals, dims=("exp", "mun", "time"),
                        coords={"exp": ds["exp"].values, "mun": C["codes"], "time": ds["time"].values})


def class_counts(C: dict, tags=None) -> pd.DataFrame:
    """
    Municipios y área terrestre (píxeles) por clase del máximo del evento.
    """
    ds = C["ds"]
    tags = list(ds["exp"].values) if tags is None else list(tags)
    mun = municipal_event(C).max("time")
    rows = []
    for e in tags:
        cm = siat_class(mun.sel(exp=e).values)
        cp = siat_class(ds["ITCHI"].sel(exp=e).max("time").values)[C["land"]]
        for k in range(1, 6):
            rows.append(dict(exp=label(e), clase=k, municipios=int((cm == k).sum()),
                             pixeles_tierra=int((cp == k).sum())))
    out = pd.DataFrame(rows).pivot_table(index="clase", columns="exp", values=["municipios", "pixeles_tierra"])
    return out


def lead_times(times, classes, t_event, levels=(2, 3, 4, 5), window_h: float | None = 120.0,
               step_h: float = 6.0) -> pd.DataFrame:
    """
    Anticipación de cada nivel k respecto a t_event, con los tiempos <= t_event
    (ventanas de 6 h que terminan antes del evento) dentro de window_h horas.

    Columnas:
      primer_tiempo         primer tiempo de la ventana con clase >= k
      inicio_episodio       inicio del último episodio continuo con clase >= k
                            (pasos consecutivos de step_h)
      fin_episodio          último tiempo de ese episodio
      anticipacion_h        t_event - inicio_episodio
      horas_desde_fin       t_event - fin_episodio: si es < step_h, el episodio
                            seguía activo al ocurrir el evento; si es mayor, el
                            peligro había bajado (o la serie se interrumpe, p. ej.
                            sin posición del ciclón)
    """
    times = pd.DatetimeIndex(times)
    classes = np.asarray(classes)
    te = pd.Timestamp(t_event)
    sel = times <= te
    if window_h is not None:
        sel &= times >= te - pd.Timedelta(hours=window_h)
    tt, cc = times[sel], classes[sel]
    hour = pd.Timedelta(hours=1)
    rows = []
    for k in levels:
        hit = cc >= k
        row = dict(nivel=k, primer_tiempo=pd.NaT, inicio_episodio=pd.NaT, fin_episodio=pd.NaT,
                   anticipacion_h=np.nan, horas_desde_fin=np.nan)
        if hit.any():
            idx = np.flatnonzero(hit)
            last = j = idx[-1]
            while j > 0 and hit[j - 1] and (tt[j] - tt[j - 1]) == pd.Timedelta(hours=step_h):
                j -= 1
            row.update(primer_tiempo=tt[idx[0]], inicio_episodio=tt[j], fin_episodio=tt[last],
                       anticipacion_h=(te - tt[j]) / hour, horas_desde_fin=(te - tt[last]) / hour)
        rows.append(row)
    return pd.DataFrame(rows)


def peak_time(C: dict, tag: str = "v020") -> pd.Timestamp:
    """Posición con mayor ITCHI acumulado sobre tierra (pico del evento sobre México)."""
    ds = C["ds"]
    total = (ds["ITCHI"].sel(exp=tag).values * C["land"][None]).reshape(ds.sizes["time"], -1).sum(axis=1)
    return pd.Timestamp(ds["time"].values[int(np.argmax(total))])


def load_declaratorias(cfg: ConfigV021 = ConfigV021()) -> pd.DataFrame | None:
    """Declaratorias asociadas a ciclones (scripts/declaratorias_6h.py); None si no existe el archivo."""
    if not cfg.decl_file.exists():
        warnings.warn(f"no existe {cfg.decl_file}: sin comparación con declaratorias", stacklevel=2)
        return None
    decl = pd.read_csv(cfg.decl_file, dtype={"CVEGEO": str})
    decl["CVEGEO"] = decl["CVEGEO"].str.zfill(5)
    return decl


def declaratoria_inicio(decl: pd.DataFrame | None, atcf: str, local_offset_h: int = 6) -> pd.Series:
    """
    Fecha de inicio más temprana de la declaratoria por municipio para el
    ciclón atcf (CT_pico), a las 00 h locales expresadas en UTC (+6 h, como
    en el notebook 07). Serie vacía sin archivo.
    """
    if decl is None:
        return pd.Series(dtype="datetime64[ns]")
    d = decl[decl["CT_pico"] == atcf]
    return pd.to_datetime(d["fecha_inicio"]).groupby(d["CVEGEO"]).min() + pd.Timedelta(hours=local_offset_h)


def tabla_ordinal(C: dict, decl_codes, tags=None) -> pd.DataFrame:
    """
    Prueba ordinal (plan v0.3, P2): por experimento y clase del máximo del
    evento, municipios en la clase, cuántos tuvieron declaratoria y P(D).
    Solo municipios dentro de la región de atribución en algún paso.
    """
    ds = C["ds"]
    tags = list(ds["exp"].values) if tags is None else list(tags)
    mun = municipal_event(C).max("time")
    region = (ds["M_direct"] | ds["M_indirect"]).any("time").values
    in_region = np.array([region.ravel()[C["pix"][c][0]].any() for c in C["codes"]])
    has_d = np.isin(np.asarray(C["codes"]), list(decl_codes))
    rows = []
    for e in tags:
        cls = siat_class(mun.sel(exp=e).values)
        for k in range(0, 6):
            m = in_region & (cls == k)
            rows.append(dict(exp=label(e), clase=k, municipios=int(m.sum()), con_declaratoria=int((m & has_d).sum()),
                             P_D=(m & has_d).sum() / m.sum() if m.sum() else np.nan))
    return pd.DataFrame(rows)


def find_municipio(gdf, nombre: str, cve_ent: str) -> str:
    """CVEGEO del municipio por nombre (sin acentos ni mayúsculas) y clave de entidad."""
    import unicodedata

    def norm(x):
        return "".join(ch for ch in unicodedata.normalize("NFKD", str(x)) if not unicodedata.combining(ch)).lower().strip()

    ent = gdf["CVEGEO"].astype(str).str.zfill(5).str[:2]
    hit = gdf[(ent == str(cve_ent).zfill(2)) & (gdf["NOMGEO"].map(norm) == norm(nombre))]
    if len(hit) != 1:
        raise ValueError(f"{nombre} ({cve_ent}): {len(hit)} coincidencias")
    return str(hit["CVEGEO"].iloc[0]).zfill(5)


def percentile_level(values: np.ndarray, quantiles: dict[str, np.ndarray | float]) -> np.ndarray:
    """Percentil más alto superado (75, 90, ...; 0 si ninguno) por elemento."""
    names = sorted(quantiles, key=lambda k: float(k[1:]))
    out = np.zeros(np.shape(values))
    for k in names:
        out = np.where(np.asarray(values) >= np.asarray(quantiles[k]), float(k[1:]), out)
    return out


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Avance v0.2.1: ITCHI por caso en malla y por municipio")
    ap.add_argument("--casos", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--forzar", action="store_true", help="recalcula aunque exista el caché")
    args = ap.parse_args()
    cfg = ConfigV021()
    for case in args.casos:
        t0 = time.time()
        C = load_or_compute(case, cfg, force=args.forzar)
        print(class_counts(C).to_string())
        print(f"{case}: {time.time() - t0:.0f} s\n")
