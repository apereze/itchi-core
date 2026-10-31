"""
ITCHI-CORE · Precipitación de 6 h (MSWEP V3) durante declaratorias asociadas a CT

Exploración: cuando hubo una declaratoria (emergencia, desastre o contingencia)
asociada a un ciclón tropical, ¿cuánta lluvia de 6 h hubo en el municipio y
cuál es el mínimo entre todas las declaratorias?

Asociación declaratoria–CT (igual que en el artículo de tamaño de CT y lluvia
extrema, tc-size-extreme-rainfall-mexico):

    1. Declaratorias "Hidrometeorológico" cuyas Observaciones mencionan
       ciclón tropical / huracán / tormenta tropical / depresión tropical.
    2. [Fecha Inicio, Fecha Fin] se expande a 00, 06, 12, 18 UTC de cada día.
    3. Se conservan los pasos con al menos una posición de CT (ROCLOUD NA/EP)
       a la misma hora. Si hay varios CT activos se guarda el más cercano al
       municipio (distancia, dentro de ROCLOUD por cuadrante, Rp).

Precipitación (scripts/mswep_v3_6h_accum.py):

    P(T) = acumulado en (T-6h, T], T = hora de la posición del CT.

Agregación:

    paso      (declaratoria, T): P sobre los píxeles del municipio
              (máximo, media, fracción de área sobre cada percentil).
    evento    (declaratoria): máximo de los pasos -> P_evento (pico de 6 h) y su
              distancia en mm a los percentiles del mismo píxel: dif_Pxx = P_evento - Pxx.
    municipio mínimo de P_evento entre todas sus declaratorias.
    píxel     mínimo entre declaratorias del máximo de 6 h en ese píxel.

Los percentiles se leen de data/processed/mswep_v3_6h_percentiles_may_nov.nc
(los que contenga el archivo; actualmente P75, P90, P95, P99, P99.9).

Multiduración (requiere percentiles_6h.py --durations 24 72):

    R24(T), R72(T)  acumulados inclusivos en (T-24h, T] y (T-72h, T]
    A72(T)          antecedente exclusivo en (T-78h, T-6h] = R72(T-6h)
                    (misma climatología que R72)

    Se calculan sobre la temporada completa del año, de modo que el
    antecedente incluye lluvia previa a Fecha Inicio y lluvia no asociada
    al CT. Para cada serie se guarda:

    *_en_pico6        valor y clase en el píxel y paso del pico de 6 h
    *_evento          máximo propio sobre píxeles y pasos de la declaratoria,
                      con su clase en su propio píxel; nivel_*_area_evento
                      es la clase máxima sobre el área

    Niveles combinados (en el pico de 6 h):

    nivel_multidur_pico  opción A: máximo de las clases de P6, R24 y R72
    nivel_base_pico      rampa solo 6 h: <P75, P75, P90, P95, P99 -> 0..4
    nivel_matriz_pico    opción B: rampa base escalada; P95-P99 -> máximo si R24 >= P95 y A72 >= P75
                         (calibrada con negativos_w06.py; ver Config.escalada_*)

Salidas (outputs/diagnostics/declaratorias_6h/):

    declaratorias_ct_pasos.csv
    declaratorias_ct_eventos.csv
    declaratorias_ct_municipios.csv  (+ .gpkg si hay driver)
    declaratorias_ct_pixel_min.nc
    declaratorias_ct_pixel_decl.csv.gz    declaratoria x píxel: máximo de 6 h, hora y CT
    declaratorias_ct_pixel_stats.nc       por píxel y tipo: n, mín, mediana, media, máx + Identificador
    declaratorias_ct_municipios_stats.csv por municipio y tipo: mín, mediana, media, máx + declaratoria/CT
    declaratorias_ct_por_ciclon.csv       por CT: declaratorias asociadas y su lluvia
    declaratorias_ct_multiduracion_resumen.csv   escalamiento y saturación por tipo
    declaratorias_ct_matriz_P6_A72.csv           tabla cruzada clase P6 x clase A72 por tipo
    qc_claves_sin_geometria.csv

Ejecutar desde scripts/:
    python declaratorias_6h.py
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

# Raíz del repo: primer directorio superior con pyproject.toml (sirve desde scripts/ o src/itchi/,
# en Windows/conda y en WSL).
REPO = next((p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
            Path(__file__).resolve().parents[1])
# ============================================================================
# CONFIGURACIÓN
# ============================================================================


@dataclass
class Config:

    # ------------------------------------------------------------------------
    # Entradas (data/external no se versiona; ver data/external/README_declaratorias.md)
    # ------------------------------------------------------------------------

    decl_csv: str = str(REPO / "data/external/declaratorias/declaratorias27052025.csv")
    inafed_csv: str = str(REPO / "data/external/declaratorias/inafed_bd_1748479609.csv")
    mun_shp: str = str(REPO / "data/external/mun23gw/mun23gw.shp")
    rocloud_dir: str = str(REPO / "data/external/Rocloud")

    mswep_6h_dir: str = "D:/MSWEP_V316_6H"
    percentiles_file: str = str(REPO / "data/processed/mswep_v3_6h_percentiles_may_nov.nc")

    # p_wet_raw: sin filtro de suficiencia. p_wet deja P99.9 casi todo en NaN
    # (requiere >= 10 000 ventanas húmedas por píxel).
    pct_var: str = "p_wet_raw"

    # ------------------------------------------------------------------------
    # Multiduración (percentiles_6h.py --durations 24 72)
    # ------------------------------------------------------------------------

    # Acumulados inclusivos R_d y antecedentes exclusivos A_d (h, múltiplos de 6)
    durations_h: tuple = (24, 72)
    antecedent_h: tuple = (72,)

    percentiles_dur_pattern: str = str(REPO / "data/processed/mswep_v3_{d}h_percentiles_may_nov.nc")

    # p_wet_raw por consistencia con 6 h. En 72 h P99.9 no es robusto (traslape de
    # ventanas); "p_wet" aplica el filtro de suficiencia efectivo y lo deja en NaN,
    # con lo que la clase máxima alcanzable en 72 h es P99.
    pct_var_dur: str = "p_wet_raw"

    # Niveles 0 sin, 1 bajo, 2 medio, 3 alto, 4 máximo. Rampa base por clase de P6
    # (<P75, P75-P90, P90-P95, P95-P99, P99-P99.9, >=P99.9) y escalada (opción B)
    # calibrada con pares municipio-CT (negativos_w06.py): las clases de P6 en
    # escalada_clases_P6 suben a escalada_nivel si se cumplen las condiciones
    # (variable, percentil mínimo); "y" exige todas, "o" basta una. Una condición
    # sin dato (p. ej. inicio de temporada) no escala.
    rampa_base: tuple = (0, 1, 2, 3, 4, 4)
    escalada_clases_P6: tuple = (3,)
    escalada_nivel: int = 4
    escalada_condiciones: tuple = (("R24", "P95"), ("A72", "P75"))
    escalada_modo: str = "y"

    start_year: int = 2000
    end_year: int = 2025

    # ------------------------------------------------------------------------
    # Filtro de declaratorias (criterio del artículo)
    # ------------------------------------------------------------------------

    clasificacion: str = "Hidrometeorológico"
    tc_keywords: tuple = ("ciclon tropical", "huracan", "tormenta tropical", "depresion tropical")

    # Observaciones que mencionan lluvia o sus efectos. Las declaratorias solo por
    # viento o marea (p. ej. "Huracán Delta" en Cozumel) dan mínimos espurios.
    rain_keywords: tuple = ("lluvia", "inundacion", "remocion", "deslave", "deslizamiento")
    solo_lluvia: bool = False

    # ------------------------------------------------------------------------
    # Asociación espacial opcional (None/False = solo coincidencia temporal,
    # como en el artículo)
    # ------------------------------------------------------------------------

    max_dist_km: float | None = None
    require_inside_rocloud: bool = False

    # Tolerancia temporal: días que se agregan antes de Fecha Inicio y después de
    # Fecha Fin (0 = como en el artículo). Hay declaratorias cuyas fechas no cubren
    # el impacto (p. ej. Delta 2020: declaratoria 5-6 oct, entrada a tierra 7 oct).
    dias_antes: int = 0
    dias_despues: int = 0

    # ------------------------------------------------------------------------
    # Ventana de análisis (México); acelera la lectura
    # ------------------------------------------------------------------------

    lat_bounds: tuple = (14.0, 33.5)
    lon_bounds: tuple = (-118.5, -86.0)

    shp_encoding: str = "utf-8"

    out_dir: str = str(REPO / "outputs/diagnostics/declaratorias_6h")


TRACK_COLUMNS = [
    "dd", "mm", "yy", "hh", "lat", "lon", "MWS", "CPSL",
    "RNE", "RNO", "RSO", "RSE", "Rp", "A", "D", "S", "CT",
]

EARTH_RADIUS_KM = 6371.0
SYNOPTIC_HOURS = (0, 6, 12, 18)

# ============================================================================
# DECLARATORIAS
# ============================================================================


def _strip_accents(text):

    if not isinstance(text, str):
        return text
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def load_declaratorias(cfg: Config) -> pd.DataFrame:
    """Declaratorias hidrometeorológicas asociadas a CT por texto (Observaciones)."""

    db = pd.read_csv(cfg.decl_csv, encoding="latin1")

    h2o = db[db["Clasificación Fenómeno"].eq(cfg.clasificacion)].copy()
    obs = h2o["Observaciones"].str.lower().map(_strip_accents)
    tc = h2o[obs.str.contains("|".join(cfg.tc_keywords), na=False)].copy()

    obs_tc = obs.loc[tc.index]
    tc["menciona_lluvia"] = obs_tc.str.contains("|".join(cfg.rain_keywords), na=False)
    if cfg.solo_lluvia:
        tc = tc[tc["menciona_lluvia"]].copy()

    tc["id_decl"] = np.arange(len(tc))  # Identificador no es único
    tc["CVEGEO"] = tc["Clave Municipio"].astype(int).astype(str).str.zfill(5)
    tc["nombre_ct"] = tc["Observaciones"].str.extract(r'["“”]\s*([^"“”]+?)\s*["“”]', expand=False).str.upper()

    fi = pd.to_datetime(tc["Fecha Inicio"], dayfirst=True, errors="coerce")
    ff = pd.to_datetime(tc["Fecha Fin"], dayfirst=True, errors="coerce").fillna(fi)

    # Fechas invertidas en la base: se intercambian y se marcan.
    inv = ff < fi
    tc["fechas_invertidas"] = inv
    tc["fecha_inicio"] = fi.where(~inv, ff)
    tc["fecha_fin"] = ff.where(~inv, fi)

    yrs = tc["fecha_inicio"].dt.year
    tc = tc[(yrs >= cfg.start_year) & (yrs <= cfg.end_year)]

    print(f"Declaratorias asociadas a CT: {len(tc):,} (mencionan lluvia: {int(tc['menciona_lluvia'].sum()):,}) "
          f"({tc['Tipo Declaratoria'].value_counts().to_dict()}) | "
          f"fechas invertidas: {int(tc['fechas_invertidas'].sum())}")

    return tc.reset_index(drop=True)


def expand_6h(decl: pd.DataFrame, dias_antes: int = 0, dias_despues: int = 0) -> pd.DataFrame:
    """[fecha_inicio - dias_antes, fecha_fin + dias_despues] -> 00/06/12/18 UTC (como generar_series_6h)."""

    steps = decl[["id_decl", "CVEGEO", "fecha_inicio", "fecha_fin"]].copy()
    d0, d1 = pd.Timedelta(days=dias_antes), pd.Timedelta(days=dias_despues)
    steps["time"] = [
        pd.date_range(a.normalize() - d0, b.normalize() + d1 + pd.Timedelta(hours=18), freq="6h")
        for a, b in zip(steps["fecha_inicio"], steps["fecha_fin"])
    ]

    return steps.explode("time").drop(columns=["fecha_inicio", "fecha_fin"]).astype({"time": "datetime64[ns]"})


# ============================================================================
# TRAYECTORIAS (ROCLOUD)
# ============================================================================


def load_tracks(cfg: Config) -> pd.DataFrame:

    parts = []
    for basin in ("NA", "EP"):
        df = pd.read_csv(os.path.join(cfg.rocloud_dir, f"ROCLOUD_{basin}.csv"),
                         names=TRACK_COLUMNS, index_col=False)
        parts.append(df)

    tr = pd.concat(parts, ignore_index=True)
    tr["time"] = pd.to_datetime(dict(year=tr["yy"], month=tr["mm"], day=tr["dd"], hour=tr["hh"]))
    tr = tr[tr["hh"].isin(SYNOPTIC_HOURS)]

    for c in ("RNE", "RNO", "RSO", "RSE", "Rp"):
        tr[c] = pd.to_numeric(tr[c], errors="coerce").where(lambda s: s > 0)

    return tr[["time", "CT", "lat", "lon", "MWS", "RNE", "RNO", "RSO", "RSE", "Rp"]].rename(
        columns={"lat": "lat_ct", "lon": "lon_ct"})


def haversine_bearing(lat1, lon1, lat2, lon2):
    """Distancia (km) y rumbo inicial (grados desde el N, horario) de 1 -> 2."""

    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)

    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    dist = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))

    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    brg = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0

    return dist, brg


def associate_tracks(steps: pd.DataFrame, tracks: pd.DataFrame, mun_pts: pd.DataFrame,
                     cfg: Config) -> pd.DataFrame:
    """Pasos con CT activo; se conserva el CT más cercano al municipio."""

    m = steps.merge(tracks, on="time", how="inner").merge(mun_pts, on="CVEGEO", how="left")

    dist, brg = haversine_bearing(m["lat_ct"].values, m["lon_ct"].values,
                                  m["lat_mun"].values, m["lon_mun"].values)
    m["dist_km"] = dist
    m["bearing"] = brg

    # Radio ROCLOUD del cuadrante que contiene al municipio (NE, SE, SW, NW)
    quad = np.select([brg < 90, brg < 180, brg < 270], ["RNE", "RSE", "RSO"], "RNO")
    m["R_quad_km"] = m[["RNE", "RSE", "RSO", "RNO"]].to_numpy()[
        np.arange(len(m)), pd.Index(["RNE", "RSE", "RSO", "RNO"]).get_indexer(quad)]
    m["dentro_rocloud"] = m["dist_km"] <= m["R_quad_km"]
    m["dentro_Rp"] = m["dist_km"] <= m["Rp"]
    m["n_ct_activos"] = m.groupby(["id_decl", "time"])["CT"].transform("size")

    m = m.sort_values("dist_km").drop_duplicates(["id_decl", "time"], keep="first")

    if cfg.max_dist_km is not None:
        m = m[m["dist_km"] <= cfg.max_dist_km]
    if cfg.require_inside_rocloud:
        m = m[m["dentro_rocloud"]]

    keep = ["id_decl", "CVEGEO", "time", "CT", "lat_ct", "lon_ct", "MWS", "Rp",
            "dist_km", "R_quad_km", "dentro_rocloud", "dentro_Rp", "n_ct_activos"]

    return m[keep].sort_values(["id_decl", "time"]).reset_index(drop=True)


# ============================================================================
# MUNICIPIOS -> PÍXELES
# ============================================================================


def load_municipios(cfg: Config):

    import geopandas as gpd

    gdf = gpd.read_file(cfg.mun_shp, encoding=cfg.shp_encoding)
    gdf["CVEGEO"] = gdf["CVEGEO"].astype(str).str.zfill(5)

    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    return gdf[["CVEGEO", "NOMGEO", "NOM_ENT", "geometry"]]


def municipio_points(gdf) -> pd.DataFrame:

    rp = gdf.geometry.representative_point()
    return pd.DataFrame({"CVEGEO": gdf["CVEGEO"].values, "lat_mun": rp.y.values, "lon_mun": rp.x.values})


def municipio_pixels(gdf, lat: np.ndarray, lon: np.ndarray, codes) -> dict:
    """
    Índices planos (lat, lon) de los centros de píxel dentro de cada municipio.
    Municipio sin centros dentro -> píxel más cercano a su punto representativo
    (se reporta como n_pix = 0 en la tabla de municipios).
    """

    import shapely

    sub = gdf[gdf["CVEGEO"].isin(set(codes))]
    nx = lon.size
    out = {}

    for code, geom in zip(sub["CVEGEO"], sub.geometry):
        x0, y0, x1, y1 = geom.bounds
        jj = np.where((lat >= y0) & (lat <= y1))[0]
        ii = np.where((lon >= x0) & (lon <= x1))[0]
        flat = np.empty(0, dtype=int)
        if jj.size and ii.size:
            LON, LAT = np.meshgrid(lon[ii], lat[jj])
            inside = shapely.contains_xy(geom, LON, LAT)
            J, I = np.nonzero(inside)
            flat = jj[J] * nx + ii[I]
        if flat.size == 0:
            p = geom.representative_point()
            flat = np.array([np.abs(lat - p.y).argmin() * nx + np.abs(lon - p.x).argmin()])
            out[code] = (flat, 0)
        else:
            out[code] = (flat, int(flat.size))

    return out


# ============================================================================
# PERCENTILES Y PRECIPITACIÓN
# ============================================================================


def open_percentiles(cfg: Config, path: str | None = None, var: str | None = None) -> xr.DataArray:

    with xr.open_dataset(path or cfg.percentiles_file) as ds:
        q = ds[var or cfg.pct_var].sel(lat=slice(*cfg.lat_bounds), lon=slice(*cfg.lon_bounds)).load()

    return q.transpose("quantile", "lat", "lon")


def series_spec(cfg: Config) -> dict:
    """{nombre: (duración h, antecedente)}: R24, R72 inclusivos; A72 exclusivo."""

    spec = {f"R{d}": (d, False) for d in cfg.durations_h}
    spec.update({f"A{d}": (d, True) for d in cfg.antecedent_h})
    return spec


def window_sum(F: np.ndarray, n: int) -> np.ndarray:
    """
    F (nt, npix), pasos de 6 h contiguos -> R (nt, npix): suma de los n pasos que
    terminan en t. NaN si falta algún paso o si t < n-1 (inicio de temporada).
    """

    if n == 1:
        return F
    out = np.full_like(F, np.nan)
    nt = F.shape[0]
    if nt >= n:
        out[n - 1:] = F[n - 1:]
        for k in range(1, n):
            out[n - 1:] += F[n - 1 - k:nt - k]   # NaN se propaga a la ventana
    return out


def dur_step_stats(arr: np.ndarray, qpix: np.ndarray, imax6: np.ndarray, name: str) -> pd.DataFrame:
    """
    arr   (nt, npix): serie de duración (R24, R72, A72) en los píxeles del municipio.
    qpix  (nq, npix): percentiles de esa duración.
    imax6 (nt,):      índice local del píxel del máximo de 6 h en cada paso.
    """

    fin = np.isfinite(arr)
    valid = fin.any(axis=1)
    a = np.where(fin, arr, -np.inf)
    rows = np.arange(len(a))
    im = a.argmax(axis=1)

    with np.errstate(invalid="ignore"):
        nivel_pix = (arr[:, None, :] >= qpix[None, :, :]).sum(axis=1)   # (nt, npix); NaN -> 0

    at6 = arr[rows, imax6]
    return pd.DataFrame({
        f"{name}_max": np.where(valid, a[rows, im], np.nan),
        f"nivel_{name}_pixel_max": np.where(valid, nivel_pix[rows, im], -1),
        f"nivel_{name}_area": np.where(valid, nivel_pix.max(axis=1), -1),
        f"{name}_en_pico6": at6,
        f"nivel_{name}_en_pico6": np.where(np.isfinite(at6), nivel_pix[rows, imax6], -1),
    })


def local_path(path: str | Path) -> Path:
    """
    Ruta de Windows ("D:/...") a su montaje en WSL/Linux ("/mnt/d/..."); sin
    cambios en Windows o si la ruta ya es de Linux.
    """
    s = str(path).replace("\\", "/")
    if os.name != "nt" and re.match(r"^[A-Za-z]:/", s):
        return Path(f"/mnt/{s[0].lower()}") / s[3:]
    return Path(s)


def open_6h_year(cfg: Config, year: int) -> xr.DataArray | None:

    path = local_path(cfg.mswep_6h_dir) / f"mswep_v3_6h_{year}.nc"
    if not path.exists():
        return None

    return xr.open_dataset(path)["precipitation"]


def pct_labels(qs) -> list[str]:

    return [f"P{q * 100:g}".replace(".", "_") for q in qs]


def step_stats(arr: np.ndarray, qpix: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """
    arr  (nt, npix): P de 6 h en los píxeles del municipio.
    qpix (nq, npix): percentiles de esos píxeles.

    nivel_* = número de percentiles igualados o superados (0 = < primer percentil).
    """

    fin = np.isfinite(arr)
    valid = fin.any(axis=1)
    a = np.where(fin, arr, -np.inf)

    imax = a.argmax(axis=1)
    pmax = a[np.arange(len(a)), imax]
    pmean = np.where(fin, arr, 0.0).sum(axis=1) / np.maximum(fin.sum(axis=1), 1)

    with np.errstate(invalid="ignore"):
        ge = arr[:, None, :] >= qpix[None, :, :]          # (nt, nq, npix); NaN -> False
        nivel_pix = ge.sum(axis=1)                         # (nt, npix)
        qvalid = (np.isfinite(qpix)[None] & fin[:, None, :]).sum(axis=2)
        frac = ge.sum(axis=2) / np.where(qvalid > 0, qvalid, np.nan)

    out = {
        "P_max": np.where(valid, pmax, np.nan),
        "P_media": np.where(valid, pmean, np.nan),
        "nivel_pixel_max": np.where(valid, nivel_pix[np.arange(len(a)), imax], -1),
        "nivel_max_area": np.where(valid, nivel_pix.max(axis=1), -1),
        "ipix_max": imax,
    }
    for k, lab in enumerate(labels):
        out[f"{lab}_pixel_max"] = qpix[k, imax]
        # distancia (mm) entre la lluvia observada y el percentil local del mismo píxel
        out[f"dif_{lab}"] = np.where(valid, pmax - qpix[k, imax], np.nan)
        out[f"frac_area_ge_{lab}"] = np.where(valid, frac[:, k], np.nan)

    return pd.DataFrame(out)


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================


def run(cfg: Config = Config()) -> dict:

    t0 = time.time()
    os.makedirs(cfg.out_dir, exist_ok=True)

    print("\n" + "=" * 75)
    print("ITCHI-CORE · Declaratorias asociadas a CT vs MSWEP V3 6 h")
    print("=" * 75)

    # ------------------------------------------------------------------ 1
    print("\n[1/5] Declaratorias, municipios y trayectorias")

    decl = load_declaratorias(cfg)
    gdf = load_municipios(cfg)
    tracks = load_tracks(cfg)

    no_geom = decl.loc[~decl["CVEGEO"].isin(gdf["CVEGEO"]),
                       ["Identificador", "Estado", "Municipio", "Clave Municipio", "CVEGEO"]].drop_duplicates("CVEGEO")
    if os.path.exists(cfg.inafed_csv):
        inafed = pd.read_csv(cfg.inafed_csv, encoding="latin1", skiprows=3, dtype=str)
        no_geom["en_inafed"] = no_geom["CVEGEO"].isin(inafed["Cve_INEGI"].str.zfill(5))
    no_geom.to_csv(os.path.join(cfg.out_dir, "qc_claves_sin_geometria.csv"), index=False)
    print(f"Claves sin polígono en {Path(cfg.mun_shp).name}: {len(no_geom)} (ver qc_claves_sin_geometria.csv)")

    decl = decl[decl["CVEGEO"].isin(gdf["CVEGEO"])]

    # ------------------------------------------------------------------ 2
    print("\n[2/5] Expansión a 6 h y asociación con posiciones de CT")

    steps = expand_6h(decl, cfg.dias_antes, cfg.dias_despues)
    pasos = associate_tracks(steps, tracks, municipio_points(gdf), cfg)
    print(f"Pasos declaratoria x 6 h: {len(steps):,} | con CT activo: {len(pasos):,} "
          f"| declaratorias con >= 1 paso: {pasos['id_decl'].nunique():,}")

    # ------------------------------------------------------------------ 3
    print("\n[3/5] Píxeles por municipio y percentiles")

    Q = open_percentiles(cfg)
    lat, lon = Q["lat"].values, Q["lon"].values
    qs = Q["quantile"].values
    labels = pct_labels(qs)
    Qflat = Q.values.reshape(len(qs), -1)

    pix = municipio_pixels(gdf, lat, lon, pasos["CVEGEO"].unique())
    print(f"Percentiles ({cfg.pct_var}): {', '.join(labels)} | municipios: {len(pix):,} "
          f"| sin centro de píxel dentro: {sum(n == 0 for _, n in pix.values())}")

    # Percentiles por duración (misma malla que los de 6 h)
    spec = series_spec(cfg)
    Qdur, labels_dur = {}, {}
    for d in sorted({d for d, _ in spec.values()}):
        Qd = open_percentiles(cfg, cfg.percentiles_dur_pattern.format(d=d), cfg.pct_var_dur)
        if not (np.allclose(Qd["lat"].values, lat) and np.allclose(Qd["lon"].values, lon)):
            raise ValueError(f"La malla de los percentiles de {d} h no coincide con la de 6 h.")
        Qdur[d] = Qd.values.reshape(Qd.sizes["quantile"], -1)
        labels_dur[d] = pct_labels(Qd["quantile"].values)
        print(f"Percentiles {d} h ({cfg.pct_var_dur}): {', '.join(labels_dur[d])}")

    # ------------------------------------------------------------------ 4
    print("\n[4/5] Precipitación de 6 h por paso")

    ny, nx = lat.size, lon.size
    pix_min = np.full(ny * nx, np.inf, dtype="float64")
    pix_n = np.zeros(ny * nx, dtype="int32")

    # 1 dic 00 UTC está en el archivo del mismo año (última ventana de la temporada)
    pasos["year"] = pasos["time"].dt.year
    results = []
    rec = []   # (declaratoria, píxel): máximo de 6 h en el píxel durante la declaratoria

    for year, py in tqdm(pasos.groupby("year"), desc="Años", unit="año"):

        da = open_6h_year(cfg, int(year))
        if da is None:
            continue

        tfile = pd.DatetimeIndex(da["time"].values)
        need = pd.DatetimeIndex(py["time"].unique()).intersection(tfile).sort_values()
        if need.empty:
            da.close()
            continue

        # Temporada completa: los acumulados de 24/72 h necesitan los pasos previos.
        # El archivo está fragmentado (tiempo completo, 25 x 25), así que leer toda la
        # temporada del recorte cuesta lo mismo que leer solo los pasos con CT.
        if not (np.diff(tfile.values) == np.timedelta64(6, "h")).all():
            raise ValueError(f"{year}: los pasos de 6 h no son contiguos; no se pueden formar acumulados.")
        Fy = (da.sel(lat=slice(*cfg.lat_bounds), lon=slice(*cfg.lon_bounds))
                .transpose("time", "lat", "lon").load().values.reshape(len(tfile), -1))
        da.close()

        if Fy.shape[1] != ny * nx:
            raise ValueError("La malla de los acumulados de 6 h no coincide con la de los percentiles.")

        # Series en los pasos requeridos: P6 y, por duración, R_d(T) o A_d(T) = R_d(T - 6 h)
        ineed = tfile.get_indexer(need)
        G = {"P6": Fy[ineed]}
        for d in sorted({d for d, _ in spec.values()}):
            Rd = window_sum(Fy, d // 6)
            for name, (dd, ant) in spec.items():
                if dd != d:
                    continue
                if ant:
                    prev = ineed - 1
                    G[name] = np.where((prev >= 0)[:, None], Rd[np.maximum(prev, 0)], np.nan).astype("float32")
                else:
                    G[name] = Rd[ineed]
            del Rd
        del Fy
        F = G["P6"]

        tpos = pd.Series(np.arange(len(need)), index=need)

        for code, pm in py[py["time"].isin(need)].groupby("CVEGEO"):

            flat, _ = pix[code]
            ut = pd.DatetimeIndex(pm["time"].unique())
            arr = F[tpos[ut].values][:, flat]
            arrs = {name: G[name][tpos[ut].values][:, flat] for name in spec}

            st = step_stats(arr, Qflat[:, flat], labels)
            imax6 = st["ipix_max"].values
            for name, (d, _) in spec.items():
                st = st.join(dur_step_stats(arrs[name], Qdur[d][:, flat], imax6, name))
            st["time"] = ut
            st["CVEGEO"] = code
            st["lat_pmax"] = lat[flat[imax6] // nx]
            st["lon_pmax"] = lon[flat[imax6] % nx]
            results.append(st.drop(columns="ipix_max"))

            # Producto por píxel: máximo de 6 h por declaratoria, luego mínimo entre declaratorias
            row_of = {t: k for k, t in enumerate(ut)}
            for idd, pe in pm.groupby("id_decl"):
                tt = pd.DatetimeIndex(pe["time"])
                ev = arr[[row_of[t] for t in tt]]
                if not np.isfinite(ev).any():
                    continue
                evf = np.where(np.isfinite(ev), ev, -np.inf)
                kmax = evf.argmax(axis=0)                              # paso del máximo en cada píxel
                ev_max = evf[kmax, np.arange(ev.shape[1])]
                ok = np.isfinite(ev_max)
                pix_min[flat[ok]] = np.minimum(pix_min[flat[ok]], ev_max[ok])
                pix_n[flat[ok]] += 1
                r = {"id_decl": idd, "ipix": flat[ok], "P6_pix": ev_max[ok], "t_max_pix": tt[kmax[ok]]}
                irow = [row_of[t] for t in tt]
                for name, (_, ant) in spec.items():
                    e = arrs[name][irow]
                    if ant:   # antecedente en el paso del máximo de 6 h del píxel
                        r[f"{name}_pix_en_tmax"] = e[kmax, np.arange(e.shape[1])][ok]
                    else:     # máximo propio del acumulado durante la declaratoria
                        with np.errstate(invalid="ignore"):
                            r[f"{name}_pix_max"] = np.fmax.reduce(e, axis=0)[ok]
                rec.append(pd.DataFrame(r))

    stats = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["CVEGEO", "time"])
    pasos = pasos.drop(columns="year").merge(stats, on=["CVEGEO", "time"], how="left")
    pasos["con_datos"] = pasos["P_max"].notna()

    # ------------------------------------------------------------------ 5
    print("\n[5/5] Eventos, municipios y producto por píxel")

    info = decl.set_index("id_decl")[["Identificador", "Estado", "Municipio", "CVEGEO", "Tipo Declaratoria",
                                      "Tipo Fenómeno", "Observaciones", "menciona_lluvia", "nombre_ct", "fecha_inicio",
                                      "fecha_fin", "fechas_invertidas"]]

    v = pasos[pasos["con_datos"]]
    peak = v.loc[v.groupby("id_decl")["P_max"].idxmax()].set_index("id_decl")
    agg = v.groupby("id_decl").agg(
        n_pasos_ct=("time", "size"),
        P_evento_media_max=("P_media", "max"),
        nivel_evento_max=("nivel_max_area", "max"),
    )
    n_all = pasos.groupby("id_decl")["time"].size().rename("n_pasos_ct_total")

    pk_cols = ["time", "CT", "dist_km", "dentro_rocloud", "dentro_Rp", "P_max", "P_media",
               "nivel_pixel_max", "nivel_max_area", "lat_pmax", "lon_pmax"] + \
              [f"{l}_pixel_max" for l in labels] + [f"dif_{l}" for l in labels] + \
              [f"frac_area_ge_{l}" for l in labels] + \
              [c for s in spec for c in (f"{s}_en_pico6", f"nivel_{s}_en_pico6")]
    peak = peak[pk_cols].rename(columns={"time": "t_pico", "CT": "CT_pico", "dist_km": "dist_pico_km",
                                         "dentro_rocloud": "dentro_rocloud_pico", "dentro_Rp": "dentro_Rp_pico",
                                         "P_max": "P_evento", "P_media": "P_media_pico",
                                         "nivel_pixel_max": "nivel_pico", "nivel_max_area": "nivel_area_pico"})

    eventos = info.join(n_all, how="inner").join(agg).join(peak)
    eventos["P_evento_sobre_P95"] = eventos["P_evento"] / eventos.get("P95_pixel_max", np.nan)
    eventos["clase_pico"] = level_label(eventos["nivel_pico"], labels)
    eventos = eventos.join(event_duration_stats(v, spec))
    for s, (d, _) in spec.items():
        eventos[f"clase_{s}_en_pico6"] = level_label(eventos[f"nivel_{s}_en_pico6"], labels_dur[d])
    eventos = combined_levels(eventos, spec, labels, labels_dur, cfg)
    eventos = eventos.reset_index()

    # Lluvia por píxel y declaratoria (sin umbrales)
    pix_decl = build_pixel_decl(rec, eventos, pasos, lat, lon, nx)
    areal = pix_decl.groupby("id_decl")["P6_pix"].mean().rename("P_evento_areal")
    eventos = eventos.merge(areal, left_on="id_decl", right_index=True, how="left")

    municipios = summarize_municipios(eventos, gdf, pix, Qflat, labels)
    ds_stats = pixel_stats(pix_decl, Q, cfg)
    mun_stats = municipio_stats(eventos, gdf)
    ciclones = ciclon_stats(eventos)

    pix_min = np.where(pix_n > 0, pix_min, np.nan).reshape(ny, nx)
    ds_pix = pixel_product(pix_min, pix_n.reshape(ny, nx), Q, labels, cfg)

    # ------------------------------------------------------------------ escritura
    f_pasos = os.path.join(cfg.out_dir, "declaratorias_ct_pasos.csv")
    f_ev = os.path.join(cfg.out_dir, "declaratorias_ct_eventos.csv")
    f_mun = os.path.join(cfg.out_dir, "declaratorias_ct_municipios.csv")
    f_pix = os.path.join(cfg.out_dir, "declaratorias_ct_pixel_min.nc")

    pasos.to_csv(f_pasos, index=False, float_format="%.3f")
    eventos.to_csv(f_ev, index=False, float_format="%.3f")
    municipios.drop(columns="geometry").to_csv(f_mun, index=False, float_format="%.3f")
    try:
        municipios.to_file(f_mun.replace(".csv", ".gpkg"), driver="GPKG")
    except Exception as e:  # sin driver GPKG: queda el CSV
        print(f"  Aviso: no se escribió el GPKG ({type(e).__name__}: {e})")
    ds_pix.to_netcdf(f_pix, encoding={v: {"zlib": True, "complevel": 4} for v in ds_pix.data_vars})

    pix_decl.to_csv(os.path.join(cfg.out_dir, "declaratorias_ct_pixel_decl.csv.gz"), index=False, float_format="%.3f")
    ds_stats.to_netcdf(os.path.join(cfg.out_dir, "declaratorias_ct_pixel_stats.nc"),
                       encoding={v: {"zlib": True, "complevel": 4} for v in ds_stats.data_vars})
    mun_stats.to_csv(os.path.join(cfg.out_dir, "declaratorias_ct_municipios_stats.csv"), index=False,
                     float_format="%.3f")
    ciclones.to_csv(os.path.join(cfg.out_dir, "declaratorias_ct_por_ciclon.csv"), index=False, float_format="%.3f")

    res_md, cruz = resumen_multiduracion(eventos, labels)
    res_md.to_csv(os.path.join(cfg.out_dir, "declaratorias_ct_multiduracion_resumen.csv"), float_format="%.2f")
    if cruz is not None:
        cruz.to_csv(os.path.join(cfg.out_dir, "declaratorias_ct_matriz_P6_A72.csv"), float_format="%.4f")

    ok = eventos["P_evento"].notna()
    print(f"\nDeclaratorias con datos: {ok.sum():,} | municipios: {municipios['n_decl'].gt(0).sum():,}")
    print(f"P_evento (mm/6 h): mín {eventos.loc[ok, 'P_evento'].min():.1f} | "
          f"P10 {eventos.loc[ok, 'P_evento'].quantile(0.1):.1f} | "
          f"mediana {eventos.loc[ok, 'P_evento'].median():.1f}")
    print("Distancia al percentil local en el píxel del pico (P_evento - Pxx, mm/6 h):")
    e = eventos.loc[ok]
    print(pd.DataFrame({
        "mediana": [e[f"dif_{l}"].median() for l in labels],
        "P10": [e[f"dif_{l}"].quantile(0.1) for l in labels],
        "% bajo el percentil": [100 * (e[f"dif_{l}"] < 0).mean() for l in labels],
    }, index=labels).astype(float).round(1).to_string())
    print("\nMultiduración (%, ver declaratorias_ct_multiduracion_resumen.csv):")
    print(res_md.round(1).to_string())

    print("\n" + "=" * 75)
    print(f"PROCESO TERMINADO · {(time.time() - t0) / 60:.1f} minutos · {cfg.out_dir}")
    print("=" * 75)

    return {"pasos": pasos, "eventos": eventos, "municipios": municipios, "pixel": ds_pix,
            "pixel_decl": pix_decl, "pixel_stats": ds_stats, "municipios_stats": mun_stats, "ciclones": ciclones}


# ============================================================================
# RESÚMENES
# ============================================================================


TIPOS = ("Emergencia", "Desastre", "Contingencia Climatológica")
ESTADISTICOS = {"min": "min", "mediana": "median", "media": "mean", "max": "max"}


def build_pixel_decl(rec, eventos, pasos, lat, lon, nx) -> pd.DataFrame:
    """
    Tabla larga (declaratoria x píxel del municipio): máximo de 6 h en el píxel durante
    la declaratoria (ventanas con CT), hora de ese máximo y CT activo a esa hora.
    """

    cols = ["id_decl", "ipix", "P6_pix", "t_max_pix"]
    d = pd.concat(rec, ignore_index=True) if rec else pd.DataFrame(columns=cols)
    d["lat"] = lat[d["ipix"].astype(int) // nx]
    d["lon"] = lon[d["ipix"].astype(int) % nx]

    info = eventos[["id_decl", "Identificador", "Tipo Declaratoria", "CVEGEO", "Municipio", "Estado"]]
    d = d.merge(info, on="id_decl", how="left")

    ct = pasos[["id_decl", "time", "CT", "dist_km"]].rename(columns={"time": "t_max_pix", "CT": "CT_t_max",
                                                                     "dist_km": "dist_t_max_km"})
    d = d.merge(ct, on=["id_decl", "t_max_pix"], how="left")

    dur = [c for c in d.columns if c.endswith(("_pix_max", "_pix_en_tmax"))]
    return d[["id_decl", "Identificador", "Tipo Declaratoria", "CVEGEO", "Municipio", "Estado", "ipix",
              "lat", "lon", "P6_pix", "t_max_pix", "CT_t_max", "dist_t_max_km"] + dur]


# ============================================================================
# MULTIDURACIÓN
# ============================================================================


GRUPOS_A = ["<P75", "P75-P95", ">=P95"]


def event_duration_stats(v: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """
    Por declaratoria y serie (R24, R72, A72): máximo propio sobre píxeles y pasos con CT,
    su clase en su propio píxel y la clase máxima sobre el área del municipio.
    """

    parts = []
    for s in spec:
        col = f"{s}_max"
        w = v[v[col].notna()]
        at = w.loc[w.groupby("id_decl")[col].idxmax()].set_index("id_decl")
        parts.append(pd.DataFrame({
            f"{s}_evento": at[col],
            f"nivel_{s}_evento": at[f"nivel_{s}_pixel_max"],
            f"nivel_{s}_area_evento": v.groupby("id_decl")[f"nivel_{s}_area"].max(),
        }))

    return pd.concat(parts, axis=1)


def combined_levels(ev: pd.DataFrame, spec: dict, labels: list[str], labels_dur: dict,
                    cfg: Config) -> pd.DataFrame:
    """
    En el pico de 6 h:
      opción A  nivel_multidur_pico = máx(clase P6, clase R24, clase R72) y la duración que lo fija
      rampa     nivel_base_pico     = Config.rampa_base[clase P6]
      opción B  nivel_matriz_pico   = rampa base con la escalada de Config.escalada_*
    Las clases solo son comparables entre duraciones si usan los mismos percentiles.
    """

    ev = ev.copy()
    r_names = [s for s, (_, ant) in spec.items() if not ant]
    if any(labels_dur[spec[s][0]] != labels for s in r_names):
        print("  Aviso: los percentiles de 6 h y de 24/72 h difieren; nivel_multidur_pico compara clases distintas.")

    # Opción A: máximo de clases; en empate gana la duración más corta (orden P6, R24, R72)
    cols = ["nivel_pico"] + [f"nivel_{s}_en_pico6" for s in r_names]
    cl = ev[cols].astype(float).where(lambda x: x >= 0)
    ev["nivel_multidur_pico"] = cl.max(axis=1)
    dom = cl.fillna(-1).to_numpy().argmax(axis=1)
    ev["duracion_dominante_pico"] = np.where(cl.notna().any(axis=1), np.array(["P6"] + r_names)[dom], None)

    # Rampa solo 6 h
    base = np.asarray(cfg.rampa_base)
    if labels[0] != "P75" or base.size != len(labels) + 1:
        print(f"  Aviso: rampa_base {base.tolist()} no es compatible con los percentiles {labels}; "
              "no se calculan nivel_base_pico ni nivel_matriz_pico.")
        return ev
    c6 = ev["nivel_pico"].astype(float)
    ok6 = (c6 >= 0).to_numpy()
    row = np.where(ok6, c6, 0).astype(int)
    lv_base = base[row]
    ev["nivel_base_pico"] = np.where(ok6, lv_base, np.nan)

    # Grupo de A72 (para la tabla cruzada)
    a_names = [s for s, (_, ant) in spec.items() if ant]
    if a_names:
        a = a_names[0]
        la = labels_dur[spec[a][0]]
        ca = ev[f"nivel_{a}_en_pico6"].astype(float)
        col = ((ca >= la.index("P75") + 1).astype(int) + (ca >= la.index("P95") + 1).astype(int)).to_numpy()
        ev[f"grupo_{a}_pico"] = np.where((ca >= 0).to_numpy(), np.array(GRUPOS_A, dtype=object)[col], None)

    # Opción B: escalada por condiciones
    conds = []
    for var, pct in cfg.escalada_condiciones:
        if var not in spec or f"nivel_{var}_en_pico6" not in ev:
            print(f"  Aviso: falta {var} para la escalada; no se calcula nivel_matriz_pico.")
            return ev
        k = labels_dur[spec[var][0]].index(pct) + 1
        conds.append((ev[f"nivel_{var}_en_pico6"].astype(float) >= k).to_numpy())
    cond = np.logical_and.reduce(conds) if cfg.escalada_modo == "y" else np.logical_or.reduce(conds)
    esc = ok6 & np.isin(row, cfg.escalada_clases_P6) & cond
    ev["nivel_matriz_pico"] = np.where(ok6, np.where(esc, np.maximum(lv_base, cfg.escalada_nivel), lv_base), np.nan)

    return ev


def resumen_multiduracion(ev: pd.DataFrame, labels: list[str]):
    """
    Diagnósticos con positivos (sin negativos no hay falsas alarmas):
      - fracción < P75 en 6 h y cuántas suben con R24/R72 (opción A)
      - fracción escalada por A72 (opción B) y saturación del nivel máximo
    Por tipo, con todas las declaratorias y solo con P_evento >= 1 mm
    (las emergencias preventivas sin lluvia no pertenecen a la calibración del peligro).
    """

    has_b = "nivel_matriz_pico" in ev
    a_col = next((c for c in ev.columns if c.startswith("grupo_A") and c.endswith("_pico")), None)
    rows, cruz = {}, []

    for t in ["todas"] + list(TIPOS):
        et = ev if t == "todas" else ev[ev["Tipo Declaratoria"].eq(t)]
        et = et[et["nivel_pico"] >= 0]
        for filtro, e in (("todas", et), ("P_evento >= 1 mm", et[et["P_evento"] >= 1.0])):
            if e.empty:
                continue
            c6, ma = e["nivel_pico"], e["nivel_multidur_pico"]
            r = {"n": len(e),
                 "% < P75 (6 h)": 100 * (c6 == 0).mean(),
                 "% de < P75 que sube con R24/R72": 100 * (ma[c6 == 0] > 0).mean() if (c6 == 0).any() else np.nan,
                 "% escalada A (R24/R72)": 100 * (ma > c6).mean(),
                 "% fijada por R24": 100 * e["duracion_dominante_pico"].eq("R24").mean(),
                 "% fijada por R72": 100 * e["duracion_dominante_pico"].eq("R72").mean()}
            if has_b:
                b = e.dropna(subset=["nivel_matriz_pico"])
                r.update({"% escalada B (A72)": 100 * (b["nivel_matriz_pico"] > b["nivel_base_pico"]).mean(),
                          "% máximo rampa 6 h": 100 * (b["nivel_base_pico"] == 4).mean(),
                          "% máximo matriz B": 100 * (b["nivel_matriz_pico"] == 4).mean()})
                r.update({f"% matriz nivel {k}": 100 * (b["nivel_matriz_pico"] == k).mean() for k in range(5)})
            rows[(t, filtro)] = r

        if has_b and a_col and not et.empty:
            ct = pd.crosstab(level_label(et["nivel_pico"], labels), et[a_col], normalize="all") \
                   .reindex(index=level_names(labels), columns=GRUPOS_A, fill_value=0.0)
            ct.index.name, ct.columns.name = "clase_P6", a_col
            cruz.append(ct.assign(tipo=t).set_index("tipo", append=True).swaplevel())

    res = pd.DataFrame.from_dict(rows, orient="index")
    res.index.names = ["tipo", "filtro"]
    return res, (pd.concat(cruz) if cruz else None)


def pixel_stats(pix_decl: pd.DataFrame, Q: xr.DataArray, cfg) -> xr.Dataset:
    """
    Por píxel y tipo de declaratoria: número de declaratorias y mínimo, mediana, media y
    máximo (entre declaratorias) del máximo de 6 h en el píxel, sin umbrales. Se guarda
    el Identificador de la declaratoria con el mínimo y con el máximo.
    """

    ny, nx = Q.sizes["lat"], Q.sizes["lon"]
    tipos = ["todas"] + [t for t in TIPOS if (pix_decl["Tipo Declaratoria"] == t).any()]

    out = {k: np.full((len(tipos), ny * nx), np.nan, dtype="float32") for k in ESTADISTICOS}
    n = np.zeros((len(tipos), ny * nx), dtype="int32")
    id_min = np.full((len(tipos), ny * nx), -1, dtype="int32")
    id_max = np.full((len(tipos), ny * nx), -1, dtype="int32")

    for k, t in enumerate(tipos):
        d = pix_decl if t == "todas" else pix_decl[pix_decl["Tipo Declaratoria"] == t]
        if d.empty:
            continue
        g = d.groupby("ipix")["P6_pix"]
        idx = g.size().index.values
        n[k, idx] = g.size().values
        for name, f in ESTADISTICOS.items():
            out[name][k, idx] = g.agg(f).values
        id_min[k, idx] = d.loc[g.idxmin().values, "Identificador"].values
        id_max[k, idx] = d.loc[g.idxmax().values, "Identificador"].values

    dims = ("tipo", "lat", "lon")
    coords = {"tipo": tipos, "lat": Q["lat"].values, "lon": Q["lon"].values}
    ds = xr.Dataset(coords=coords)
    ds["n_decl"] = (dims, n.reshape(len(tipos), ny, nx))
    ds["n_decl"].attrs.update(units="1", long_name="declarations covering the pixel")
    for name in ESTADISTICOS:
        ds[f"P_{name}"] = (dims, out[name].reshape(len(tipos), ny, nx))
        ds[f"P_{name}"].attrs.update(units="mm (6 h)-1", long_name=f"{name} over declarations of the "
                                                                   "per-declaration maximum 6-h precipitation")
    ds["Identificador_min"] = (dims, id_min.reshape(len(tipos), ny, nx))
    ds["Identificador_max"] = (dims, id_max.reshape(len(tipos), ny, nx))
    for v in ("Identificador_min", "Identificador_max"):
        ds[v].attrs.update(_FillValue=-1, long_name=f"CENAPRED Identificador of the declaration giving P_{v[-3:]}")

    ds.attrs.update(framework="ITCHI-CORE",
                    description="Per-pixel 6-h precipitation during TC-associated declarations (no thresholds)",
                    history=f"scripts/declaratorias_6h.py {pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%MZ}")
    return ds


def _stats_con_id(e: pd.DataFrame, var: str) -> dict:

    x = e[var]
    if x.notna().sum() == 0:
        return {}
    i0, i1 = x.idxmin(), x.idxmax()
    r = {f"{var}_{k}": x.agg(f) for k, f in ESTADISTICOS.items()}
    r.update({f"{var}_Identificador_min": e.at[i0, "Identificador"], f"{var}_CT_min": e.at[i0, "CT_pico"],
              f"{var}_t_min": e.at[i0, "t_pico"],
              f"{var}_Identificador_max": e.at[i1, "Identificador"], f"{var}_CT_max": e.at[i1, "CT_pico"],
              f"{var}_t_max": e.at[i1, "t_pico"]})
    return r


def municipio_stats(eventos: pd.DataFrame, gdf) -> pd.DataFrame:
    """
    Por municipio y tipo: mínimo, mediana, media y máximo entre declaratorias de
    P_evento (máximo en el municipio) y P_evento_areal (media de los píxeles), con la
    declaratoria y el CT del mínimo y del máximo, y la lista de declaratorias.
    """

    ev = eventos[eventos["P_evento"].notna()]
    rows = []
    for t in ["todas"] + list(TIPOS):
        d = ev if t == "todas" else ev[ev["Tipo Declaratoria"] == t]
        for code, e in d.groupby("CVEGEO"):
            r = {"tipo": t, "CVEGEO": code, "n_decl": len(e),
                 "identificadores": ";".join(map(str, sorted(e["Identificador"].unique()))),
                 "ciclones": ";".join(sorted(e["CT_pico"].dropna().unique()))}
            r.update(_stats_con_id(e, "P_evento"))
            r.update(_stats_con_id(e, "P_evento_areal"))
            rows.append(r)

    out = pd.DataFrame(rows)
    names = pd.DataFrame(gdf[["CVEGEO", "NOMGEO", "NOM_ENT"]])
    return names.merge(out, on="CVEGEO", how="right")


def ciclon_stats(eventos: pd.DataFrame) -> pd.DataFrame:
    """Por CT (el más cercano en el pico de cada declaratoria): declaratorias asociadas y su lluvia."""

    ev = eventos[eventos["P_evento"].notna()]
    rows = []
    for ct, e in ev.groupby("CT_pico"):
        r = {"CT": ct, "n_decl": len(e), "n_municipios": e["CVEGEO"].nunique(),
             **{f"n_{t.split()[0].lower()}": int((e["Tipo Declaratoria"] == t).sum()) for t in TIPOS},
             "estados": ";".join(sorted(e["Estado"].unique())),
             "identificadores": ";".join(map(str, sorted(e["Identificador"].unique()))),
             "observaciones": " | ".join(e["Observaciones"].drop_duplicates().head(5)),
             "fecha_inicio_min": e["fecha_inicio"].min(), "fecha_fin_max": e["fecha_fin"].max()}
        r.update({f"P_evento_{k}": e["P_evento"].agg(f) for k, f in ESTADISTICOS.items()})
        rows.append(r)
    return pd.DataFrame(rows).sort_values("fecha_inicio_min")


def level_names(labels: list[str]) -> list[str]:

    return [f"<{labels[0]}"] + [f"{a}-{b}" for a, b in zip(labels[:-1], labels[1:])] + [f">={labels[-1]}"]


def level_label(level: pd.Series, labels: list[str]) -> pd.Series:

    names = np.array(level_names(labels), dtype=object)
    lv = level.to_numpy(dtype="float64")
    out = np.full(lv.shape, None, dtype=object)
    ok = np.isfinite(lv) & (lv >= 0)
    out[ok] = names[lv[ok].astype(int)]
    return pd.Series(out, index=level.index)


def summarize_municipios(eventos: pd.DataFrame, gdf, pix: dict, Qflat: np.ndarray, labels: list[str]):
    """Mínimo de P_evento por municipio y percentiles de referencia del municipio."""

    ev = eventos[eventos["P_evento"].notna()]
    imin = ev.groupby("CVEGEO")["P_evento"].idxmin()
    at_min = ev.loc[imin].set_index("CVEGEO")

    g = ev.groupby("CVEGEO")["P_evento"]
    s = pd.DataFrame({
        "n_decl": g.size(),
        "n_emergencia": ev[ev["Tipo Declaratoria"].eq("Emergencia")].groupby("CVEGEO").size(),
        "n_desastre": ev[ev["Tipo Declaratoria"].eq("Desastre")].groupby("CVEGEO").size(),
        "P_min": g.min(),
        "P_p10": g.quantile(0.10),
        "P_mediana": g.median(),
        "P_max": g.max(),
        "Identificador_min": at_min["Identificador"],
        "tipo_min": at_min["Tipo Declaratoria"],
        "obs_min": at_min["Observaciones"],
        "lluvia_min": at_min["menciona_lluvia"],
        "t_pico_min": at_min["t_pico"],
        "CT_min": at_min["CT_pico"],
        "dist_min_km": at_min["dist_pico_km"],
        "clase_min": at_min["clase_pico"],
        "P_min_sobre_P95": at_min["P_evento_sobre_P95"],
    })
    # Percentiles del píxel donde ocurrió el mínimo y distancia (mm) a cada uno
    for lab in labels:
        s[f"{lab}_pixel_min"] = at_min[f"{lab}_pixel_max"]
        s[f"dif_min_{lab}"] = at_min[f"dif_{lab}"]
    for c in ("n_emergencia", "n_desastre"):
        s[c] = s[c].fillna(0).astype(int)

    # Percentiles de referencia: media y máximo sobre los píxeles del municipio
    ref = {}
    for code in s.index:
        flat, n = pix[code]
        q = Qflat[:, flat]
        row = {"n_pix": n}
        for k, lab in enumerate(labels):
            row[f"{lab}_mun_media"] = np.nanmean(q[k]) if np.isfinite(q[k]).any() else np.nan
            row[f"{lab}_mun_max"] = np.nanmax(q[k]) if np.isfinite(q[k]).any() else np.nan
        ref[code] = row
    s = s.join(pd.DataFrame.from_dict(ref, orient="index"))

    out = gdf.merge(s, left_on="CVEGEO", right_index=True, how="left")
    out["n_decl"] = out["n_decl"].fillna(0).astype(int)

    return out


def pixel_product(pix_min, pix_n, Q: xr.DataArray, labels: list[str], cfg: Config) -> xr.Dataset:

    coords = {"lat": Q["lat"], "lon": Q["lon"]}
    pmin = xr.DataArray(pix_min.astype("float32"), dims=("lat", "lon"), coords=coords)

    with np.errstate(invalid="ignore"):
        nivel = (pmin >= Q).sum("quantile").where(pmin.notnull())

    ds = xr.Dataset({
        "P_min_decl": pmin.assign_attrs(
            units="mm (6 h)-1",
            long_name="minimum over TC-associated declarations of the per-declaration maximum 6-h precipitation"),
        "n_decl": xr.DataArray(pix_n, dims=("lat", "lon"), coords=coords).assign_attrs(
            units="1", long_name="number of declarations covering the pixel"),
        "nivel_min": nivel.astype("float32").assign_attrs(
            units="1", long_name="number of local percentiles reached by P_min_decl",
            flag_meanings=" ".join(level_names(labels))),
    })
    for k, lab in enumerate(labels):
        qk = Q.isel(quantile=k).drop_vars("quantile")
        ds[f"P_min_menos_{lab}"] = (pmin - qk).astype("float32").assign_attrs(
            units="mm (6 h)-1", long_name=f"P_min_decl minus local {lab}")
        ds[f"P_min_sobre_{lab}"] = (pmin / qk.where(qk > 0)).astype("float32").assign_attrs(units="1")

    ds.attrs.update(
        framework="ITCHI-CORE",
        source="MSWEP V3 6-h accumulations (T-6h, T]; CENAPRED declarations; ROCLOUD tracks",
        percentiles=f"{Path(cfg.percentiles_file).name}:{cfg.pct_var}",
        association=(f"time match with nearest active TC; dias_antes={cfg.dias_antes}; "
                     f"dias_despues={cfg.dias_despues}; max_dist_km={cfg.max_dist_km}; "
                     f"inside_rocloud={cfg.require_inside_rocloud}; solo_lluvia={cfg.solo_lluvia}"),
        history=f"scripts/declaratorias_6h.py {pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%MZ}",
    )

    return ds


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    out = run(Config())
