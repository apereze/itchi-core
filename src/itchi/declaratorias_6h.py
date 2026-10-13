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

Salidas (outputs/diagnostics/declaratorias_6h/):

    declaratorias_ct_pasos.csv
    declaratorias_ct_eventos.csv
    declaratorias_ct_municipios.csv  (+ .gpkg si hay driver)
    declaratorias_ct_pixel_min.nc
    declaratorias_ct_pixel_decl.csv.gz    declaratoria x píxel: máximo de 6 h, hora y CT
    declaratorias_ct_pixel_stats.nc       por píxel y tipo: n, mín, mediana, media, máx + Identificador
    declaratorias_ct_municipios_stats.csv por municipio y tipo: mín, mediana, media, máx + declaratoria/CT
    declaratorias_ct_por_ciclon.csv       por CT: declaratorias asociadas y su lluvia
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

REPO = Path("/mnt/c/Users/adolf/Documents/ITCHI/itchi-core")
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


def open_percentiles(cfg: Config) -> xr.DataArray:

    with xr.open_dataset(cfg.percentiles_file) as ds:
        q = ds[cfg.pct_var].sel(lat=slice(*cfg.lat_bounds), lon=slice(*cfg.lon_bounds)).load()

    return q.transpose("quantile", "lat", "lon")


def open_6h_year(cfg: Config, year: int) -> xr.DataArray | None:

    path = os.path.join(cfg.mswep_6h_dir, f"mswep_v3_6h_{year}.nc")
    if not os.path.exists(path):
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

        F = (da.sel(time=need, lat=slice(*cfg.lat_bounds), lon=slice(*cfg.lon_bounds))
               .transpose("time", "lat", "lon").load().values.reshape(len(need), -1))
        da.close()

        if F.shape[1] != ny * nx:
            raise ValueError("La malla de los acumulados de 6 h no coincide con la de los percentiles.")

        tpos = pd.Series(np.arange(len(need)), index=need)

        for code, pm in py[py["time"].isin(need)].groupby("CVEGEO"):

            flat, _ = pix[code]
            ut = pd.DatetimeIndex(pm["time"].unique())
            arr = F[tpos[ut].values][:, flat]

            st = step_stats(arr, Qflat[:, flat], labels)
            st["time"] = ut
            st["CVEGEO"] = code
            st["lat_pmax"] = lat[flat[st["ipix_max"].values] // nx]
            st["lon_pmax"] = lon[flat[st["ipix_max"].values] % nx]
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
                rec.append(pd.DataFrame({"id_decl": idd, "ipix": flat[ok], "P6_pix": ev_max[ok],
                                         "t_max_pix": tt[kmax[ok]]}))

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
              [f"frac_area_ge_{l}" for l in labels]
    peak = peak[pk_cols].rename(columns={"time": "t_pico", "CT": "CT_pico", "dist_km": "dist_pico_km",
                                         "dentro_rocloud": "dentro_rocloud_pico", "dentro_Rp": "dentro_Rp_pico",
                                         "P_max": "P_evento", "P_media": "P_media_pico",
                                         "nivel_pixel_max": "nivel_pico", "nivel_max_area": "nivel_area_pico"})

    eventos = info.join(n_all, how="inner").join(agg).join(peak)
    eventos["P_evento_sobre_P95"] = eventos["P_evento"] / eventos.get("P95_pixel_max", np.nan)
    eventos["clase_pico"] = level_label(eventos["nivel_pico"], labels)
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

    return d[["id_decl", "Identificador", "Tipo Declaratoria", "CVEGEO", "Municipio", "Estado", "ipix",
              "lat", "lon", "P6_pix", "t_max_pix", "CT_t_max", "dist_t_max_km"]]


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
