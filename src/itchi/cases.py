"""
ITCHI on case studies, on the grid and by municipality.

For a tropical cyclone of the catalogue (CASES) it

1. reads the IBTrACS track with observed R34, the W06 radii (PA21) and ROCLOUD
   (load_track);
2. keeps the positions whose attribution region (ROCLOUD with R_W06 floor,
   largest quadrant) reaches a land cell of Mexico (select_positions);
3. reads P6, R24 and A72 of MSWEP V3 and the 6/24/72-h percentiles;
4. computes ITCHI with pipeline.compute_itchi_snapshot at every position for
   each experiment (by default only the configuration of configs/default.yaml)
   and the continuous H_P at every 6-h time (compute_case_fields);
5. aggregates by municipality (itchi.municipal) and summarises: classes,
   ordinal check against the declaratorias, lead time before an event.

Results are cached as NetCDF in CaseConfig.cache_dir (load_or_compute).
"""

from __future__ import annotations

import os
import re
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from itchi.climatology import load_percentile_climatology
from itchi.config import get_project_root, itchi_settings_from_config, load_default_config, load_experiment_config
from itchi.constants import QUADRANTS
from itchi.ibtracs import _decode_text_sequence, get_ibtracs_storm_ids, ibtracs_to_track_dataframe, read_ibtracs_dataset
from itchi.municipal import land_mask, load_municipalities, municipal_values
from itchi.parametric_radii import add_parametric_radii
from itchi.plotting import classify_itchi_to_siat_levels
from itchi.precipitation import compute_antecedent_precipitation, compute_precipitation_matrix_hazard
from itchi.rocloud import DEFAULT_ROCLOUD_COLUMN_MAP, ROCLOUD_TEXT_COLUMNS, standardize_rocloud_text_dataframe
from itchi.tracks import DEFAULT_R34_COLUMN_MAP, filter_synoptic_times
from itchi.units import NAUTICAL_MILE_TO_KM
from itchi.wind_profiles import V_R34_MS

ROOT = get_project_root()
EARTH_RADIUS_KM = 6371.0
V_OUTER_MS = 2.0                        # R_W06: W06 wind = 2 m/s (PA21)
ROC_COLS = [DEFAULT_ROCLOUD_COLUMN_MAP[q] for q in QUADRANTS]
R34_COLS = [DEFAULT_R34_COLUMN_MAP[q] for q in QUADRANTS]

# ============================================================================
# CATÁLOGO
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
        evento="La Pintada", municipio="Atoyac de Álvarez", cve_ent="12", cvegeo="12011",
        t_local="2013-09-16 15:30", utc_offset_h=-5, incertidumbre_h=1.0,
        nota=("Alud que sepultó parte de La Pintada (Sierra de Atoyac, Guerrero) hacia las 15:30 del lunes "
              "16 de septiembre de 2013 (La Jornada, 3 nov 2013; Proceso, 21 sep 2013). El centro de Manuel se "
              "disipó sobre México hacia las 06 UTC del 16 y se regeneró en el golfo de California el 17 "
              "(NHC TCR EP132013), así que no hay posición del ciclón al ocurrir el deslizamiento. Ingrid "
              "estaba activa en el golfo de México al mismo tiempo."),
    ),
    "EARL2016": dict(
        evento="Huauchinango (colonias La Joya, Guadalupe, Aurora, La Cumbre; Xaltepec)",
        municipio="Huauchinango", cve_ent="21", cvegeo="21071",
        t_local="2016-08-07 03:00", utc_offset_h=-5, incertidumbre_h=3.0,
        nota=("Deslizamientos en la madrugada del domingo 7 de agosto de 2016 con remanentes de Earl; "
              "la hora exacta no se reporta (se toma 03:00 ± 3 h). 24 muertos en Huauchinango "
              "(ERN, boletín del 8 de agosto de 2016)."),
    ),
}


def t_event_utc(case: str) -> pd.Timestamp:
    d = DESLIZAMIENTOS[case]
    return pd.Timestamp(d["t_local"]) - pd.Timedelta(hours=d["utc_offset_h"])

# Recorte de la malla: México con 1° de margen
MEXICO_BOX = dict(lon_min=-119.0, lon_max=-85.0, lat_min=13.5, lat_max=33.5)


def local_path(path) -> Path:
    """"D:/..." -> "/mnt/d/..." en WSL/Linux; sin cambios en Windows; relativas a la raíz del repositorio."""
    s = str(path).replace("\\", "/")
    if os.name != "nt" and re.match(r"^[A-Za-z]:/", s):
        return Path(f"/mnt/{s[0].lower()}") / s[3:]
    p = Path(s)
    return p if p.is_absolute() or re.match(r"^[A-Za-z]:/", s) else ROOT / p


def _cfg_value(*keys):
    node = load_default_config()
    for k in keys:
        node = node[k]
    return node


@dataclass
class CaseConfig:
    """Rutas y opciones de los casos (por defecto, las de configs/default.yaml)."""

    mswep_6h_dir: Path = field(default_factory=lambda: local_path(_cfg_value("precipitation", "accumulations_dir")))
    clim: dict = field(default_factory=lambda: {
        d: local_path(_cfg_value("precipitation", f"climatology_{d}h")) for d in (6, 24, 72)})
    clim_var: str = field(default_factory=lambda: _cfg_value("precipitation", "climatology_variable"))
    tracks_dir: Path = field(default_factory=lambda: local_path(_cfg_value("paths", "tracks")))
    rocloud_dir: Path = field(default_factory=lambda: local_path(_cfg_value("paths", "rocloud")))
    mun_shp: Path = field(default_factory=lambda: local_path(_cfg_value("paths", "municipalities")))
    decl_file: Path = field(default_factory=lambda: local_path(_cfg_value("paths", "declaratorias")))
    cache_dir: Path = field(default_factory=lambda: local_path(_cfg_value("outputs", "cases")))
    fig_dir: Path = field(default_factory=lambda: local_path(_cfg_value("outputs", "figures")) / "casos")
    # experimento -> overlay (None = configs/default.yaml)
    experiments: dict = field(default_factory=lambda: {"itchi": None})
    box: dict = field(default_factory=lambda: dict(MEXICO_BOX))
    # H_P continuo: horas antes de la primera y después de la última posición
    hp_pad_before_h: int = 48
    hp_pad_after_h: int = 72
    municipal_stat: str = "max"            # "max" | "mean" | "p90"


def experiment_settings(cfg: CaseConfig | None = None) -> dict[str, dict]:
    """Argumentos de compute_itchi_snapshot de cada experimento."""
    cfg = CaseConfig() if cfg is None else cfg
    return {tag: itchi_settings_from_config(None if f is None else load_experiment_config(local_path(f)))
            for tag, f in cfg.experiments.items()}


# ============================================================================
# TRAYECTORIA Y SELECCIÓN DE POSICIONES
# ============================================================================


def read_rocloud_csv(path: str | Path) -> pd.DataFrame:
    """ROCLOUD plano (17 columnas, sin encabezado): 0 = sin detección; MWS en km/h."""
    raw = pd.read_csv(path, header=None, names=list(ROCLOUD_TEXT_COLUMNS), na_values=[-9999])
    df = standardize_rocloud_text_dataframe(raw)
    df[ROC_COLS] = df[ROC_COLS].where(df[ROC_COLS] > 0)
    df["vmax_kt_rocloud"] = df.pop("vmax_kt") / 1.852
    return df


def find_ibtracs_sid(dataset, name: str, season: int) -> str:
    sids = np.asarray(get_ibtracs_storm_ids(dataset))
    names = np.asarray(_decode_text_sequence(dataset["name"].values))
    seasons = np.asarray(dataset["season"].values).astype(int)
    match = sids[(names == name) & (seasons == season)]
    if match.size != 1:
        raise ValueError(f"{name} {season}: coincidencias {match}")
    return str(match[0])


def augment_track(track: pd.DataFrame, rocloud: pd.DataFrame, atcf: str, v_outer_ms: float = V_OUTER_MS) -> pd.DataFrame:
    """Radios W06 PA21 (pa21_r2_km = R_W06, pa21_r34_km), R34 medio observado y ROCLOUD."""
    track = add_parametric_radii(
        track, use_observed_rmw=False, calibrate_to_r34=False, rmax_variant="pa21",
        rmw_unit="km", r34_unit="nm", prefix="pa21_",
        thresholds_ms={"r2_km": v_outer_ms, "r34_km": V_R34_MS},
    )
    track["r34_obs_nmi"] = track[R34_COLS].where(track[R34_COLS] > 0).mean(axis=1)
    track["r34_obs_km"] = track["r34_obs_nmi"] * NAUTICAL_MILE_TO_KM
    roc = rocloud.loc[rocloud["storm_id"] == atcf, ["time", "rocloud_mean_km"] + ROC_COLS]
    return track.merge(roc, on="time", how="left")


def load_track(case: str, cfg: CaseConfig | None = None) -> pd.DataFrame:
    """Trayectoria sinóptica IBTrACS con R34, radios W06 PA21 y ROCLOUD."""
    cfg = CaseConfig() if cfg is None else cfg
    c = CASES[case]
    dataset = read_ibtracs_dataset(cfg.tracks_dir / f"IBTrACS.{c['basin']}.v04r01.nc")
    track = ibtracs_to_track_dataframe(dataset, find_ibtracs_sid(dataset, c["name"], c["season"]))
    track = filter_synoptic_times(track).dropna(subset=["vmax_kt", "lat", "lon"]).reset_index(drop=True)
    track = augment_track(track, read_rocloud_csv(cfg.rocloud_dir / f"ROCLOUD_{c['basin']}.csv"), c["atcf"])
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


def read_precip(times: pd.DatetimeIndex, cfg: CaseConfig | None = None):
    """
    P6, R24 y A72 en todos los tiempos de 6 h entre min(times) - pad y
    max(times) + pad (sin huecos), recortados a cfg.box.
    """
    from itchi.mswep import subset_mswep_domain

    cfg = CaseConfig() if cfg is None else cfg
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


def read_climatologies(template: xr.DataArray, cfg: CaseConfig | None = None) -> tuple[dict, dict, dict]:
    """Percentiles de 6, 24 y 72 h en la malla de template."""
    cfg = CaseConfig() if cfg is None else cfg
    return tuple(load_percentile_climatology(cfg.clim[d], variable=cfg.clim_var, template=template) for d in (6, 24, 72))


def precipitation_hazard(p6, r24, a72, q6, q24, q72, spec: dict, nan_value: float | None = 0.0):
    hp = compute_precipitation_matrix_hazard(
        precipitation=p6, quantiles_6h=q6,
        escalation_values={"R24": r24, "A72": a72},
        escalation_quantiles={"R24": q24, "A72": q72}, matrix=spec)
    if nan_value is None:
        return hp
    return hp.fillna(nan_value) if isinstance(hp, xr.DataArray) else np.where(np.isfinite(hp), hp, nan_value)


def siat_class(x) -> np.ndarray:
    """Clase SIAT 0–5 en tramos de 0.2, (inferior, superior]; NaN -> 0."""
    return np.asarray(classify_itchi_to_siat_levels(np.nan_to_num(np.asarray(x, dtype=float)))).astype(np.int8)


def compute_case_fields(track: pd.DataFrame, p6: xr.DataArray, r24: xr.DataArray, a72: xr.DataArray,
                        q6: dict, q24: dict, q72: dict, settings: dict[str, dict],
                        verbose: bool = True) -> xr.Dataset:
    """
    ITCHI de cada experimento en las posiciones de track (tiempos dentro de
    p6) y H_P continuo de cada experimento en todos los tiempos de p6.

    No lee archivos: track es la trayectoria aumentada (load_track +
    select_positions) y p6, r24, a72 son DataArrays (time, lat, lon).
    """
    from itchi.pipeline import compute_itchi_snapshot

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
                res = compute_itchi_snapshot(p6.values[j], r24.values[j], a72.values[j], **common, **settings[tag])
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
    hp_cont = np.stack([
        np.asarray(precipitation_hazard(p6.values, r24.values, a72.values, *q_np,
                                        spec=settings[tag]["precip_matrix"],
                                        nan_value=settings[tag]["precipitation_nan_value"]), np.float32)
        for tag in tags])

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
    data["H_P_cont"] = (("exp", "time_all", "lat", "lon"), hp_cont)
    for k, v in zip((6, 24, 72), q_np, strict=True):
        for name, f in v.items():
            data[f"Q{k}h_{name}"] = (("lat", "lon"), np.asarray(f, np.float32))
    ds = xr.Dataset(data, coords={"exp": tags, "time": tr["time"].values, "time_all": t_all.values,
                                  "lat": lat, "lon": lon})
    return ds


def case_grid(season: int, cfg: CaseConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """lat y lon del recorte cfg.box en el archivo de 6 h de la temporada."""
    from itchi.mswep import subset_mswep_domain

    cfg = CaseConfig() if cfg is None else cfg
    with xr.open_dataset(cfg.mswep_6h_dir / f"mswep_v3_6h_{season}.nc") as ds:
        p = subset_mswep_domain(ds["precipitation"].isel(time=0, drop=True), **cfg.box)
        return p["lat"].values.copy(), p["lon"].values.copy()


def load_or_compute(case: str, cfg: CaseConfig | None = None, force: bool = False, verbose: bool = True):
    """
    Campos del caso (caché en cfg.cache_dir), trayectoria completa, municipios.

    Devuelve dict con: ds (xr.Dataset), track (todas las posiciones),
    gdf, pix, codes y land (máscara terrestre).
    """
    cfg = CaseConfig() if cfg is None else cfg
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    f = cfg.cache_dir / f"{case.lower()}.nc"
    track_all = load_track(case, cfg)

    # Malla del recorte: la del archivo de 6 h de la temporada (la misma de P6)
    lat, lon = case_grid(CASES[case]["season"], cfg)
    gdf, pix = load_municipalities(cfg.mun_shp, lat, lon)
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
                        experiments="; ".join(f"{k}: {v or 'configs/default.yaml'}" for k, v in cfg.experiments.items()))
        ds.to_netcdf(f, encoding={v: {"zlib": True, "complevel": 2} for v in ds.data_vars})
        if verbose:
            print(f"  guardado {f} ({f.stat().st_size / 1e6:.0f} MB)")

    if not (np.allclose(ds["lat"].values, lat) and np.allclose(ds["lon"].values, lon)):
        warnings.warn("la malla del caché no coincide con la de los municipios; usa force=True", stacklevel=2)
    return dict(case=case, ds=ds, track=track_all, gdf=gdf, pix=pix, codes=codes,
                land=land_mask(pix, (lat.size, lon.size)))


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
            rows.append(dict(exp=e, clase=k, municipios=int((cm == k).sum()),
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


def peak_time(C: dict, tag: str | None = None) -> pd.Timestamp:
    """Posición con mayor ITCHI acumulado sobre tierra (pico del evento sobre México)."""
    ds = C["ds"]
    tag = str(ds["exp"].values[0]) if tag is None else tag
    total = (ds["ITCHI"].sel(exp=tag).values * C["land"][None]).reshape(ds.sizes["time"], -1).sum(axis=1)
    return pd.Timestamp(ds["time"].values[int(np.argmax(total))])


def load_declaratorias(cfg: CaseConfig | None = None) -> pd.DataFrame | None:
    """Declaratorias asociadas a ciclones (scripts/declaratorias_6h.py); None si no existe el archivo."""
    cfg = CaseConfig() if cfg is None else cfg
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
            rows.append(dict(exp=e, clase=k, municipios=int(m.sum()), con_declaratoria=int((m & has_d).sum()),
                             P_D=(m & has_d).sum() / m.sum() if m.sum() else np.nan))
    return pd.DataFrame(rows)


def percentile_level(values: np.ndarray, quantiles: dict[str, np.ndarray | float]) -> np.ndarray:
    """Percentil más alto superado (75, 90, ...; 0 si ninguno) por elemento."""
    names = sorted(quantiles, key=lambda k: float(k[1:]))
    out = np.zeros(np.shape(values))
    for k in names:
        out = np.where(np.asarray(values) >= np.asarray(quantiles[k]), float(k[1:]), out)
    return out
