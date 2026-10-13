"""
Utilidades compartidas por los notebooks 02 y 03 (casos Cristóbal 2020 y
Odile 2014): radios W06 con metodología PA21, radios de Chavas y ROCLOUD.

Radios exteriores comparados (km):

- ``pa21_r2_km``    : W06 con Rmax y forma por regresión (PA21), V = 2 m s-1.
- ``chavas_r2_km``  : Chavas a partir del R34 observado, V = 2 m s-1.
                      Fuente ``tcwindprofile`` si está instalado y el mapeo
                      de argumentos es válido; si no, el modelo exterior E04
                      propio (``itchi.chavas_outer``).
- ``rocloud_*``     : ROCLOUD por cuadrante y ``rocloud_mean_km`` (Rp).
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from itchi.chavas_outer import e04_outer_wind, e04_radius_of_wind
from itchi.constants import QUADRANTS
from itchi.ibtracs import (
    _decode_text_sequence,
    get_ibtracs_storm_ids,
    ibtracs_to_track_dataframe,
    read_ibtracs_dataset,
)
from itchi.parametric_radii import add_parametric_radii
from itchi.radii import fill_missing_quadrant_radii, resolve_direct_radius
from itchi.rocloud import (
    DEFAULT_ROCLOUD_COLUMN_MAP,
    ROCLOUD_TEXT_COLUMNS,
    standardize_rocloud_text_dataframe,
)
from itchi.tracks import DEFAULT_R34_COLUMN_MAP, filter_synoptic_times
from itchi.units import NAUTICAL_MILE_TO_KM, convert_quadrant_radii_to_km
from itchi.wind import compute_radial_wind_profile
from itchi.wind_profiles import KNOT_TO_MS, V_R34_MS, Willoughby2006Profile

EARTH_RADIUS_KM = 6371.0
V_OUTER_MS = 2.0
PENV_HPA = 1008.0
ROC_COLS = [DEFAULT_ROCLOUD_COLUMN_MAP[q] for q in QUADRANTS]
R34_COLS = [DEFAULT_R34_COLUMN_MAP[q] for q in QUADRANTS]
QUADRANT_BEARINGS = {"RNE": (0.0, 90.0), "RSE": (90.0, 180.0),
                     "RSW": (180.0, 270.0), "RNW": (270.0, 360.0)}


# ------------------------------------------------------------------ lectura
def read_rocloud_csv(path: str | Path) -> pd.DataFrame:
    """ROCLOUD plano (17 columnas, sin encabezado): 0 = sin detección; MWS en km/h."""
    raw = pd.read_csv(path, header=None, names=list(ROCLOUD_TEXT_COLUMNS),
                      na_values=[-9999])
    df = standardize_rocloud_text_dataframe(raw)
    df[ROC_COLS] = df[ROC_COLS].where(df[ROC_COLS] > 0)
    df["vmax_kt_rocloud"] = df.pop("vmax_kt") / 1.852
    return df


def translation_speed_kt(track: pd.DataFrame) -> np.ndarray:
    """Rapidez de traslación (kt) por diferencias centradas de la trayectoria."""
    lat = np.deg2rad(track["lat"].to_numpy(float))
    lon = np.deg2rad(track["lon"].to_numpy(float))
    if len(lat) < 2:
        return np.full(len(lat), np.nan)
    a = (np.sin(np.diff(lat) / 2) ** 2
         + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(lon) / 2) ** 2)
    dist_km = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
    dt_h = np.diff(track["time"].to_numpy()).astype("timedelta64[s]").astype(float) / 3600
    speed = dist_km / dt_h / NAUTICAL_MILE_TO_KM
    return np.concatenate([[speed[0]], 0.5 * (speed[1:] + speed[:-1]), [speed[-1]]])


def find_ibtracs_sid(dataset, name: str, season: int) -> str:
    sids = np.asarray(get_ibtracs_storm_ids(dataset))
    names = np.asarray(_decode_text_sequence(dataset["name"].values))
    seasons = np.asarray(dataset["season"].values).astype(int)
    match = sids[(names == name) & (seasons == season)]
    if match.size != 1:
        raise ValueError(f"{name} {season}: coincidencias {match}")
    return str(match[0])


def load_case_track(ibtracs_file: str | Path, rocloud_file: str | Path, name: str,
                    season: int, atcf: str, v_outer_ms: float = V_OUTER_MS) -> pd.DataFrame:
    """Trayectoria sinóptica con R34 observado, radios W06 PA21 y ROCLOUD."""
    dataset = read_ibtracs_dataset(ibtracs_file)
    track = ibtracs_to_track_dataframe(dataset, find_ibtracs_sid(dataset, name, season))
    track = filter_synoptic_times(track).dropna(subset=["vmax_kt", "lat", "lon"])
    track = track.reset_index(drop=True)
    return augment_track(track, read_rocloud_csv(rocloud_file), atcf, v_outer_ms)


def augment_track(track: pd.DataFrame, rocloud: pd.DataFrame, atcf: str,
                  v_outer_ms: float = V_OUTER_MS) -> pd.DataFrame:
    track = add_parametric_radii(
        track, use_observed_rmw=False, calibrate_to_r34=False, rmax_variant="pa21",
        rmw_unit="km", r34_unit="nm", prefix="pa21_",
        thresholds_ms={"r2_km": v_outer_ms, "r34_km": V_R34_MS},
    )
    track["r34_obs_nmi"] = track[R34_COLS].where(track[R34_COLS] > 0).mean(axis=1)
    track["r34_obs_km"] = track["r34_obs_nmi"] * NAUTICAL_MILE_TO_KM
    track["vtrans_kt"] = translation_speed_kt(track)
    roc = rocloud.loc[rocloud["storm_id"] == atcf, ["time", "rocloud_mean_km"] + ROC_COLS]
    return track.merge(roc, on="time", how="left")


# ------------------------------------------------------------------ Chavas
def tcw_function():
    """Devuelve ``run_full_wind_model`` o None si tcwindprofile no está instalado."""
    try:
        from tcwindprofile import run_full_wind_model
    except ImportError:
        return None
    return run_full_wind_model


# subcadenas del nombre del parámetro -> variable
TCW_ARG_PATTERNS: dict[tuple[str, ...], str] = {
    ("vmax",): "vmax_kt", ("r34",): "r34", ("lat",): "lat",
    ("trans",): "vtrans_kt", ("env",): "penv_hpa",
}


def _first_match(names, patterns):
    for name in names:
        if all(p in str(name).lower() for p in patterns):
            return name
    return None


def tcw_profile(row: Mapping, penv_hpa: float = PENV_HPA, verbose: bool = False):
    """
    Perfil completo de tcwindprofile para un registro: (r_km, v_ms, r0_km).

    El mapeo de argumentos se infiere de la firma; con verbose=True se imprimen
    los argumentos y las claves de salida para verificarlo.
    """
    func = tcw_function()
    if func is None or not np.isfinite(row["r34_obs_km"]):
        return None
    signature = inspect.signature(func)
    values = {"vmax_kt": float(row["vmax_kt"]), "lat": float(row["lat"]),
              "vtrans_kt": float(row["vtrans_kt"]), "penv_hpa": penv_hpa}
    kwargs = {}
    for patterns, var in TCW_ARG_PATTERNS.items():
        name = _first_match(signature.parameters, patterns)
        if name is None:
            continue
        if var == "r34":
            use_km = "km" in name.lower()
            kwargs[name] = float(row["r34_obs_km"] if use_km else row["r34_obs_nmi"])
        else:
            kwargs[name] = values[var]
    if "plot" in signature.parameters:
        kwargs["plot"] = False
    out = func(**kwargs)
    if verbose:
        print("firma:", signature)
        print("argumentos:", kwargs)
        print("salida:", type(out), list(out) if isinstance(out, dict) else "")
    if not isinstance(out, dict):
        return None
    k_rr, k_vv, k_r0 = (_first_match(out, ("rr",)), _first_match(out, ("vv",)),
                        _first_match(out, ("r0",)))
    if k_rr is None or k_vv is None:
        return None
    rr = np.asarray(out[k_rr], dtype=float).ravel()
    vv = np.asarray(out[k_vv], dtype=float).ravel()
    if "km" not in str(k_rr).lower() and np.nanmax(rr) > 1.0e4:
        rr = rr / 1.0e3                                           # m -> km
    r0 = float(np.ravel(out[k_r0])[0]) if k_r0 is not None else np.nan
    return rr, vv, r0


def outer_radius_from_profile(r_km: np.ndarray, v_ms: np.ndarray, v_thr: float) -> float:
    """Radio exterior donde el perfil cruza v_thr (rama decreciente)."""
    i_max = int(np.nanargmax(v_ms))
    r_out, v_out = r_km[i_max:], v_ms[i_max:]
    if not np.any(v_out <= v_thr):
        return np.nan
    return float(np.interp(v_thr, v_out[::-1], r_out[::-1]))


def add_chavas_radii(track: pd.DataFrame, v_outer_ms: float = V_OUTER_MS,
                     use_tcwindprofile: bool = True, verbose_first: bool = True) -> pd.DataFrame:
    """Añade e04_*, tcw_* y chavas_r2_km / chavas_r0_km / chavas_source."""
    track = track.copy()
    e04 = [e04_radius_of_wind(r34, lat, v_outer_ms) if np.isfinite(r34) else (np.nan, np.nan)
           for r34, lat in zip(track["r34_obs_km"], track["lat"])]
    track["e04_r0_km"], track["e04_r2_km"] = np.array(e04, dtype=float).reshape(-1, 2).T

    track["tcw_r0_km"], track["tcw_r2_km"] = np.nan, np.nan
    if use_tcwindprofile and tcw_function() is not None:
        first = verbose_first
        for i, row in track.iterrows():
            prof = tcw_profile(row, verbose=first)
            first = False
            if prof is not None:
                rr, vv, r0 = prof
                track.loc[i, "tcw_r0_km"] = r0
                track.loc[i, "tcw_r2_km"] = outer_radius_from_profile(rr, vv, v_outer_ms)

    use_tcw = track["tcw_r2_km"].notna()
    track["chavas_r2_km"] = track["tcw_r2_km"].where(use_tcw, track["e04_r2_km"])
    track["chavas_r0_km"] = track["tcw_r0_km"].where(use_tcw, track["e04_r0_km"])
    track["chavas_source"] = np.where(use_tcw, "tcwindprofile",
                                      np.where(track["e04_r2_km"].notna(), "e04", "none"))
    return track


# ------------------------------------------------------------------ perfiles
def radial_profiles(row: Mapping, r_km: np.ndarray) -> dict[str, np.ndarray]:
    """Perfiles V(r) en m s-1 sobre la misma malla radial (NaN donde no aplica)."""
    vmax_kt, lat = float(row["vmax_kt"]), float(row["lat"])
    rmw = float(row["rmw_km"]) if np.isfinite(row["rmw_km"]) else float(row["pa21_rmax_km"])
    out = {
        "Rankine (wind.py)": compute_radial_wind_profile(r_km, vmax_kt=vmax_kt,
                                                         rmw_km=rmw) * KNOT_TO_MS,
        "W06 PA21": Willoughby2006Profile(vmax_kt * KNOT_TO_MS, lat,
                                          rmax_variant="pa21")(r_km),
    }
    prof = tcw_profile(row) if tcw_function() is not None else None
    if prof is not None:
        rr, vv, _ = prof
        out["Chavas (tcwindprofile)"] = np.interp(r_km, rr, vv, left=np.nan, right=0.0)
    elif np.isfinite(row.get("e04_r0_km", np.nan)):
        v = e04_outer_wind(r_km, float(row["e04_r0_km"]), lat)
        out["Chavas E04 (r >= R34)"] = np.where(r_km >= row["r34_obs_km"], v, np.nan)
    return out


# ------------------------------------------------------------------ geometría
def destination(lat: float, lon: float, bearing_deg: np.ndarray,
                dist_km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Punto destino sobre la esfera (rumbo desde el norte, sentido horario)."""
    phi1, lam1 = np.deg2rad(lat), np.deg2rad(lon)
    theta = np.deg2rad(np.asarray(bearing_deg, dtype=float))
    delta = np.asarray(dist_km, dtype=float) / EARTH_RADIUS_KM
    phi2 = np.arcsin(np.sin(phi1) * np.cos(delta)
                     + np.cos(phi1) * np.sin(delta) * np.cos(theta))
    lam2 = lam1 + np.arctan2(np.sin(theta) * np.sin(delta) * np.cos(phi1),
                             np.cos(delta) - np.sin(phi1) * np.sin(phi2))
    return np.rad2deg(phi2), (np.rad2deg(lam2) + 540.0) % 360.0 - 180.0


def circle_outline(lat: float, lon: float, radius_km: float, n: int = 181):
    """Contorno (lon, lat) de un radio simétrico."""
    if not np.isfinite(radius_km):
        return np.array([]), np.array([])
    bearings = np.linspace(0.0, 360.0, n)
    plat, plon = destination(lat, lon, bearings, np.full(n, radius_km))
    return plon, plat


def quadrant_outline(lat: float, lon: float, radii_km: Mapping[str, float], n: int = 31):
    """Contorno (lon, lat) de radios por cuadrante; NaN abre el contorno en ese sector."""
    lons, lats = [], []
    for q in QUADRANTS:
        b0, b1 = QUADRANT_BEARINGS[q]
        radius = float(radii_km.get(q, np.nan))
        bearings = np.linspace(b0, b1, n)
        if np.isfinite(radius) and radius > 0:
            plat, plon = destination(lat, lon, bearings, np.full(n, radius))
        else:
            plat, plon = np.full(n, np.nan), np.full(n, np.nan)
        lons.append(plon)
        lats.append(plat)
    lons.append(lons[0][:1])
    lats.append(lats[0][:1])
    return np.concatenate(lons), np.concatenate(lats)


def rocloud_radii(row: Mapping) -> dict[str, float]:
    return {q: float(row[DEFAULT_ROCLOUD_COLUMN_MAP[q]]) for q in QUADRANTS}


def r34_radii_km(row: Mapping) -> dict[str, float]:
    values = {q: row[DEFAULT_R34_COLUMN_MAP[q]] for q in QUADRANTS}
    return convert_quadrant_radii_to_km(values, input_unit="nm")


# ------------------------------------------------------------------ atribución
def direct_radii_km(row: Mapping) -> dict[str, float]:
    """Radio directo resuelto como en el pipeline (R34 obs, respaldo W06 PA21)."""
    rmw = float(row["rmw_km"]) if np.isfinite(row["rmw_km"]) else float(row["pa21_rmax_km"])
    fallback = float(row["pa21_r34_km"]) if np.isfinite(row["pa21_r34_km"]) else None
    return resolve_direct_radius(r34_radii_km(row), vmax_kt=float(row["vmax_kt"]),
                                 rmw_km=rmw, fallback_radius_km=fallback).radii


def attribution_radii_km(row: Mapping, mode: str):
    """
    Radio de atribución por cuadrante (km) para mode in {"rocloud", "rocloud_mean", "pa21", "chavas"}.

    Devuelve (radios o None, n cuadrantes corregidos). Cuadrantes faltantes o
    menores que el radio directo se rellenan con la media de los válidos y, si
    aun así quedan por debajo, se igualan al radio directo.
    """
    direct = direct_radii_km(row)
    if mode == "rocloud":
        raw = rocloud_radii(row)
    elif mode == "rocloud_mean":
        raw = dict.fromkeys(QUADRANTS, float(row["rocloud_mean_km"]))
    elif mode == "pa21":
        raw = dict.fromkeys(QUADRANTS, float(row["pa21_r2_km"]))
    elif mode == "chavas":
        raw = dict.fromkeys(QUADRANTS, float(row["chavas_r2_km"]))
    else:
        raise ValueError(mode)

    invalid = [q for q in QUADRANTS if not (np.isfinite(raw[q]) and raw[q] >= direct[q])]
    if len(invalid) == len(QUADRANTS):
        return None, len(invalid)
    filled = fill_missing_quadrant_radii(
        {q: (np.nan if q in invalid else raw[q]) for q in QUADRANTS}).radii
    return {q: max(filled[q], direct[q]) for q in QUADRANTS}, len(invalid)
