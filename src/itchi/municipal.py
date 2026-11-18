"""
Municipal aggregation of ITCHI fields.

A municipality is represented by the grid cells whose centre falls inside it
(declaratorias_6h.municipio_pixels); a municipality without centres takes the
cell closest to its representative point. Its value is a statistic of those
cells (maximum by default).
"""

from __future__ import annotations

import unicodedata
import warnings
from pathlib import Path

import numpy as np


def load_municipalities(shapefile: str | Path, lat: np.ndarray, lon: np.ndarray):
    """
    Municipalities whose representative point is inside the grid, and their cells.

    Returns (gdf, pix) with pix[CVEGEO] = (flat cell indices, number of cell
    centres inside the municipality).
    """
    import geopandas as gpd

    from itchi.declaratorias_6h import municipio_pixels

    gdf = gpd.read_file(shapefile)
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


def _norm_name(x) -> str:
    """
    Nombre comparable: repara el texto UTF-8 leído como latin-1 ("Ã\x81lvarez"),
    quita acentos y deja solo letras, dígitos y espacios en minúsculas.
    """
    t = str(x)
    if "Ã" in t or "Â" in t:
        try:
            t = t.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    t = "".join(ch for ch in unicodedata.normalize("NFKD", t) if not unicodedata.combining(ch)).lower()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in t).split())


def find_municipio(gdf, nombre: str, cve_ent: str, cvegeo: str | None = None) -> str:
    """
    CVEGEO del municipio por nombre y clave de entidad. La comparación no
    distingue acentos, mayúsculas ni puntuación y repara acentos mal
    decodificados. Si el nombre no aparece y se da cvegeo, usa esa clave
    (avisando con el NOMGEO del shapefile); si no, el error lista los
    municipios de la entidad que comparten alguna palabra con el nombre.
    """
    codes = gdf["CVEGEO"].astype(str).str.zfill(5)
    in_ent = codes.str[:2] == str(cve_ent).zfill(2)
    names = gdf["NOMGEO"].map(_norm_name)
    target = _norm_name(nombre)
    hit = gdf[in_ent & (names == target)]
    if len(hit) == 0:                                   # todas las palabras del nombre
        words = set(target.split()) - {"de", "del", "la", "las", "los", "el"}
        hit = gdf[in_ent & names.map(lambda n: words <= set(n.split()))]
    if len(hit) == 1:
        return str(hit["CVEGEO"].iloc[0]).zfill(5)
    if len(hit) == 0 and cvegeo is not None and (codes == str(cvegeo).zfill(5)).any():
        nom = gdf.loc[codes == str(cvegeo).zfill(5), "NOMGEO"].iloc[0]
        warnings.warn(f"{nombre!r} no se encontró por nombre; se usa CVEGEO {cvegeo} (NOMGEO = {nom!r})", stacklevel=2)
        return str(cvegeo).zfill(5)
    first = target.split()[0] if target else ""
    similar = sorted(gdf.loc[in_ent & names.str.contains(first, regex=False), "NOMGEO"].astype(str))[:15]
    raise ValueError(f"{nombre} ({cve_ent}): {len(hit)} coincidencias. "
                     f"Municipios de la entidad {cve_ent} con '{first}': {similar} "
                     f"(municipios de la entidad en la malla: {int(in_ent.sum())})")
