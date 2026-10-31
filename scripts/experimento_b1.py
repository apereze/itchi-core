"""
ITCHI-CORE · Experimento del Bloque 1 (v0.2.1)

Pregunta: ¿qué cambia en la discriminación de declaratorias si (P) el percentil 95
de 6 h es el peligro máximo de lluvia, con los umbrales de R24 y A72 escalando las
clases inferiores, y (W) la cola de viento se corta en 12 m/s?

Unidad: par municipio–ciclón de negativos_w06.py (municipio dentro de R_W06 en
algún paso; D = 1 si hubo declaratoria asociada al ciclón).

Lluvia (librería itchi, mismas funciones que la malla):

    v0.2          matriz v0.2 (P95-P99 -> 0.8 si R24 >= P95 y A72 >= P75)
    M0            solo P6: P75 -> 0, P90 -> 0.5, P95 -> 1 (referencia sin R24/A72)
    M1            M0 + escalada recorrida: con R24 >= P95 y A72 >= P75, P90 ya es máximo
    M1[...]       rejilla de umbrales (R24 >= P90/P95, A72 >= P75/P90/P95) y ablaciones
                  (solo R24, solo A72): el impacto de cada umbral
    M2            máximo de las tres duraciones, cada una con su escalera de percentiles,
                  solo con P6 >= P75 (compuerta)
    M3            tabla 4x4x4 (clases P6 x R24 x A72) calibrada con P(D | celda):
                  cada celda recibe el peligro de la clase de solo 6 h con la misma
                  probabilidad de declaratoria; nunca por debajo de M0; validación
                  cruzada dejando un año fuera

Viento (aproximación de docs/verification_v02.md, sección 4): perfil W06 (PA21) del
CT en la distancia del municipio; H_W = h34 + (1 - h34)(V - 34)/(Vmax - 34) si V >= 34 kt,
cola h34 (V - V_out)/(34 - V_out) fuera, V_out = max(V(R_W06) = 2 m/s, corte).

    sin viento | W0 cola hasta R_W06 | W1 corte 12 m/s | ref corte 17 kt

ITCHI = 1 - (1 - H_W)(1 - H_P); clases SIAT de 0.2 (1: > 0, ..., 5: > 0.8).

Modos:
    pasos   (si existe pasos_municipio_ct_w06.csv.gz; negativos_w06.py --guardar-pasos)
            el índice se calcula en cada paso y el par toma su máximo
    pico    (si no) el índice se evalúa en el paso del pico de P6 del par; subestima
            los esquemas cuyo máximo no coincide con el pico de P6

Puntajes: AUC (promedio por año), PSS por clase, POD, POFD, FAR, CSI, SEDI, sesgo de
frecuencia; ΔPSS contra v0.2 con intervalo bootstrap por ciclón (90 %); ΔPOD a igual
POFD contra la curva ROC de v0.2 (si ≈ 0, el cambio solo desplaza el umbral).

Salidas (outputs/diagnostics/experimento_b1/):
    celdas_P6_R24_A72.csv    n, declaratorias y P(D) por celda de clases (evidencia)
    lluvia_candidatos.csv    solo lluvia: esquemas x clase
    diseno_2x2.csv           lluvia x viento: puntajes y diferencias contra v0.2
    tabla_m3.yaml            tabla M3 ajustada con todos los años (spec class_table)
    por_par.csv.gz           H_P e ITCHI de cada par por esquema (curvas ROC del notebook 09)
    resumen_b1.md            tablas principales

Ejecutar desde scripts/ (después de negativos_w06.py):
    python experimento_b1.py
    python experimento_b1.py --modo pico --n-boot 200
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    import itchi  # noqa: F401
except ModuleNotFoundError:   # paquete no instalado: usar src/ del repo
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itchi.climatology import quantile_dataarray_to_dict
from itchi.precipitation import (
    PRECIP_MATRIX_V02,
    PRECIP_TABLES_B1,
    build_precipitation_class_table,
    compute_percentile_coordinate,
    compute_precipitation_hazard_from_spec,
)
from itchi.wind_profiles import KNOT_TO_MS, Willoughby2006Profile

REPO = Path(__file__).resolve().parents[1]
NODES = ("Q75", "Q90", "Q95")
SIAT_EDGES = (1e-9, 0.2 + 1e-9, 0.4 + 1e-9, 0.6 + 1e-9, 0.8 + 1e-9)
V_OUTER_MS = 2.0


@dataclass
class ConfigB1:

    diag_dir: str = str(REPO / "outputs/diagnostics/declaratorias_6h")
    pares_file: str = "pares_municipio_ct_w06.csv.gz"
    pasos_file: str = "pasos_municipio_ct_w06.csv.gz"
    clim_pattern: str = str(REPO / "data/processed/mswep_v3_{d}h_percentiles_may_nov.nc")
    clim_var: str = "p_wet_raw"
    out_dir: str = str(REPO / "outputs/diagnostics/experimento_b1")
    modo: str = "auto"                     # "auto" | "pasos" | "pico"
    h34: float = 0.2
    viento: dict = field(default_factory=lambda: {
        "sin viento": None,
        "W0 cola hasta R_W06": V_OUTER_MS,
        "W1 corte 12 m/s": 12.0,
        "ref corte 17 kt": 17.0 * KNOT_TO_MS,
    })
    referencia: tuple = ("v0.2", "W0 cola hasta R_W06")
    m3_pseudo: float = 50.0               # pseudo-conteos hacia P(D | clase de P6)
    n_boot: int = 1000
    seed: int = 20261029


# ============================================================================
# ESQUEMAS DE LLUVIA
# ============================================================================


def m1_variant(r24: str | None, a72: str | None) -> dict:
    """M1 con otra condición de escalada (None = sin condición sobre esa variable)."""

    when = {k: v for k, v in (("R24", r24), ("A72", a72)) if v is not None}
    spec = {k: v for k, v in PRECIP_TABLES_B1["M1"].items() if k != "rules"}
    spec["rules"] = ({"when": when, "p6_values": (0.0, 0.0, 1.0, 1.0)},)
    return spec


def precip_schemes() -> dict[str, dict]:
    """Esquemas fijos (sin ajuste). M3 se ajusta aparte."""

    schemes = {"v0.2": PRECIP_MATRIX_V02, "M0": PRECIP_TABLES_B1["M0"],
               "M1": PRECIP_TABLES_B1["M1"], "M2": PRECIP_TABLES_B1["M2"]}
    for r in ("Q90", "Q95"):
        for a in ("Q75", "Q90", "Q95"):
            name = f"M1[R24>=P{r[1:]}, A72>=P{a[1:]}]"
            if (r, a) != ("Q95", "Q75"):
                schemes[name] = m1_variant(r, a)
    schemes["M1[solo R24>=P95]"] = m1_variant("Q95", None)
    schemes["M1[solo A72>=P75]"] = m1_variant(None, "Q75")
    return schemes


def class_cells(p6, r24, a72, q6, q24, q72) -> np.ndarray:
    """Clases 0..3 (<P75, P75-P90, P90-P95, >=P95) de cada variable; -1 si P6 falta."""

    cells = []
    for x, q in ((p6, q6), (r24, q24), (a72, q72)):
        u = compute_percentile_coordinate(x, q, NODES)
        cells.append(np.where(np.isfinite(u), np.clip(np.floor(u), 0, 3), -1).astype(int))
    cells = np.stack(cells, axis=1)
    cells[:, 1:] = np.maximum(cells[:, 1:], 0)          # escalada faltante = clase 0
    return cells


def fit_m3_table(cells: np.ndarray, d: np.ndarray, pseudo: float = 50.0,
                 base: tuple = (0.0, 0.0, 0.5, 1.0)) -> np.ndarray:
    """
    Tabla 4x4x4 de peligro desde P(D | celda).

    1. P(D | c6) marginal y P(D | celda) suavizada hacia ella con ``pseudo`` conteos.
    2. Envolvente monótona: máximo sobre las celdas dominadas (no decreciente en
       cada eje; sesgo hacia arriba en celdas poco pobladas).
    3. Peligro = interpolación de P(D | celda) sobre la curva (P(D | c6), base):
       la celda vale lo que la clase de solo 6 h con la misma probabilidad.
    4. Nunca por debajo de M0 (P95 sigue siendo el máximo) y nodo P75 sin
       escalada = 0 (con la compuerta, no hay alerta con P6 < P75).
    """

    ok = cells[:, 0] >= 0
    c, y = cells[ok], d[ok].astype(float)
    n = np.zeros((4, 4, 4))
    k = np.zeros((4, 4, 4))
    np.add.at(n, (c[:, 0], c[:, 1], c[:, 2]), 1.0)
    np.add.at(k, (c[:, 0], c[:, 1], c[:, 2]), y)

    n6, k6 = n.sum(axis=(1, 2)), k.sum(axis=(1, 2))
    p6 = np.maximum.accumulate(np.where(n6 > 0, k6 / np.maximum(n6, 1.0), 0.0))
    p = (k + pseudo * p6[:, None, None]) / (n + pseudo)
    for axis in range(3):
        p = np.maximum.accumulate(p, axis=axis)

    xp = p6 + 1e-12 * np.arange(4)                       # estrictamente creciente
    h = np.interp(p, xp, np.asarray(base, dtype=float))
    h = np.maximum(h, np.asarray(base)[:, None, None])
    h[0] = 0.0
    h[1, 0, 0] = 0.0
    for axis in range(3):
        h = np.maximum.accumulate(h, axis=axis)
    return h


def m3_spec(table: np.ndarray) -> dict:
    spec = {"kind": "class_table", "axes": {"P6": NODES, "R24": NODES, "A72": NODES},
            "table": np.round(table, 4).tolist(), "gate": {"P6": "Q75"}}
    build_precipitation_class_table(spec)               # valida forma y monotonía
    return spec


# ============================================================================
# VIENTO E ÍNDICE
# ============================================================================


def wind_kt_w06(vmax_kt: np.ndarray, lat: np.ndarray, dist_km: np.ndarray,
                return_rmax: bool = False):
    """Viento W06 (PA21, sin calibrar a R34) del CT en la distancia del municipio (kt)."""

    prof = Willoughby2006Profile(np.asarray(vmax_kt, float) * KNOT_TO_MS,
                                 np.asarray(lat, float), None, rmax_variant="pa21")
    v = prof(np.asarray(dist_km, float)) / KNOT_TO_MS
    return (v, np.broadcast_to(prof.rmax, v.shape)) if return_rmax else v


def wind_hazard(v_kt: np.ndarray, vmax_kt: np.ndarray, h34: float,
                v_out_ms: float | None, inside_rmax: np.ndarray | None = None) -> np.ndarray:
    """
    Misma forma que wind.normalize_wind_hazard_with_tail, con Vmax por renglón.
    Región directa: V >= 34 kt, o dentro del radio de viento máximo (inside_rmax).
    """

    if v_out_ms is None:
        return np.zeros_like(v_kt)
    v_out = max(float(v_out_ms), V_OUTER_MS) / KNOT_TO_MS
    tail = h34 * np.clip((v_kt - v_out) / max(34.0 - v_out, 1e-6), 0.0, 1.0)
    inner = h34 + (1.0 - h34) * np.clip((v_kt - 34.0) / np.maximum(vmax_kt - 34.0, 1e-6), 0.0, 1.0)
    direct = v_kt >= 34.0
    if inside_rmax is not None:
        direct = direct | inside_rmax
    return np.where(direct & (vmax_kt > 34.0), inner, tail)


def siat_class(x: np.ndarray) -> np.ndarray:
    return np.digitize(np.nan_to_num(x, nan=0.0), SIAT_EDGES)


# ============================================================================
# PUNTAJES
# ============================================================================


def contingency(pred: np.ndarray, obs: np.ndarray) -> dict:
    a = float(np.sum(pred & obs)); b = float(np.sum(pred & ~obs))
    c = float(np.sum(~pred & obs)); d = float(np.sum(~pred & ~obs))
    return scores_from_counts(a, b, c, d)


def scores_from_counts(a, b, c, d) -> dict:
    n = a + b + c + d
    pod = a / (a + c) if a + c else np.nan
    pofd = b / (b + d) if b + d else np.nan
    ar = (a + b) * (a + c) / n if n else np.nan
    out = {"a": a, "b": b, "c": c, "d": d, "POD": pod, "POFD": pofd, "PSS": pod - pofd,
           "FAR": b / (a + b) if a + b else np.nan,
           "CSI": a / (a + b + c) if a + b + c else np.nan,
           "ETS": (a - ar) / (a + b + c - ar) if (a + b + c - ar) else np.nan,
           "sesgo": (a + b) / (a + c) if a + c else np.nan}
    out["SEDI"] = sedi(pod, pofd)
    return out


def sedi(h: float, f: float) -> float:
    """Ferro y Stephenson (2011), doi:10.1175/WAF-D-10-05030.1."""

    if not (0.0 < h < 1.0 and 0.0 < f < 1.0):
        return np.nan
    num = np.log(f) - np.log(h) - np.log(1 - f) + np.log(1 - h)
    den = np.log(f) + np.log(h) + np.log(1 - f) + np.log(1 - h)
    return float(num / den)


def auc(score: np.ndarray, obs: np.ndarray) -> float:
    """Área bajo la ROC (Mann-Whitney, empates a la mitad)."""

    s = np.asarray(score, float)
    o = np.asarray(obs, bool)
    npos, nneg = o.sum(), (~o).sum()
    if npos == 0 or nneg == 0:
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[o].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def auc_by_year(score: np.ndarray, obs: np.ndarray, year: np.ndarray) -> float:
    vals = [auc(score[year == y], obs[year == y]) for y in np.unique(year)]
    return float(np.nanmean(vals))


def roc_curve(score: np.ndarray, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(POFD, POD) para alerta si score >= umbral, en todos los umbrales distintos."""

    s = np.asarray(score, float)
    o = np.asarray(obs, bool)
    order = np.argsort(-s, kind="mergesort")
    s, o = s[order], o[order]
    last = np.r_[np.nonzero(np.diff(s))[0], s.size - 1]
    tp, fp = np.cumsum(o)[last], np.cumsum(~o)[last]
    pod = np.r_[0.0, tp / max(o.sum(), 1)]
    pofd = np.r_[0.0, fp / max((~o).sum(), 1)]
    return pofd, pod


def pod_at_pofd(score_ref: np.ndarray, obs: np.ndarray, pofd: float) -> float:
    """POD de la referencia a igual POFD (ROC interpolada = umbral aleatorizado)."""

    f, h = roc_curve(score_ref, obs)
    return float(np.interp(pofd, f, h))


def bootstrap_counts(pred: np.ndarray, obs: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Tabla de contingencia por grupo (ciclón): (ngrupos, 4) = a, b, c, d."""

    g = pd.factorize(groups)[0]
    ng = g.max() + 1
    out = np.zeros((ng, 4))
    for j, m in enumerate((pred & obs, pred & ~obs, ~pred & obs, ~pred & ~obs)):
        out[:, j] = np.bincount(g, weights=m.astype(float), minlength=ng)
    return out


def bootstrap_weights(n_groups: int, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_groups, size=(n_boot, n_groups))
    w = np.zeros((n_boot, n_groups))
    for b in range(n_boot):
        w[b] = np.bincount(idx[b], minlength=n_groups)
    return w


def pss_from_counts(counts: np.ndarray) -> np.ndarray:
    a, b, c, d = counts.T
    with np.errstate(invalid="ignore", divide="ignore"):
        return a / (a + c) - b / (b + d)


# ============================================================================
# DATOS
# ============================================================================


def percentiles_at(cfg: ConfigB1, d: int, lat: np.ndarray, lon: np.ndarray) -> dict:
    """Percentiles de duración d en los píxeles (lat, lon) más cercanos."""

    import xarray as xr

    pts = pd.DataFrame({"lat": np.round(lat, 4), "lon": np.round(lon, 4)})
    uniq = pts.drop_duplicates().reset_index(drop=True)
    with xr.open_dataset(cfg.clim_pattern.format(d=d)) as ds:
        q = ds[cfg.clim_var].sel(lat=xr.DataArray(uniq["lat"].to_numpy(), dims="p"),
                                 lon=xr.DataArray(uniq["lon"].to_numpy(), dims="p"),
                                 method="nearest").load()
    pos = pts.merge(uniq.reset_index(), on=["lat", "lon"], how="left")["index"].to_numpy()
    return {k: v.values[pos] for k, v in quantile_dataarray_to_dict(q).items()}


def load_rows(cfg: ConfigB1) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Pares (con etiqueta) y renglones donde se evalúa el índice."""

    pares = pd.read_csv(os.path.join(cfg.diag_dir, cfg.pares_file), dtype={"CVEGEO": str},
                        parse_dates=["t_pico"])
    pares = pares[pares["nivel_pico"] >= 0].reset_index(drop=True)
    pares["pair"] = np.arange(len(pares))
    pares["year"] = pares["t_pico"].dt.year

    f_pasos = os.path.join(cfg.diag_dir, cfg.pasos_file)
    modo = cfg.modo
    if modo == "auto":
        modo = "pasos" if os.path.exists(f_pasos) else "pico"

    if modo == "pasos":
        rows = pd.read_csv(f_pasos, dtype={"CVEGEO": str})
        rows = rows.merge(pares[["CVEGEO", "CT", "pair"]], on=["CVEGEO", "CT"], how="inner")
        rows = rows.rename(columns={"P6_max": "P6", "R24_en_pico6": "R24", "A72_en_pico6": "A72",
                                    "dist_km": "dist", "vmax_kt": "vmax", "lat_ct": "lat_w"})
    else:
        rows = pares.rename(columns={"P_evento": "P6", "R24_en_pico6": "R24", "A72_en_pico6": "A72",
                                     "dist_pico_km": "dist", "vmax_pico_kt": "vmax"})
        rows["lat_w"] = rows["lat_pmax"]
    cols = ["pair", "P6", "R24", "A72", "dist", "vmax", "lat_w", "lat_pmax", "lon_pmax"]
    return pares, rows[cols].reset_index(drop=True), modo


# ============================================================================
# EJECUCIÓN
# ============================================================================


def md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Tabla markdown sin dependencias (pandas.to_markdown requiere tabulate)."""

    t = df.reset_index() if index else df
    fmt = [[("" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v))) for v in r]
           for r in t.itertuples(index=False)]
    head = "| " + " | ".join(str(c) for c in t.columns) + " |"
    sep = "|" + "|".join("---" for _ in t.columns) + "|"
    return "\n".join([head, sep] + ["| " + " | ".join(r) + " |" for r in fmt])


def evaluate(name_p, name_w, itchi_pair, pares, refs, w_boot):
    """
    Puntajes de un esquema y diferencias contra cada referencia de ``refs``
    ({nombre: (puntaje continuo, conteos por ciclón por clase)}). La primera
    referencia da las columnas sin sufijo; las demás llevan " vs <nombre>".
    """
    d = pares["D"].to_numpy(bool)
    year = pares["year"].to_numpy()
    cl = siat_class(itchi_pair)
    row = {"lluvia": name_p, "viento": name_w,
           "AUC_anual": auc_by_year(itchi_pair, d, year),
           "% pares con H_W > 0": np.nan}
    for k in range(1, 6):
        s = contingency(cl >= k, d)
        row[f"PSS>={k}"] = s["PSS"]
        if k in (3, 5):
            for m in ("POD", "POFD", "FAR", "CSI", "SEDI", "sesgo"):
                row[f"{m}>={k}"] = s[m]
    groups = pares["CT"].to_numpy()
    for i, (tag, (ref_score, counts_ref)) in enumerate(refs.items()):
        suf = "" if i == 0 else f" vs {tag}"
        for k in (3, 5):
            row[f"dPOD@POFD>={k}{suf}"] = row[f"POD>={k}"] - pod_at_pofd(ref_score, d, row[f"POFD>={k}"])
        for k in (1, 3, 5):
            cnt = bootstrap_counts(cl >= k, d, groups)
            diff = pss_from_counts(w_boot @ cnt) - pss_from_counts(w_boot @ counts_ref[k])
            lo, hi = np.nanpercentile(diff, 5), np.nanpercentile(diff, 95)
            row[f"dPSS>={k}{suf}"] = row[f"PSS>={k}"] - float(pss_from_counts(counts_ref[k].sum(0)[None])[0])
            row[f"dPSS>={k}{suf} lo"], row[f"dPSS>={k}{suf} hi"] = lo, hi
            row[f"dPSS>={k}{suf} IC90"] = f"[{lo:+.3f}, {hi:+.3f}]"
    return row


def run(cfg: ConfigB1 = ConfigB1()) -> dict:

    os.makedirs(cfg.out_dir, exist_ok=True)
    pares, rows, modo = load_rows(cfg)
    d = pares["D"].to_numpy(bool)
    print(f"pares: {len(pares):,} | positivos: {d.sum():,} ({100 * d.mean():.1f} %) | "
          f"renglones evaluados: {len(rows):,} (modo {modo})")

    lat, lon = rows["lat_pmax"].to_numpy(float), rows["lon_pmax"].to_numpy(float)
    q = {dd: percentiles_at(cfg, dd, lat, lon) for dd in (6, 24, 72)}
    p6, r24, a72 = (rows[c].to_numpy(float) for c in ("P6", "R24", "A72"))
    pair = rows["pair"].to_numpy()

    def per_pair_max(values: np.ndarray) -> np.ndarray:
        out = np.full(len(pares), np.nan)
        np.fmax.at(out, pair, values)
        return np.nan_to_num(out, nan=0.0)

    # Lluvia: esquemas fijos
    hp = {}
    for name, spec in precip_schemes().items():
        hp[name] = np.nan_to_num(compute_precipitation_hazard_from_spec(
            p6, q[6], {"R24": r24, "A72": a72}, {"R24": q[24], "A72": q[72]}, spec), nan=0.0)

    # Evidencia por celda (en el pico de P6 de cada par)
    pk = pares
    qpk = {dd: percentiles_at(cfg, dd, pk["lat_pmax"].to_numpy(float), pk["lon_pmax"].to_numpy(float))
           for dd in (6, 24, 72)}
    cells = class_cells(pk["P_evento"].to_numpy(float), pk["R24_en_pico6"].to_numpy(float),
                        pk["A72_en_pico6"].to_numpy(float), qpk[6], qpk[24], qpk[72])
    lab = np.array(["<P75", "P75-P90", "P90-P95", ">=P95"])
    tab = (pd.DataFrame({"P6": lab[cells[:, 0].clip(0)], "R24": lab[cells[:, 1]],
                         "A72": lab[cells[:, 2]], "D": d})[cells[:, 0] >= 0]
           .groupby(["P6", "R24", "A72"], sort=False)["D"].agg(n="size", declaratorias="sum"))
    tab["P(D) %"] = 100 * tab["declaratorias"] / tab["n"]
    tab = tab.reindex(pd.MultiIndex.from_product([lab, lab, lab], names=["P6", "R24", "A72"]))
    tab.to_csv(os.path.join(cfg.out_dir, "celdas_P6_R24_A72.csv"), float_format="%.3f")

    # M3: validación cruzada dejando un año fuera (ajuste en el pico, aplicación en los renglones)
    years = pares["year"].to_numpy()
    row_year = years[pair]
    m3 = np.zeros(len(rows))
    for y in np.unique(years):
        table = fit_m3_table(cells[years != y], d[years != y], cfg.m3_pseudo)
        sel = row_year == y
        m3[sel] = np.nan_to_num(compute_precipitation_hazard_from_spec(
            p6[sel], {k: v[sel] for k, v in q[6].items()},
            {"R24": r24[sel], "A72": a72[sel]},
            {"R24": {k: v[sel] for k, v in q[24].items()}, "A72": {k: v[sel] for k, v in q[72].items()}},
            m3_spec(table)), nan=0.0)
    hp["M3 (VC por año)"] = m3
    final = m3_spec(fit_m3_table(cells, d, cfg.m3_pseudo))
    with open(os.path.join(cfg.out_dir, "tabla_m3.yaml"), "w", encoding="utf-8") as fh:
        yaml.safe_dump({"precipitation": {"hazard_model": "class_table", "class_table": {
            "axes": {k: list(v) for k, v in final["axes"].items()}, "table": final["table"],
            "gate": final["gate"]}}}, fh, allow_unicode=True, sort_keys=False)

    # Viento
    dist = rows["dist"].to_numpy(float)
    v_kt, rmax = wind_kt_w06(rows["vmax"].to_numpy(float), rows["lat_w"].to_numpy(float), dist,
                             return_rmax=True)
    vmax = rows["vmax"].to_numpy(float)
    hw = {name: wind_hazard(v_kt, vmax, cfg.h34, v_out, inside_rmax=dist <= rmax)
          for name, v_out in cfg.viento.items()}

    # Referencia (v0.2, W0) y bootstrap por ciclón
    ref_p, ref_w = cfg.referencia
    ref_itchi = per_pair_max(1 - (1 - hw[ref_w]) * (1 - hp[ref_p]))
    groups = pares["CT"].to_numpy()
    w_boot = bootstrap_weights(pd.factorize(groups)[0].max() + 1, cfg.n_boot, cfg.seed)
    cl_ref = siat_class(ref_itchi)
    counts_ref = {k: bootstrap_counts(cl_ref >= k, d, groups) for k in (1, 3, 5)}
    def ref_entry(score: np.ndarray) -> tuple:
        cl = siat_class(score)
        return score, {k: bootstrap_counts(cl >= k, d, groups) for k in (1, 3, 5)}

    por_par = pares[["CVEGEO", "CT", "year", "D"]].copy()
    hp_pair = {name: per_pair_max(h) for name, h in hp.items()}
    refs_lluvia = {ref_p: ref_entry(hp_pair[ref_p]), "M0": ref_entry(hp_pair["M0"])}

    # Solo lluvia: contra v0.2 y contra M0 (aporte de R24 y A72)
    lluvia = []
    for name, hpair in hp_pair.items():
        lluvia.append(evaluate(name, "sin viento", hpair, pares, refs_lluvia, w_boot))
        por_par[f"HP|{name}"] = hpair.astype("float32")
    lluvia = pd.DataFrame(lluvia).drop(columns=["viento", "% pares con H_W > 0"])
    lluvia.to_csv(os.path.join(cfg.out_dir, "lluvia_candidatos.csv"), index=False, float_format="%.4f")

    # Lluvia x viento
    main = ["v0.2", "M0", "M1", "M2", "M3 (VC por año)"]
    diseno = []
    for name_p in main:
        for name_w, h_w in hw.items():
            itchi_pair = per_pair_max(1 - (1 - h_w) * (1 - hp[name_p]))
            r = evaluate(name_p, name_w, itchi_pair, pares, {"ref": (ref_itchi, counts_ref)}, w_boot)
            r["% pares con H_W > 0"] = 100 * float(np.mean(per_pair_max(h_w) > 0))
            diseno.append(r)
            por_par[f"ITCHI|{name_p}|{name_w}"] = itchi_pair.astype("float32")
    diseno = pd.DataFrame(diseno)
    diseno.to_csv(os.path.join(cfg.out_dir, "diseno_2x2.csv"), index=False, float_format="%.4f")

    por_par.to_csv(os.path.join(cfg.out_dir, "por_par.csv.gz"), index=False, float_format="%.4f")
    write_summary(cfg, pares, modo, tab, lluvia, diseno)
    print(f"salidas en {cfg.out_dir}")
    return {"pares": pares, "celdas": tab, "lluvia": lluvia, "diseno": diseno, "tabla_m3": final,
            "por_par": por_par, "modo": modo}


def write_summary(cfg, pares, modo, tab, lluvia, diseno) -> None:
    d = pares["D"].to_numpy(bool)
    cols_l = ["lluvia", "AUC_anual", "PSS>=1", "PSS>=3", "PSS>=5", "POD>=5", "POFD>=5", "sesgo>=5",
              "SEDI>=5", "dPSS>=5", "dPSS>=5 IC90", "dPOD@POFD>=5", "dPSS>=5 vs M0", "dPSS>=5 vs M0 IC90"]
    cols_d = ["lluvia", "viento", "AUC_anual", "% pares con H_W > 0", "PSS>=1", "PSS>=3", "PSS>=5",
              "dPSS>=1", "dPSS>=1 IC90", "dPSS>=5", "dPSS>=5 IC90", "dPOD@POFD>=5", "sesgo>=5"]
    two = diseno[diseno["lluvia"].isin([cfg.referencia[0], "M1"])
                 & diseno["viento"].isin([cfg.referencia[1], "W1 corte 12 m/s"])]
    lab = ["<P75", "P75-P90", "P90-P95", ">=P95"]
    cell = tab.apply(lambda r: "" if pd.isna(r["n"]) else f"{r['P(D) %']:.1f} ({int(r['n']):,})", axis=1)
    pivot = cell.unstack("A72").reindex(
        index=pd.MultiIndex.from_product([lab, lab], names=["P6", "R24"]), columns=lab)
    lines = [
        "# Experimento del Bloque 1 (ITCHI v0.2.1)", "",
        f"Pares: {len(pares):,}; con declaratoria: {d.sum():,} ({100 * d.mean():.1f} %). "
        f"Modo: {modo}. Bootstrap por ciclón: {cfg.n_boot} réplicas; intervalos de 90 %. "
        f"Referencia: {cfg.referencia[0]} con {cfg.referencia[1]}.", "",
        "## 1. P(D) por celda de clases en el pico de P6 del par: % (n pares). "
        "Filas P6 x R24, columnas A72; cada variable contra sus propios percentiles", "",
        md_table(pivot), "",
        "## 2. Solo lluvia: candidatos", "",
        md_table(lluvia[cols_l], index=False), "",
        "## 3. Diseño 2x2 (v0.2 / M1 x cola hasta R_W06 / corte 12 m/s)", "",
        md_table(two[cols_d], index=False), "",
        "## 4. Lluvia x viento, todos los esquemas", "",
        md_table(diseno[cols_d], index=False), "",
        "dPOD@POFD: POD del esquema menos POD de la referencia a la misma POFD (ROC interpolada). "
        "Cercano a 0: el cambio solo desplaza el umbral sobre la misma curva ROC.",
    ]
    if modo == "pico":
        lines += ["", "Aviso: modo pico. El índice se evaluó en el paso del pico de P6; los esquemas "
                  "cuyo máximo cae en otro paso (escalada por R24) quedan subestimados. "
                  "Correr negativos_w06.py --guardar-pasos."]
    with open(os.path.join(cfg.out_dir, "resumen_b1.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Experimento del Bloque 1 de ITCHI v0.2.1")
    ap.add_argument("--modo", choices=("auto", "pasos", "pico"), default="auto")
    ap.add_argument("--n-boot", type=int, default=ConfigB1.n_boot)
    a = ap.parse_args()
    run(ConfigB1(modo=a.modo, n_boot=a.n_boot))
