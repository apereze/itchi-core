"""
ITCHI-CORE · Verificación de niveles de lluvia con positivos y negativos
(pares municipio–CT dentro del radio W06)

Universo: pares (municipio, CT) en los que el punto representativo del municipio
queda dentro del radio exterior W06 con metodología PA21, R(V = 2 m s-1), en al
menos un paso sinóptico. Es el mismo radio que usa ITCHI ("w06", modo "pa21"
de itchi_size_helpers): Rmax y forma por regresión, sin calibrar a R34.

    vmax: MWS de ROCLOUD (km/h) / 1.852 -> kt
    R_W06 = radius_of_wind(W06(vmax, lat; rmax_variant="pa21"), 2 m s-1)

Etiqueta: D = 1 si hubo al menos una declaratoria del municipio asociada a ese CT
(CT más cercano en los pasos de declaratorias_ct_pasos.csv); D = 0 si no.
Se guarda además el tipo (emergencia, desastre, contingencia).

Lluvia por par, con la misma ventana para positivos y negativos: los pasos en que
el municipio está dentro de R_W06. En cada paso se toma el píxel del máximo de 6 h
del municipio; el pico del par es el paso con mayor P6. En ese píxel y paso:
clase de P6, R24, R72 y A72 (antecedente exclusivo) con sus percentiles. Niveles
(funciones de declaratorias_6h.py):

    base      rampa solo 6 h (P75/P90/P95/P99 -> 1..4)
    A         máximo de clases P6, R24, R72 convertido a nivel con la rampa base
    A_cap     igual, con la clase de R72 topada en P95 (P99 de 72 h no es robusto)
    B         matriz clase P6 x grupo A72 (Config.matriz_A72)

Verificación por esquema y umbral de nivel k (alerta si nivel >= k), para todos
los positivos y por tipo: POD, FAR, POFD, Peirce (PSS = POD - POFD), CSI, sesgo.

Entradas: las de declaratorias_6h.py y sus salidas (declaratorias_ct_pasos.csv,
declaratorias_ct_eventos.csv), más los percentiles de 6, 24 y 72 h.

Salidas (outputs/diagnostics/declaratorias_6h/):
    pares_municipio_ct_w06.csv.gz       un renglón por par, con etiqueta y niveles
    verificacion_niveles_w06.csv        tablas de contingencia y puntajes
    positivos_fuera_w06.csv             pares con declaratoria que nunca entran a R_W06

Ejecutar desde scripts/ (después de declaratorias_6h.py):
    python negativos_w06.py
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

import declaratorias_6h as d6
from itchi.parametric_radii import compute_parametric_radii

KMH_PER_KT = 1.852
TIPOS = d6.TIPOS


@dataclass
class ConfigW06(d6.Config):

    v_outer_ms: float = 2.0          # PA21
    rmax_variant: str = "pa21"       # como H.augment_track
    months: tuple = (5, 6, 7, 8, 9, 10, 11)


# ============================================================================
# UNIVERSO DE PARES
# ============================================================================


def tracks_w06(cfg: ConfigW06) -> pd.DataFrame:
    """Posiciones ROCLOUD con vmax en kt y radio exterior W06 (PA21) en km."""

    tr = d6.load_tracks(cfg)
    tr = tr[tr["time"].dt.year.between(cfg.start_year, cfg.end_year)
            & tr["time"].dt.month.isin(cfg.months)].copy()
    tr["vmax_kt"] = pd.to_numeric(tr["MWS"], errors="coerce") / KMH_PER_KT
    rad = compute_parametric_radii(
        vmax_kt=tr["vmax_kt"].to_numpy(float), lat=tr["lat_ct"].to_numpy(float),
        thresholds_ms={"r_w06_km": cfg.v_outer_ms}, rmax_variant=cfg.rmax_variant,
    )
    tr["r_w06_km"] = rad["r_w06_km"]
    return tr.dropna(subset=["r_w06_km", "lat_ct", "lon_ct"]).reset_index(drop=True)


def pairs_inside(tracks: pd.DataFrame, mun_pts: pd.DataFrame) -> pd.DataFrame:
    """(CT, time, CVEGEO) con el punto representativo del municipio dentro de R_W06."""

    lat_m, lon_m = mun_pts["lat_mun"].to_numpy(), mun_pts["lon_mun"].to_numpy()
    out = []
    for _, g in tracks.groupby(tracks["time"].dt.year):          # por año: memoria acotada
        dist, _ = d6.haversine_bearing(g["lat_ct"].to_numpy()[:, None], g["lon_ct"].to_numpy()[:, None],
                                       lat_m[None, :], lon_m[None, :])
        i, j = np.nonzero(dist <= g["r_w06_km"].to_numpy()[:, None])
        out.append(pd.DataFrame({
            "CT": g["CT"].to_numpy()[i], "time": g["time"].to_numpy()[i],
            "CVEGEO": mun_pts["CVEGEO"].to_numpy()[j], "dist_km": dist[i, j],
            "r_w06_km": g["r_w06_km"].to_numpy()[i], "vmax_kt": g["vmax_kt"].to_numpy()[i],
        }))
    return pd.concat(out, ignore_index=True)


def labels_from_declaratorias(cfg: ConfigW06) -> pd.DataFrame:
    """(CVEGEO, CT) con declaratoria, con indicadores por tipo."""

    pasos = pd.read_csv(os.path.join(cfg.out_dir, "declaratorias_ct_pasos.csv"),
                        dtype={"CVEGEO": str}, usecols=["id_decl", "CVEGEO", "CT"])
    ev = pd.read_csv(os.path.join(cfg.out_dir, "declaratorias_ct_eventos.csv"),
                     dtype={"CVEGEO": str}, usecols=["id_decl", "Tipo Declaratoria"])
    p = pasos.drop_duplicates(["id_decl", "CT"]).merge(ev, on="id_decl", how="left")
    lab = pd.crosstab([p["CVEGEO"], p["CT"]], p["Tipo Declaratoria"]).reindex(columns=list(TIPOS), fill_value=0)
    lab = (lab > 0).astype(int)
    lab.columns = [f"D_{t.split()[0].lower()}" for t in lab.columns]
    lab["D"] = 1
    return lab.reset_index()


# ============================================================================
# LLUVIA POR MUNICIPIO Y PASO (VECTORIZADO)
# ============================================================================


def class_at(values: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Número de percentiles igualados o superados; q (nq, n). NaN en values -> -1."""

    with np.errstate(invalid="ignore"):
        c = (values[None, :] >= q).sum(axis=0)
    return np.where(np.isfinite(values), c, -1)


def municipal_peaks(V: np.ndarray, starts: np.ndarray, lengths: np.ndarray):
    """
    V (nt, Ntot): valores en los píxeles de los municipios, concatenados por municipio.
    Devuelve el máximo por municipio (nt, nmun) y la posición en V del píxel del máximo
    (Ntot si todo es NaN).
    """

    ntot = V.shape[1]
    with np.errstate(invalid="ignore"):
        mx = np.fmax.reduceat(V, starts, axis=1)
        eq = V == np.repeat(mx, lengths, axis=1)
    pos = np.minimum.reduceat(np.where(eq, np.arange(ntot)[None, :], ntot), starts, axis=1)
    return mx, pos


def rain_by_step(cfg: ConfigW06, pairs: pd.DataFrame) -> pd.DataFrame:
    """Por (CVEGEO, time) del universo: P6 máximo del municipio y valores y clases en su píxel."""

    Q = d6.open_percentiles(cfg)
    lat, lon = Q["lat"].values, Q["lon"].values
    ny, nx = lat.size, lon.size
    Q6 = Q.values.reshape(Q.sizes["quantile"], -1)
    spec = d6.series_spec(cfg)
    Qd = {d: d6.open_percentiles(cfg, cfg.percentiles_dur_pattern.format(d=d), cfg.pct_var_dur)
          .values.reshape(-1, ny * nx) for d in sorted({d for d, _ in spec.values()})}

    gdf = d6.load_municipios(cfg)
    codes = np.array(sorted(pairs["CVEGEO"].unique()))
    pix = d6.municipio_pixels(gdf, lat, lon, codes)
    codes = np.array([c for c in codes if c in pix])
    flats = [pix[c][0] for c in codes]
    lengths = np.array([f.size for f in flats])
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    allpix = np.concatenate(flats)
    col_of = pd.Series(np.arange(codes.size), index=codes)

    out = []
    keys = pairs[["CVEGEO", "time"]].drop_duplicates()
    for year, ky in tqdm(keys.groupby(keys["time"].dt.year), desc="Años", unit="año"):

        da = d6.open_6h_year(cfg, int(year))
        if da is None:
            continue
        tfile = pd.DatetimeIndex(da["time"].values)
        need = pd.DatetimeIndex(ky["time"].unique()).intersection(tfile).sort_values()
        if need.empty:
            da.close()
            continue
        if not (np.diff(tfile.values) == np.timedelta64(6, "h")).all():
            raise ValueError(f"{year}: los pasos de 6 h no son contiguos.")
        Fy = (da.sel(lat=slice(*cfg.lat_bounds), lon=slice(*cfg.lon_bounds))
                .transpose("time", "lat", "lon").load().values.reshape(len(tfile), -1))
        da.close()
        if Fy.shape[1] != ny * nx:
            raise ValueError("La malla de los acumulados de 6 h no coincide con la de los percentiles.")

        ineed = tfile.get_indexer(need)
        G = {"P6": Fy[ineed][:, allpix]}
        for d in sorted({d for d, _ in spec.values()}):
            Rd = d6.window_sum(Fy, d // 6)
            for name, (dd, ant) in spec.items():
                if dd == d:
                    rows = np.maximum(ineed - 1, 0) if ant else ineed
                    g = Rd[rows][:, allpix]
                    G[name] = np.where((ineed - 1 >= 0)[:, None], g, np.nan) if ant else g
            del Rd
        del Fy

        mx6, pos = municipal_peaks(G["P6"], starts, lengths)       # (nt, nmun)

        # Solo los (municipio, tiempo) del universo
        ti = pd.Series(np.arange(need.size), index=need)
        k = ky[ky["time"].isin(need) & ky["CVEGEO"].isin(col_of.index)]
        it = ti[k["time"]].to_numpy()
        im = col_of[k["CVEGEO"]].to_numpy()
        p = pos[it, im]
        ok = p < allpix.size
        p = np.where(ok, p, 0)
        gp = allpix[p]                                              # píxel global del máximo

        rec = {"CVEGEO": k["CVEGEO"].to_numpy(), "time": k["time"].to_numpy(),
               "P6_max": mx6[it, im], "nivel_P6": np.where(ok, class_at(mx6[it, im], Q6[:, gp]), -1)}
        for name, (d, ant) in spec.items():
            v = np.where(ok, G[name][it, p], np.nan)
            rec[f"{name}_en_pico6"] = v
            rec[f"nivel_{name}_en_pico6"] = class_at(v, Qd[d][:, gp])
            if not ant:   # máximo propio del acumulado en el municipio
                rec[f"{name}_max"] = np.fmax.reduceat(G[name], starts, axis=1)[it, im]
        rec["lat_pmax"], rec["lon_pmax"] = lat[gp // nx], lon[gp % nx]
        out.append(pd.DataFrame(rec))

    return pd.concat(out, ignore_index=True)


# ============================================================================
# PARES, NIVELES Y VERIFICACIÓN
# ============================================================================


def build_pairs(cfg: ConfigW06, steps: pd.DataFrame, rain: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Un renglón por (CVEGEO, CT): pico de 6 h del par y sus clases; etiqueta; niveles."""

    s = steps.merge(rain, on=["CVEGEO", "time"], how="inner")
    s = s[s["P6_max"].notna()]
    peak = s.loc[s.groupby(["CVEGEO", "CT"])["P6_max"].idxmax()].set_index(["CVEGEO", "CT"])
    agg = s.groupby(["CVEGEO", "CT"]).agg(
        n_pasos=("time", "size"), t_ini=("time", "min"), t_fin=("time", "max"),
        dist_min_km=("dist_km", "min"), vmax_max_kt=("vmax_kt", "max"),
        R24_par=("R24_max", "max"), R72_par=("R72_max", "max"),
    )
    keep = ["time", "dist_km", "r_w06_km", "vmax_kt", "P6_max", "nivel_P6", "lat_pmax", "lon_pmax"] + \
           [c for c in peak.columns if c.endswith("_en_pico6")]
    pr = agg.join(peak[keep].rename(columns={"time": "t_pico", "P6_max": "P_evento", "nivel_P6": "nivel_pico",
                                             "dist_km": "dist_pico_km", "vmax_kt": "vmax_pico_kt"}))
    pr = pr.reset_index().merge(labels, on=["CVEGEO", "CT"], how="left")
    dcols = ["D"] + [c for c in labels.columns if c.startswith("D_")]
    pr[dcols] = pr[dcols].fillna(0).astype(int)

    # Niveles: rampa base, opción B y opción A (clase -> nivel con la columna 0 de la matriz)
    spec = d6.series_spec(cfg)
    labs = d6.pct_labels(xr.open_dataset(cfg.percentiles_file)["quantile"].values)
    pr = d6.combined_levels(pr, spec, labs, {24: labs, 72: labs}, cfg)
    M = np.asarray(cfg.matriz_A72)
    k95 = labs.index("P95") + 1
    for name, c72 in (("A", pr["nivel_R72_en_pico6"]), ("A_cap", pr["nivel_R72_en_pico6"].clip(upper=k95))):
        cl = pd.concat([pr["nivel_pico"], pr["nivel_R24_en_pico6"], c72], axis=1).where(lambda x: x >= 0)
        cmax = cl.max(axis=1)
        pr[f"nivel_{name}_pico"] = np.where(cmax.notna(), M[cmax.fillna(0).astype(int), 0], np.nan)
    return pr


def scores(pred: np.ndarray, obs: np.ndarray) -> dict:
    a = int(np.sum(pred & obs)); b = int(np.sum(pred & ~obs))
    c = int(np.sum(~pred & obs)); n = int(np.sum(~pred & ~obs))
    pod = a / (a + c) if a + c else np.nan
    pofd = b / (b + n) if b + n else np.nan
    return {"aciertos": a, "falsas": b, "fallos": c, "neg_correctos": n,
            "POD": pod, "FAR": b / (a + b) if a + b else np.nan, "POFD": pofd,
            "PSS": pod - pofd, "CSI": a / (a + b + c) if a + b + c else np.nan,
            "sesgo": (a + b) / (a + c) if a + c else np.nan}


def verify(pr: pd.DataFrame) -> pd.DataFrame:
    """Alerta si nivel >= k, k = 1..4, por esquema y por definición de evento."""

    schemes = {"base": "nivel_base_pico", "A": "nivel_A_pico", "A_cap": "nivel_A_cap_pico",
               "B": "nivel_matriz_pico"}
    targets = {"cualquier declaratoria": pr["D"] == 1}
    for t in TIPOS:
        col = f"D_{t.split()[0].lower()}"
        if col in pr:
            # positivos de un tipo; los pares con otros tipos se excluyen, no cuentan como negativos
            targets[t] = pr[col] == 1
    rows = []
    for tname, obs in targets.items():
        sub = (pr["D"] == 0) | obs
        for sname, col in schemes.items():
            v = pr.loc[sub, col]
            okv = v.notna()
            for k in range(1, 5):
                rows.append({"evento": tname, "esquema": sname, "nivel >=": k,
                             **scores((v[okv] >= k).to_numpy(), obs[sub][okv].to_numpy())})
    return pd.DataFrame(rows)


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================


def run(cfg: ConfigW06 = ConfigW06()) -> dict:

    t0 = time.time()
    print("\n" + "=" * 75)
    print("ITCHI-CORE · Verificación de niveles con pares municipio–CT dentro de R_W06")
    print("=" * 75)

    print("\n[1/4] Trayectorias, radio W06 (PA21) y universo de pares")
    tr = tracks_w06(cfg)
    gdf = d6.load_municipios(cfg)
    steps = pairs_inside(tr, d6.municipio_points(gdf))
    print(f"posiciones: {len(tr):,} | R_W06 mediana {tr['r_w06_km'].median():.0f} km "
          f"(P10 {tr['r_w06_km'].quantile(0.1):.0f}, P90 {tr['r_w06_km'].quantile(0.9):.0f}) | "
          f"pasos municipio–CT: {len(steps):,} | pares: {steps[['CVEGEO', 'CT']].drop_duplicates().shape[0]:,}")

    print("\n[2/4] Etiquetas desde declaratorias")
    labels = labels_from_declaratorias(cfg)
    universe = steps[["CVEGEO", "CT"]].drop_duplicates()
    fuera = labels.merge(universe, on=["CVEGEO", "CT"], how="left", indicator=True)
    fuera = fuera[fuera["_merge"] == "left_only"].drop(columns="_merge")
    print(f"pares con declaratoria: {len(labels):,} | fuera de R_W06 en todos sus pasos: {len(fuera):,} "
          f"({100 * len(fuera) / max(len(labels), 1):.1f} %)")

    print("\n[3/4] Lluvia por municipio y paso (6, 24, 72 h)")
    rain = rain_by_step(cfg, steps)

    print("\n[4/4] Niveles y verificación")
    pr = build_pairs(cfg, steps, rain, labels)
    ver = verify(pr)

    pr.to_csv(os.path.join(cfg.out_dir, "pares_municipio_ct_w06.csv.gz"), index=False, float_format="%.3f")
    ver.to_csv(os.path.join(cfg.out_dir, "verificacion_niveles_w06.csv"), index=False, float_format="%.4f")
    fuera.to_csv(os.path.join(cfg.out_dir, "positivos_fuera_w06.csv"), index=False)

    print(f"\npares con lluvia: {len(pr):,} | positivos: {int(pr['D'].sum()):,} "
          f"({100 * pr['D'].mean():.1f} %)")
    print("\nDistribución de niveles (% de pares) por etiqueta:")
    for col in ("nivel_base_pico", "nivel_matriz_pico", "nivel_A_cap_pico"):
        print(f"\n{col}")
        print((100 * pd.crosstab(pr["D"], pr[col], normalize="index")).round(1).to_string())
    print("\nVerificación, cualquier declaratoria:")
    print(ver[ver["evento"] == "cualquier declaratoria"]
          .drop(columns="evento").round(3).to_string(index=False))
    print("\n" + "=" * 75)
    print(f"PROCESO TERMINADO · {(time.time() - t0) / 60:.1f} minutos · {cfg.out_dir}")
    print("=" * 75)
    return {"pares": pr, "verificacion": ver, "fuera": fuera}


if __name__ == "__main__":
    out = run(ConfigW06())
