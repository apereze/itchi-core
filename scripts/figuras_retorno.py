"""
ITCHI-CORE · Figuras de los periodos de retorno (notebooks/third test/10_periodos_retorno.ipynb)

    fig_r1_curva_pixel     cómo se lee un periodo de retorno: máximos estacionales,
                           GEV y POT en ciudades de referencia
    fig_r2_mapas_niveles   lluvia con periodo de retorno de 2 y 10 años por duración
    fig_r3_rareza_percentil cada cuánto se alcanza el P95 y el P99 de cada píxel
    fig_r4_consistencia    GEV contra POT en T = 10 años (deben coincidir)
    fig_r5_forma           parámetro de forma y píxeles con forma acotada (control)

Las funciones reciben un RP: arreglos NumPy con las mismas dimensiones que el
archivo de scripts/periodos_retorno.py (load_rp lo lee; from_result lo arma
desde compute() sin escribir el archivo).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

try:
    import itchi  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itchi.return_periods import (
    annual_return_period,
    gev_return_level,
    pot_recurrence_interval,
    pot_return_level,
    recurrence_from_annual,
)

CIUDADES = {
    "Acapulco": (16.85, -99.88), "Villahermosa": (17.99, -92.93), "Mérida": (20.97, -89.62),
    "Monterrey": (25.67, -100.31), "La Paz": (24.14, -110.31), "Ciudad de México": (19.43, -99.13),
}
MEXICO_BOX = dict(lat=(14.0, 33.5), lon=(-118.5, -86.0))
plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


@dataclass
class RP:
    lat: np.ndarray
    lon: np.ndarray
    durations: list
    T: np.ndarray
    R: np.ndarray
    seasons: np.ndarray
    data: dict

    def get(self, var: str, duration: int) -> np.ndarray:
        return self.data[var][self.durations.index(duration)]

    def pixel(self, lat: float, lon: float) -> tuple[int, int]:
        return int(np.abs(self.lat - lat).argmin()), int(np.abs(self.lon - lon).argmin())

    def box(self, lat=MEXICO_BOX["lat"], lon=MEXICO_BOX["lon"]) -> tuple[slice, slice]:
        jj = np.nonzero((self.lat >= lat[0]) & (self.lat <= lat[1]))[0]
        ii = np.nonzero((self.lon >= lon[0]) & (self.lon <= lon[1]))[0]
        return slice(jj.min(), jj.max() + 1), slice(ii.min(), ii.max() + 1)


VARS = ("seasonal_max", "n_seasons", "gev_xi", "gev_sigma", "gev_mu", "gev_xi_clipped", "gpd_u",
        "gpd_sigma", "gpd_xi", "gpd_rate", "gpd_xi_clipped", "rl_gev", "rl_pot", "rl_pot_emp")


def load_rp(path) -> RP:
    import xarray as xr

    with xr.open_dataset(path) as ds:
        lat = ds["lat"].values
        order = np.argsort(lat)
        data = {v: ds[v].values[..., order, :] for v in VARS if v in ds}
        if "peaks" in ds:
            data["peaks"] = ds["peaks"].values[..., order, :]
        return RP(lat[order], ds["lon"].values, [int(d) for d in ds["duration"].values],
                  ds["T"].values, ds["R"].values, ds["season"].values, data)


def from_result(res: dict, cfg) -> RP:
    st, durs = res["state"], list(cfg.durations)
    f = res["fits"]
    data = {"seasonal_max": np.stack([st["am"][d] for d in durs]),
            "peaks": np.stack([st["top"][d] for d in durs]),
            "n_seasons": np.stack([f[d]["n_seasons"] for d in durs])}
    for fam, keys in (("gev", ("xi", "sigma", "mu", "xi_clipped")), ("gpd", ("u", "sigma", "xi", "rate", "xi_clipped"))):
        for k in keys:
            data[f"{fam}_{k}"] = np.stack([f[d][fam][k] for d in durs]).astype(float)
    for k in ("rl_gev", "rl_pot", "rl_pot_emp"):
        data[k] = np.stack([f[d][k] for d in durs])
    lat = np.asarray(st["lat"])
    order = np.argsort(lat)
    data = {k: v[..., order, :] for k, v in data.items()}
    return RP(lat[order], np.asarray(st["lon"]), durs, np.asarray(cfg.T_gev), np.asarray(cfg.R_pot),
              np.asarray(res["years"]), data)


# ============================================================================
# MAPAS
# ============================================================================


def _map_axes(nrows: int, ncols: int, figsize):
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                                 subplot_kw={"projection": ccrs.PlateCarree()})
        for ax in axes.flat:
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.6)
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.4, edgecolor="0.3")
        kw = {"transform": ccrs.PlateCarree()}
    except ImportError:
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        kw = {}
    return fig, axes, kw


def _draw(ax, rp: RP, field: np.ndarray, kw: dict, box=True, **pkw):
    js, is_ = rp.box() if box else (slice(None), slice(None))
    m = ax.pcolormesh(rp.lon[is_], rp.lat[js], field[js, is_], shading="auto", **kw, **pkw)
    for name, (la, lo) in CIUDADES.items():
        ax.plot(lo, la, "k.", ms=3, **kw)
    ax.set_xlim(rp.lon[is_][[0, -1]])
    ax.set_ylim(rp.lat[js][[0, -1]])
    return m


def fig_r2_mapas_niveles(rp: RP, T_show=(2.0, 10.0)):
    """Lluvia (mm) con periodo de retorno T, por duración (GEV)."""

    durs = rp.durations
    fig, axes, kw = _map_axes(len(T_show), len(durs), (5.2 * len(durs), 3.6 * len(T_show)))
    for j, d in enumerate(durs):
        rl = rp.get("rl_gev", d)
        vmax = np.nanpercentile(rl[list(rp.T).index(max(T_show))][rp.box()], 98)
        for i, T in enumerate(T_show):
            m = _draw(axes[i, j], rp, rl[list(rp.T).index(T)], kw, cmap="YlGnBu", vmin=0, vmax=vmax)
            axes[i, j].set_title(f"R{d}: lluvia de periodo de retorno {T:g} años", fontsize=10)
        fig.colorbar(m, ax=axes[:, j], shrink=0.8, label=f"mm en {d} h")
    fig.suptitle("Lluvia que se supera en promedio una vez cada T años (máximos de mayo a noviembre, GEV)",
                 y=1.0)
    return fig


def fig_r3_rareza_percentil(rp: RP, percentiles: dict, labels=("Q95", "Q99")):
    """
    Intervalo de recurrencia (años) de los percentiles actuales de cada píxel.

    percentiles: {duración: {"Q95": campo (lat, lon), ...}} en la malla de rp.
    """

    bounds = [0.0, 0.34, 0.5, 1.0, 2.0, 5.0, 10.0, 1e9]
    names = ["< 0.33 (≥ 3 por temporada)", "0.33–0.5", "0.5–1", "1–2", "2–5", "5–10", "> 10 años"]
    cmap = ListedColormap(["#F7FBFF", "#C6DBEF", "#9ECAE1", "#FDD49E", "#FC8D59", "#D7301F", "#7F0000"])
    norm = BoundaryNorm(bounds, cmap.N)
    durs = [d for d in rp.durations if d in percentiles]
    fig, axes, kw = _map_axes(len(labels), len(durs), (5.2 * len(durs), 3.6 * len(labels)))
    stats = {}
    for j, d in enumerate(durs):
        u, s, xi, rate = (rp.get(f"gpd_{k}", d) for k in ("u", "sigma", "xi", "rate"))
        for i, q in enumerate(labels):
            x = percentiles[d][q]
            Rq = pot_recurrence_interval(x, u, s, xi, rate)
            Rq = np.where(np.isfinite(x) & np.isfinite(u) & (x < u), 0.1, Rq)   # más frecuente que el umbral
            m = _draw(axes[i, j], rp, Rq, kw, cmap=cmap, norm=norm)
            axes[i, j].set_title(f"{q.replace('Q', 'P')} de {d} h: ¿cada cuánto se alcanza?", fontsize=10)
            box = Rq[rp.box()]
            stats[(d, q)] = np.nanmedian(box)
    cb = fig.colorbar(m, ax=axes, shrink=0.8, ticks=[(a + b) / 2 for a, b in zip(bounds[:-2], bounds[1:-1])] + [11])
    cb.ax.set_yticklabels(names)
    cb.set_label("intervalo medio entre eventos (años)")
    fig.suptitle("Rareza real de los percentiles que usa ITCHI v0.2 (mediana en México: "
                 + ", ".join(f"{q.replace('Q', 'P')}-{d} h {v:.2f} años" for (d, q), v in stats.items()) + ")",
                 y=1.0, fontsize=10)
    return fig, stats


def fig_r4_consistencia(rp: RP, T: float = 10.0):
    """GEV contra POT para el mismo periodo de retorno anual T."""

    durs = rp.durations
    fig, axes = plt.subplots(1, len(durs), figsize=(4.8 * len(durs), 4.2), squeeze=False)
    Rt = float(recurrence_from_annual(T))
    for ax, d in zip(axes[0], durs, strict=True):
        g = gev_return_level(T, rp.get("gev_xi", d), rp.get("gev_sigma", d), rp.get("gev_mu", d))[rp.box()]
        p = pot_return_level(Rt, *(rp.get(f"gpd_{k}", d) for k in ("u", "sigma", "xi", "rate")))[rp.box()]
        ok = np.isfinite(g) & np.isfinite(p)
        ax.hexbin(g[ok], p[ok], gridsize=60, bins="log", cmap="Greys", mincnt=1)
        hi = np.nanpercentile(np.r_[g[ok], p[ok]], 99.5) if ok.any() else 1.0
        ax.plot([0, hi], [0, hi], "r-", lw=1)
        rel = np.nanmedian(np.abs(p[ok] - g[ok]) / g[ok]) if ok.any() else np.nan
        ax.set(xlim=(0, hi), ylim=(0, hi), xlabel=f"GEV (máximos estacionales), mm en {d} h",
               ylabel="POT (picos independientes), mm", title=f"R{d}, T = {T:g} años · diferencia mediana {rel:.0%}")
    fig.suptitle("Dos estimadores independientes del mismo nivel: si coinciden, el resultado es robusto", y=1.03)
    fig.tight_layout()
    return fig


def fig_r5_forma(rp: RP):
    """Parámetro de forma (GEV y GPD) y fracción de píxeles con forma acotada."""

    durs = rp.durations
    fig, axes, kw = _map_axes(2, len(durs), (5.2 * len(durs), 7.2))
    for j, d in enumerate(durs):
        for i, fam in enumerate(("gev", "gpd")):
            xi = rp.get(f"{fam}_xi", d)
            m = _draw(axes[i, j], rp, xi, kw, cmap="RdBu_r", vmin=-0.5, vmax=0.5)
            clip = np.nanmean(rp.get(f"{fam}_xi_clipped", d)[rp.box()])
            axes[i, j].set_title(f"{fam.upper()} ξ, R{d} · acotada en {clip:.1%} de los píxeles", fontsize=10)
    fig.colorbar(m, ax=axes, shrink=0.8, label="ξ (> 0: cola pesada)")
    fig.suptitle("Control: forma de la cola. Con 26 temporadas ξ es muy incierto por píxel", y=1.0)
    return fig


# ============================================================================
# CURVA EN UN PÍXEL
# ============================================================================


def fig_r1_curva_pixel(rp: RP, ciudades=CIUDADES, duration: int = 24):
    """Máximos estacionales (puntos), GEV (línea) y POT (discontinua) en ciudades."""

    names = list(ciudades)
    ncol = 3
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.8 * nrow), squeeze=False)
    Tgrid = np.geomspace(1.05, 100.0, 200)
    for ax, name in zip(axes.flat, names, strict=False):
        j, i = rp.pixel(*ciudades[name])
        am = rp.get("seasonal_max", duration)[:, j, i]
        am = np.sort(am[np.isfinite(am)])[::-1]
        n = am.size
        if n:
            m = np.arange(1, n + 1)
            T_emp = (n + 0.12) / (m - 0.44)                       # Gringorten
            ax.plot(T_emp, am, "o", color="#2A6FC9", ms=4, label=f"máximos estacionales ({n})")
        xi, s, mu = (rp.get(f"gev_{k}", duration)[j, i] for k in ("xi", "sigma", "mu"))
        ax.plot(Tgrid, gev_return_level(Tgrid, xi, s, mu), "-", color="#2A6FC9", lw=2, label="GEV")
        gp = [rp.get(f"gpd_{k}", duration)[j, i] for k in ("u", "sigma", "xi", "rate")]
        Rgrid = np.geomspace(1.0 / max(gp[3], 1e-9) if np.isfinite(gp[3]) else 0.3, 100.0, 200)
        ax.plot(annual_return_period(Rgrid), pot_return_level(Rgrid, *gp), "--", color="#D9822B", lw=2,
                label="POT (picos independientes)")
        for T, ls in ((2, ":"), (10, ":")):
            ax.axvline(T, color="0.7", lw=0.8, ls=ls)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 25, 50, 100], ["1", "2", "5", "10", "25", "50", "100"])
        ax.set(title=f"{name} ({rp.lat[j]:.1f}°, {rp.lon[i]:.1f}°)", xlabel="periodo de retorno T (años)",
               ylabel=f"lluvia en {duration} h (mm)", xlim=(1, 100))
    for ax in axes.flat[len(names):]:
        ax.set_visible(False)
    axes.flat[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"Cómo leer un periodo de retorno: la lluvia de {duration} h que se espera una vez cada T años",
                 y=1.01)
    fig.tight_layout()
    return fig


# ============================================================================
# TABLA RESUMEN
# ============================================================================


def tabla_resumen(rp: RP, percentiles: dict | None = None, T_show=(2.0, 5.0, 10.0, 25.0)):
    """
    Medianas sobre México: lluvia con periodo de retorno T (GEV) y, si se dan
    los percentiles, el intervalo de recurrencia de cada uno (POT, años).
    """
    import pandas as pd

    rows = []
    box = rp.box()
    for d in rp.durations:
        r = {"duración (h)": d}
        rl = rp.get("rl_gev", d)
        for T in T_show:
            r[f"T = {T:g} años (mm)"] = float(np.nanmedian(rl[list(rp.T).index(T)][box]))
        if percentiles and d in percentiles:
            u, s, xi, rate = (rp.get(f"gpd_{k}", d) for k in ("u", "sigma", "xi", "rate"))
            for q, x in percentiles[d].items():
                Rq = pot_recurrence_interval(x, u, s, xi, rate)
                below = np.isfinite(x) & np.isfinite(u) & (x < u)
                med = float(np.nanmedian(np.where(below, 0.0, Rq)[box]))
                r[f"{q.replace('Q', 'P')}: cada (años)"] = med if med > 0 else f"< {1 / np.nanmedian(rate[box]):.2f}"
        rows.append(r)
    return pd.DataFrame(rows).set_index("duración (h)")
