"""
ITCHI-CORE · Figuras del avance v0.2.1 (notebooks 11–14 de notebooks/third test/)

Parte A, sin datos (notebook 11): cómo determina el peligro ITCHI v0.2.1.

    fig_a1_flujo              diagrama: entradas -> H_P, radios, H_W -> regiones -> ITCHI -> clase -> municipio
    fig_a2_lluvia_rampas      H_P contra P6 con máximo en P99 y en P95, sin y con suelo saturado
    fig_a3_lluvia_plano       clase de H_P en el plano P6 x A72, sin y con persistencia (R24)
    fig_a4_viento_escalas     H_W contra viento local: v0.2.0 (relativo a Vmax), corte 12 m/s, Saffir-Simpson
    fig_a5_viento_radial      H_W contra distancia en tres tormentas (TT, H2, H5)
    fig_a6_regiones           un snapshot sintético con el pipeline: regiones, H_P, H_W e ITCHI
    fig_a7_combinacion        clase de ITCHI = 1 - (1 - H_W)(1 - H_P)
    fig_a8_municipio          de píxel a municipio: máximo, media y P90

Parte B, con los campos de casos_v021.load_or_compute (notebooks 12 y 13):

    fig_mapa_evento           clase del máximo del evento, en malla y por municipio, por experimento y Δ
    fig_mapa_campo            máximo del evento de un campo continuo (H_P, H_W) por experimento
    fig_mapa_posicion         clase de ITCHI en una posición, en malla y por municipio
    fig_conteos               municipios por clase, por caso y experimento
    fig_percentiles           P90, P95 y P99 de 6 h sobre México

Parte C, deslizamientos (notebook 14):

    fig_serie_deslizamiento   P6, R24, A72 contra sus percentiles; H_P, H_W, ITCHI y clase municipales
    fig_mapa_deslizamiento    mapas de clase antes del deslizamiento con el municipio resaltado
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from matplotlib.path import Path as MplPath

try:
    import itchi  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itchi.constants import QUADRANTS, SAFFIR_SIMPSON_HAZARD_NODES, SAFFIR_SIMPSON_KT
from itchi.parametric_radii import compute_w06_reference_radii, compute_w06_wind_hazard_with_tail
from itchi.plotting import SIAT_DANGER_COLORS
from itchi.precipitation import PRECIP_MATRIX_V02, PRECIP_MATRIX_V02_P95, compute_precipitation_matrix_hazard
from itchi.wind import normalize_wind_hazard_absolute, normalize_wind_hazard_with_tail
from itchi.wind_profiles import KNOT_TO_MS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import casos_v021 as cv  # noqa: E402

# Percentiles medianos sobre México, mm por ventana (verification_v02.md, tabla 1)
Q_MEX = {
    6: {"Q75": 9.8, "Q90": 20.1, "Q95": 29.3, "Q99": 52.5},
    24: {"Q75": 15.0, "Q90": 31.7, "Q95": 47.0, "Q99": 87.9},
    72: {"Q75": 24.4, "Q90": 48.8, "Q95": 70.3, "Q99": 126.9},
}

CLASS_NAMES = ("sin peligro", "mínimo", "bajo", "moderado", "alto", "máximo")
CLASS_CMAP = ListedColormap(SIAT_DANGER_COLORS)
CLASS_NORM = BoundaryNorm(np.arange(-0.5, 6.5, 1.0), CLASS_CMAP.N)
CLASS_CMAP_NO0 = ListedColormap(["none", *SIAT_DANGER_COLORS[1:]])
DIFF_CMAP = plt.get_cmap("RdBu_r")
KT_12MS = 12.0 / KNOT_TO_MS
KT_2MS = 2.0 / KNOT_TO_MS
EXP_COLOR = {"v020": "#2A6FC9", "p95": "#D9822B", "w12": "#2A9D8F", "ss": "#C0392B", "p95ss": "#8E5CC2",
             "P99": "#2A6FC9", "P95": "#D9822B"}
EXP_LS = {"v020": "-", "p95": "--", "w12": "-.", "ss": "-", "p95ss": ":", "P99": "-", "P95": "--"}
MATRICES = (("P99", PRECIP_MATRIX_V02), ("P95", PRECIP_MATRIX_V02_P95))


def siat(x) -> np.ndarray:
    return cv.siat_class(x)


def class_legend(ax, loc="lower left", fontsize=8, classes=range(1, 6), **kw):
    handles = [Patch(facecolor=SIAT_DANGER_COLORS[k], edgecolor="0.3", label=f"{k} {CLASS_NAMES[k]}") for k in classes]
    return ax.legend(handles=handles, loc=loc, fontsize=fontsize, framealpha=0.9, **kw)


def class_bands(ax, axis="y", alpha=0.18):
    # Fondo con las cinco clases de 0.2 sobre el eje del peligro
    for k in range(1, 6):
        lo, hi = 0.2 * (k - 1), 0.2 * k
        if axis == "y":
            ax.axhspan(lo, hi, color=SIAT_DANGER_COLORS[k], alpha=alpha, lw=0, zorder=0)
        else:
            ax.axvspan(lo, hi, color=SIAT_DANGER_COLORS[k], alpha=alpha, lw=0, zorder=0)


# ============================================================================
# PARTE A. CÓMO DETERMINA EL PELIGRO ITCHI v0.2.1 (SIN DATOS)
# ============================================================================


def _box(ax, xy, w, h, text, fc="#F4F6F8", ec="0.25", fs=8.5, weight="normal"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.015",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight=weight, wrap=True)
    return (x, y, w, h)


def _arrow(ax, p0, p1, color="0.3", style="-|>", lw=1.2, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=11, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def fig_a1_flujo():
    """Diagrama de flujo de ITCHI v0.2.1, con lo que cambia en el avance resaltado."""
    fig, ax = plt.subplots(figsize=(15, 8.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    c_in, c_p, c_w, c_g, c_out, c_new = "#EAF1FB", "#E7F5EC", "#FDEEE3", "#F2ECF8", "#FFF7D6", "#FDE2E1"

    # Entradas
    _box(ax, (0.01, 0.80), 0.20, 0.15, "MSWEP V3, 6 h\nP6 = (T-6 h, T]\nR24 = (T-24 h, T]\nA72 = (T-78 h, T-6 h]", c_in)
    _box(ax, (0.01, 0.55), 0.20, 0.17, "Climatología local\nmayo-nov 2000-2025\nP75, P90, P95, P99\nde 6, 24 y 72 h por píxel", c_in)
    _box(ax, (0.01, 0.30), 0.20, 0.17, "IBTrACS (cada 6 h)\ncentro, Vmax, RMW\nR34 por cuadrante", c_in)
    _box(ax, (0.01, 0.08), 0.20, 0.15, "ROCLOUD por cuadrante\n(radio de la nubosidad\nasociada al ciclón)", c_in)

    # Lluvia
    _box(ax, (0.27, 0.67), 0.22, 0.28,
         "PELIGRO POR LLUVIA  H_P\nrampa de P6 sobre sus percentiles:\nP75 → 0, P90 → 0.25, P95 → 0.5,\nmáximo → 1\n\n"
         "escalada (suelo saturado):\nf = w(R24: P90→P95) · w(A72: 0→P75)\nH_P = (1-f) rampa + f rampa escalada\n(P95 → 0.8)",
         c_p, fs=8.2)
    _box(ax, (0.27, 0.565), 0.22, 0.085, "caso 1: máximo en P99 (v0.2.0)\nfrente a máximo en P95", c_new, fs=8.2, weight="bold")

    # Radios
    _box(ax, (0.27, 0.27), 0.22, 0.26,
         "RADIOS POR CUADRANTE\nR_direct = R34 observado\n(sin R34: RMW o R34 paramétrico)\n"
         "R_W06 = radio donde W06 (PA21)\nbaja a 2 m/s\nR_attr = ROCLOUD con piso R_W06", c_g, fs=8.2)

    # Viento
    _box(ax, (0.27, 0.02), 0.22, 0.22,
         "PELIGRO POR VIENTO  H_W\nperfil W06 calibrado a R34\nv0.2.0: relativo a Vmax,\ncola de 34 kt hasta R_W06 (2 m/s)",
         c_w, fs=8.2)
    _box(ax, (0.505, 0.02), 0.19, 0.22,
         "caso 2 (propuesta v0.2.1)\ncola cortada en 12 m/s\nviento absoluto Saffir-Simpson:\ncola → 1, TT → 2, H1-H2 → 3,\nH3 → 4, H4-H5 → 5",
         c_new, fs=8.2, weight="bold")

    # Regiones y combinación
    _box(ax, (0.55, 0.55), 0.20, 0.40,
         "REGIONES\n\ndirecta: r ≤ R_direct\nH_dir = 1-(1-H_W)(1-H_P)\n\nindirecta: R_direct < r ≤ R_attr\n"
         "H_ind = 1-(1-H_W,cola)(1-H_P)\n(la cola de viento solo\nhasta R_W06 o 12 m/s)\n\nexterior: r > R_attr → 0", c_g, fs=8.2)
    _box(ax, (0.60, 0.30), 0.15, 0.18, "ITCHI =\n1-(1-H_dir)(1-H_ind)\n(regiones excluyentes:\nITCHI = H_dir o H_ind)", c_out, fs=8.2, weight="bold")

    # Salidas
    _box(ax, (0.79, 0.73), 0.20, 0.22, "CLASE (tramos de 0.2)\n1 mínimo   (0-0.2]\n2 bajo        (0.2-0.4]\n3 moderado (0.4-0.6]\n"
         "4 alto        (0.6-0.8]\n5 máximo   (0.8-1]", c_out, fs=8.4)
    _box(ax, (0.79, 0.45), 0.20, 0.21, "MALLA (0.1°)\npor posición (6 h) y\nmáximo del evento\nITCHI_max(x) = max_t ITCHI(x, t)", c_out, fs=8.4)
    _box(ax, (0.79, 0.17), 0.20, 0.21, "MUNICIPIO\nmáximo de los píxeles cuyo\ncentro cae en el municipio\n(alternativas: media, P90)", c_out, fs=8.4)
    _box(ax, (0.79, 0.01), 0.20, 0.12, "caso 3: serie municipal\ny anticipación al\ndeslizamiento", c_new, fs=8.2, weight="bold")

    for y in (0.875, 0.635):
        _arrow(ax, (0.21, y), (0.27, 0.80 if y > 0.7 else 0.75))
    _arrow(ax, (0.21, 0.385), (0.27, 0.40))
    _arrow(ax, (0.21, 0.155), (0.27, 0.35))
    _arrow(ax, (0.21, 0.36), (0.27, 0.13))
    _arrow(ax, (0.38, 0.27), (0.38, 0.24))
    _arrow(ax, (0.49, 0.81), (0.55, 0.81))
    _arrow(ax, (0.49, 0.40), (0.55, 0.66))
    _arrow(ax, (0.57, 0.24), (0.57, 0.55))
    _arrow(ax, (0.675, 0.55), (0.675, 0.48))
    _arrow(ax, (0.75, 0.39), (0.79, 0.84))
    _arrow(ax, (0.75, 0.39), (0.79, 0.555))
    _arrow(ax, (0.89, 0.45), (0.89, 0.38))
    _arrow(ax, (0.89, 0.17), (0.89, 0.13))
    ax.text(0.01, 1.035, "ITCHI v0.2.1: cómo se determina el peligro (en rojo, lo que cambia en este avance)",
            fontsize=13, weight="bold", va="top")
    return fig


def _hp_curve(p6, r24, a72, matrix, q=Q_MEX):
    shape = np.shape(p6)
    return compute_precipitation_matrix_hazard(
        np.asarray(p6, float), q[6],
        {"R24": np.broadcast_to(r24, shape).astype(float), "A72": np.broadcast_to(a72, shape).astype(float)},
        {"R24": q[24], "A72": q[72]}, matrix=matrix)


def fig_a2_lluvia_rampas(q=Q_MEX):
    """H_P contra P6 (mediana de México): máximo en P99 frente a P95, sin y con escalada."""
    p6 = np.linspace(0.0, 65.0, 1301)
    states = {
        "sin escalada (f = 0)\nlluvia aislada, o persistente con suelo seco": (p6, 0.0),
        "escalada parcial (f = 0.5)\nR24 ≥ P95 de 24 h, A72 = P75/2 de 72 h": (np.maximum(p6, q[24]["Q95"]), 0.5 * q[72]["Q75"]),
        "escalada completa (f = 1), suelo saturado\nR24 ≥ P95 de 24 h y A72 ≥ P75 de 72 h": (np.maximum(p6, q[24]["Q95"]), q[72]["Q75"]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    for ax, (title, (r24, a72)) in zip(axes, states.items(), strict=True):
        class_bands(ax)
        for name, m in MATRICES:
            h = _hp_curve(p6, r24, a72, m, q)
            ax.plot(p6, h, color=EXP_COLOR[name], ls=EXP_LS[name], lw=2.2, label=f"máximo en {name}")
        for k in ("Q75", "Q90", "Q95", "Q99"):
            ax.axvline(q[6][k], color="0.35", lw=0.7, ls=":")
            ax.text(q[6][k], 1.01, k.replace("Q", "P"), ha="center", fontsize=8,
                    transform=ax.get_xaxis_transform())
        ax.set_title(title, fontsize=9.5, pad=18)
        ax.set_xlabel("P6 (mm en 6 h)")
        ax.set_xlim(0, 65)
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("H_P")
    axes[0].legend(loc="upper left", fontsize=9)
    fig.suptitle("Peligro por lluvia: el máximo pasa del P99 al P95 y la escalada por suelo saturado actúa en el P90 "
                 "(percentiles medianos de México)",
                 fontsize=11, y=1.04)
    return fig


def class_thresholds_mm(matrix, f: float, q=Q_MEX) -> pd.Series:
    """P6 (mm) a partir del cual H_P entra en cada clase, con peso de escalada f."""
    p6 = np.linspace(0.0, 80.0, 16001)
    r24, a72 = (np.maximum(p6, q[24]["Q95"]), f * q[72]["Q75"]) if f > 0 else (p6, 0.0)
    h = _hp_curve(p6, r24, a72, matrix, q)
    out = {}
    for k in range(1, 6):
        hit = np.flatnonzero(h > 0.2 * (k - 1))
        out[f"clase {k}"] = p6[hit[0]] if hit.size else np.nan
    return pd.Series(out)


def tabla_umbrales_lluvia(q=Q_MEX) -> pd.DataFrame:
    rows = {}
    for name, m in MATRICES:
        for f in (0.0, 1.0):
            rows[(f"máximo {name}", "suelo saturado" if f else "sin escalada")] = class_thresholds_mm(m, f, q)
    return pd.DataFrame(rows).T.round(1)


def fig_a3_lluvia_plano(q=Q_MEX):
    """Clase de H_P en el plano P6 x A72, sin persistencia (R24 = P6) y con R24 ≥ P95."""
    p6 = np.linspace(0.0, 65.0, 261)
    a72 = np.linspace(0.0, 60.0, 241)
    P, A = np.meshgrid(p6, a72)
    rows = {"R24 = P6 (solo llovió en las últimas 6 h)": P, "R24 ≥ P95 de 24 h (lluvia persistente)": np.maximum(P, q[24]["Q95"])}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5), sharex=True, sharey=True)
    for i, (rname, R) in enumerate(rows.items()):
        for j, (name, m) in enumerate(MATRICES):
            ax = axes[i, j]
            h = compute_precipitation_matrix_hazard(P, q[6], {"R24": R, "A72": A}, {"R24": q[24], "A72": q[72]}, matrix=m)
            ax.pcolormesh(P, A, siat(h), cmap=CLASS_CMAP, norm=CLASS_NORM, shading="auto")
            for k in ("Q75", "Q90", "Q95", "Q99"):
                ax.axvline(q[6][k], color="k", lw=0.6, ls=":")
                if i == 0:
                    ax.text(q[6][k], 1.01, k.replace("Q", "P"), ha="center", fontsize=8,
                            transform=ax.get_xaxis_transform())
            ax.axhline(q[72]["Q75"], color="k", lw=0.9, ls="--")
            ax.text(64, q[72]["Q75"] + 0.8, "A72 = P75 (suelo saturado)", ha="right", fontsize=8)
            ax.set_title(f"máximo en {name} · {rname}", fontsize=9.5, pad=18)
    for ax in axes[-1]:
        ax.set_xlabel("P6 (mm en 6 h)")
    for ax in axes[:, 0]:
        ax.set_ylabel("A72 (mm en las 72 h previas)")
    class_legend(axes[0, 0], loc="upper left", classes=range(0, 6))
    fig.suptitle("Clase de H_P: el suelo saturado (A72 ≥ P75) escala solo si R24 ≥ P90 de 24 h; con R24 = P6 eso ocurre "
                 "cuando la ventana de 6 h ya supera el P90 de 24 h (≈ 32 mm)", fontsize=10.5)
    fig.tight_layout()
    return fig


def fig_a4_viento_escalas():
    """H_W contra viento local en la región directa y en la cola, para las tres variantes."""
    v = np.linspace(0.0, 180.0, 1801)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
    titles = ("v0.2.0: relativo a Vmax, cola hasta R_W06 (2 m/s)",
              "corte de la cola en 12 m/s (todavía relativo a Vmax)",
              "propuesta v0.2.1: absoluto, Saffir-Simpson, cola desde 12 m/s")
    vmaxes = (45.0, 70.0, 100.0, 140.0)
    colors = ("#6B7280", "#2A9D8F", "#D9822B", "#C0392B")
    for ax, title, v_out in zip(axes, titles, (KT_2MS, KT_12MS, KT_12MS), strict=True):
        class_bands(ax)
        if ax is not axes[2]:
            for vm, c in zip(vmaxes, colors, strict=True):
                vv = v[v <= vm]
                h = normalize_wind_hazard_with_tail(vv, vm, direct_mask=vv >= 34.0, outer_wind_kt=v_out, h34=0.2)
                ax.plot(vv, h, color=c, lw=2, label=f"Vmax = {vm:.0f} kt")
                ax.plot(vm, h[-1], "o", color=c, ms=5)
        else:
            h = normalize_wind_hazard_absolute(v, outer_wind_kt=v_out)
            ax.plot(v, h, color="k", lw=2.4, label="cualquier tormenta")
            for k, vk in SAFFIR_SIMPSON_KT.items():
                ax.axvline(vk, color="0.3", lw=0.7, ls=":")
                ax.text(vk + 1, 0.02, k.replace("TS", "TT"), fontsize=8, rotation=90, va="bottom")
        ax.axvline(v_out, color="0.2", lw=1.0, ls="--")
        ax.text(v_out + 1, 0.93, f"fin de la cola\n{v_out:.1f} kt", fontsize=8, va="top")
        ax.axvline(34.0, color="0.2", lw=1.0)
        ax.set_xlim(0, 180)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("viento sostenido local V (kt)")
        ax.set_title(title, fontsize=9.5)
        ax.legend(loc="center right", fontsize=8)
    axes[0].set_ylabel("H_W")
    fig.suptitle("Peligro por viento: en v0.2.0 la clase depende de Vmax (una tormenta de 45 kt llega a la clase 5); "
                 "con Saffir-Simpson solo depende del viento local", fontsize=10.5, y=1.02)
    return fig


def _radial_hazard(vmax, lat=18.0, r=None, variant="v020"):
    r = np.linspace(0.0, 900.0, 1801) if r is None else r
    ref = compute_w06_reference_radii(vmax, lat)
    r34 = ref["r34_km"] if np.isfinite(ref["r34_km"]) else ref["rmax_km"]
    kw = dict(radius_km=r, direct_radius_km=r34, attribution_radius_km=ref["r_outer_km"], vmax_kt=vmax, lat=lat,
              r34_mean_km=r34 if vmax > 34 else None)
    if variant == "v020":
        out = compute_w06_wind_hazard_with_tail(**kw)
    elif variant == "w12":
        out = compute_w06_wind_hazard_with_tail(**kw, tail_end_wind_kt=KT_12MS)
    else:
        out = compute_w06_wind_hazard_with_tail(**kw, tail_end_wind_kt=KT_12MS, hazard_scale="saffir_simpson")
    return r, out, ref


def fig_a5_viento_radial(storms=((50.0, "tormenta tropical, 50 kt"), (90.0, "huracán categoría 2, 90 kt"),
                                 (150.0, "huracán categoría 5, 150 kt"))):
    """H_W contra distancia al centro (perfil W06, PA21) para tres intensidades."""
    fig, axes = plt.subplots(1, len(storms), figsize=(16, 4.8), sharey=True)
    for ax, (vmax, title) in zip(axes, storms, strict=True):
        class_bands(ax)
        for tag, lab in (("v020", "v0.2.0"), ("w12", "corte 12 m/s"), ("ss", "v0.2.1 Saffir-Simpson")):
            r, out, ref = _radial_hazard(vmax, variant=tag)
            ax.plot(r, out["H_W"], color=EXP_COLOR[tag], ls=EXP_LS[tag], lw=2.2, label=lab)
        ax2 = ax.twinx()
        ax2.plot(r, out["wind_kt"], color="0.35", lw=1.0, ls=":", label="viento W06 (kt)")
        ax2.set_ylim(0, 180)
        if ax is axes[-1]:
            ax2.set_ylabel("viento W06 (kt, punteado)")
        else:
            ax2.set_yticklabels([])
        r_12 = r[np.flatnonzero((out["wind_kt"] >= KT_12MS))[-1]]
        for x, name in ((ref["rmax_km"], "RMW"), (ref["r34_km"], "R34"), (r_12, "12 m/s"), (ref["r_outer_km"], "R_W06")):
            if np.isfinite(x):
                ax.axvline(x, color="0.25", lw=0.8, ls="--")
                ax.text(x + 6, 0.97, name, fontsize=8, rotation=90, va="top")
        ax.set_xlim(0, 900)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("distancia al centro (km)")
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("H_W")
    axes[0].legend(loc="center right", fontsize=8.5)
    fig.suptitle("Peligro por viento contra distancia: el corte en 12 m/s acorta la cola; Saffir-Simpson "
                 "baja la tormenta débil y deja igual la intensa", fontsize=10.5, y=1.02)
    return fig


def synthetic_snapshot(vmax=100.0, center=(-100.0, 18.0), seed=4):
    """Snapshot sintético con el pipeline: lluvia en espiral, R34 asimétrico y ROCLOUD con un cuadrante faltante."""
    from itchi.pipeline import compute_itchi_snapshot_v02

    lon, lat = np.meshgrid(np.arange(-112.5, -87.5, 0.1), np.arange(6.5, 29.5, 0.1))
    dx = (lon - center[0]) * np.cos(np.radians(center[1])) * 111.2
    dy = (lat - center[1]) * 111.2
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    rng = np.random.default_rng(seed)
    phase = np.mod(th + np.pi, 2 * np.pi)
    band = sum(np.exp(-((r - 110 - 55 * (phase + 2 * np.pi * k)) / 60.0) ** 2) for k in (-1, 0, 1))
    core = np.exp(-(r / 90.0) ** 2)
    p6 = 70 * core + 45 * band + rng.gamma(0.5, 2.0, r.shape)
    r24 = p6 * 2.2
    a72 = 40 * np.exp(-((dx + 150) ** 2 + (dy - 80) ** 2) / 300.0 ** 2)      # suelo húmedo al noroeste
    q = {d: {k: np.full(r.shape, val) for k, val in Q_MEX[d].items()} for d in Q_MEX}
    case = dict(lon=lon, lat=lat, center_lon=center[0], center_lat=center[1], vmax_kt=vmax, rmw_km=30.0,
                r34_by_quadrant={"RNE": 110, "RSE": 90, "RSW": 60, "RNW": 75}, r34_unit="nm",
                rocloud_by_quadrant={"RNE": 1150.0, "RSE": 950.0, "RSW": np.nan, "RNW": 300.0}, rocloud_unit="km")
    common = dict(quantiles_6h=q[6], quantiles_24h=q[24], quantiles_72h=q[72], **case)
    out = {
        "v020": compute_itchi_snapshot_v02(p6, r24, a72, **common),
        "ss": compute_itchi_snapshot_v02(p6, r24, a72, **common, wind_tail_end_kt=KT_12MS,
                                         wind_hazard_scale="saffir_simpson"),
        "p95": compute_itchi_snapshot_v02(p6, r24, a72, **common, precip_matrix=PRECIP_MATRIX_V02_P95),
    }
    return lon, lat, p6, out, case


def _quadrant_ring(ax, center, radii_km, **kw):
    import itchi.itchi_size_helpers as H

    x, y = H.quadrant_outline(center[1], center[0], radii_km)
    ax.plot(x, y, **kw)


def fig_a6_regiones():
    """Un snapshot sintético calculado con compute_itchi_snapshot_v02 (v0.2.0, P95 y Saffir-Simpson)."""
    lon, lat, p6, out, case = synthetic_snapshot()
    center = (case["center_lon"], case["center_lat"])
    res = out["v020"]
    region = np.where(res["M_direct"], 2, np.where(res["M_indirect"], 1, 0)).astype(float)
    panels = [
        ("regiones", region, ListedColormap(["#FFFFFF", "#E8DDF5", "#B59AD6"]), BoundaryNorm([-0.5, 0.5, 1.5, 2.5], 3)),
        ("P6 (mm)", p6, plt.get_cmap("Blues"), None),
        ("H_P · máximo P99", out["v020"]["H_P"], plt.get_cmap("YlGnBu"), None),
        ("H_P · máximo P95", out["p95"]["H_P"], plt.get_cmap("YlGnBu"), None),
        ("H_W · v0.2.0", out["v020"]["H_W"], plt.get_cmap("OrRd"), None),
        ("H_W · v0.2.1 Saffir-Simpson", out["ss"]["H_W"], plt.get_cmap("OrRd"), None),
        ("clase ITCHI · v0.2.0", siat(out["v020"]["ITCHI"]), CLASS_CMAP_NO0, CLASS_NORM),
        ("clase ITCHI · v0.2.1 Saffir-Simpson", siat(out["ss"]["ITCHI"]), CLASS_CMAP_NO0, CLASS_NORM),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8.6), sharex=True, sharey=True)
    for ax, (title, f, cmap, norm) in zip(axes.ravel(), panels, strict=True):
        kw = dict(cmap=cmap, shading="auto")
        if norm is not None:
            kw["norm"] = norm
        elif title.startswith(("H_", "clase")):
            kw.update(vmin=0, vmax=1)
        m = ax.pcolormesh(lon, lat, f, **kw)
        if not title.startswith(("regiones", "clase")):
            fig.colorbar(m, ax=ax, shrink=0.75, pad=0.02)
        _quadrant_ring(ax, center, res["R_direct_by_quadrant"], color="k", lw=1.3)
        _quadrant_ring(ax, center, res["R_W06_by_quadrant"], color="#D9822B", lw=1.3, ls="--")
        _quadrant_ring(ax, center, res["R_attr_by_quadrant"], color="m", lw=1.5)
        ax.plot(*center, "k*", ms=9)
        ax.set_title(title, fontsize=10)
        ax.set_aspect(1 / np.cos(np.radians(center[1])))
    ax = axes[0, 0]
    ax.plot([], [], "k-", label="R_direct (R34)")
    ax.plot([], [], "--", color="#D9822B", label="R_W06 (fin de la cola en v0.2.0)")
    ax.plot([], [], "m-", label="R_attr (ROCLOUD, piso R_W06)")
    ax.legend(loc="lower left", fontsize=7.5)
    ax.text(center[0], center[1] + 0.4, "directa", ha="center", fontsize=8)
    ax.text(center[0] - 4.5, center[1] + 3.5, "indirecta", ha="center", fontsize=8)
    class_legend(axes[1, 3], loc="lower left", fontsize=7.5)
    fig.suptitle("Snapshot sintético (100 kt; ROCLOUD mayor que R_W06 en NE y SE, sin dato en SO y menor en NO): "
                 "qué región, qué variable y qué escala decide cada píxel", fontsize=11)
    fig.tight_layout()
    return fig


def fig_a7_combinacion():
    """Clase de ITCHI = 1 - (1 - H_W)(1 - H_P) en el plano (H_P, H_W)."""
    h = np.linspace(0.0, 1.0, 401)
    HP, HW = np.meshgrid(h, h)
    itchi = 1.0 - (1.0 - HW) * (1.0 - HP)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.pcolormesh(HP, HW, siat(itchi), cmap=CLASS_CMAP, norm=CLASS_NORM, shading="auto")
    cs = ax.contour(HP, HW, itchi, levels=[0.2, 0.4, 0.6, 0.8], colors="k", linewidths=0.8)
    ax.clabel(cs, fmt="%.1f", fontsize=8)
    for x in (0.2, 0.4, 0.6, 0.8):
        ax.axvline(x, color="0.4", lw=0.5, ls=":")
        ax.axhline(x, color="0.4", lw=0.5, ls=":")
    ax.annotate("H_P = 0.5 y H_W = 0.5\n→ ITCHI = 0.75 (clase 4)", xy=(0.5, 0.5), xytext=(0.45, 0.06), fontsize=8.5,
                arrowprops=dict(arrowstyle="->", lw=0.9), bbox=dict(fc="w", ec="0.5", alpha=0.9))
    ax.plot(0.5, 0.5, "ko", ms=4)
    ax.set_xlabel("H_P")
    ax.set_ylabel("H_W")
    ax.set_title("Unión probabilística: ITCHI ≥ max(H_P, H_W);\ndos peligros moderados suben una clase", fontsize=10)
    class_legend(ax, loc="upper right", fontsize=7.5, classes=range(0, 6))
    return fig


def fig_a8_municipio(seed=2):
    """De píxel a municipio: tres municipios sintéticos sobre una malla de 0.1°."""
    rng = np.random.default_rng(seed)
    x = np.arange(0.05, 3.0, 0.1)
    X, Y = np.meshgrid(x, x)
    field = np.clip(0.95 * np.exp(-((X - 0.8) ** 2 + (Y - 2.0) ** 2) / 0.35) + 0.25 * np.exp(-((X - 2.2) ** 2 + (Y - 0.8) ** 2) / 0.9)
                    + rng.normal(0, 0.03, X.shape), 0, 1)
    polys = {
        "A (grande, núcleo de lluvia en una esquina)": [(0.2, 1.1), (1.7, 1.2), (1.9, 2.8), (0.3, 2.85)],
        "B (mediano)": [(1.7, 1.2), (2.85, 1.0), (2.9, 2.6), (1.9, 2.8)],
        "C (pequeño)": [(0.7, 0.35), (1.08, 0.35), (1.08, 0.62), (0.7, 0.62)],
        "D (sin centros de píxel)": [(2.21, 0.21), (2.29, 0.21), (2.29, 0.29), (2.21, 0.29)],
    }
    pts = np.column_stack([X.ravel(), Y.ravel()])
    pix = {}
    for name, verts in polys.items():
        inside = np.flatnonzero(MplPath(verts).contains_points(pts))
        if inside.size == 0:
            c = np.mean(verts, axis=0)
            inside = np.array([np.argmin(((pts - c) ** 2).sum(axis=1))])
            pix[name] = (inside, 0)
        else:
            pix[name] = (inside, inside.size)
    codes = list(polys)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
    titles = ["clase por píxel", "municipio: máximo (usado)", "municipio: media", "municipio: P90"]
    for ax, title, stat in zip(axes, titles, (None, "max", "mean", "p90"), strict=True):
        if stat is None:
            f = field
        else:
            vals = cv.municipal_values(field, pix, codes, stat)
            f = cv.paint_municipal(vals, pix, codes, field.shape)
            for name, v in zip(codes, vals, strict=True):
                c = np.mean(polys[name], axis=0)
                ax.text(c[0], c[1], f"{v:.2f}", ha="center", va="center", fontsize=8.5, weight="bold",
                        bbox=dict(fc="w", ec="none", alpha=0.7, pad=1))
        ax.pcolormesh(X, Y, siat(f) if stat is None else np.where(np.isnan(f), np.nan, siat(f)),
                      cmap=CLASS_CMAP_NO0, norm=CLASS_NORM, shading="auto")
        for verts in polys.values():
            vv = np.vstack([verts, verts[:1]])
            ax.plot(vv[:, 0], vv[:, 1], "k-", lw=1.4)
        if stat is None:
            ax.plot(X, Y, "k.", ms=1.2, alpha=0.4)
            for name, verts in polys.items():
                ax.text(*np.min(verts, axis=0), name.split(" ")[0], fontsize=10, weight="bold", va="top")
        ax.set_title(title, fontsize=10)
        ax.set_aspect(1)
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 3)
        ax.set_xticks([])
        ax.set_yticks([])
    class_legend(axes[0], loc="upper left", fontsize=7.5, bbox_to_anchor=(0.0, -0.02), ncol=5)
    fig.suptitle("De píxel a municipio. Puntos: centros de píxel de 0.1°. El municipio D no contiene ningún centro "
                 "y toma el píxel más cercano", fontsize=10.5)
    return fig


# ============================================================================
# PARTE B. MAPAS DE LOS CASOS
# ============================================================================


def _pc():
    import cartopy.crs as ccrs

    return ccrs.PlateCarree()


def event_extent(C: dict, pad: float = 1.5, step: float = 2.0, tags=None) -> tuple[tuple, tuple]:
    """Extensión de la huella del evento sobre México (ITCHI_max > 0 en tierra)."""
    ds = C["ds"]
    tags = list(ds["exp"].values) if tags is None else list(tags)
    m = np.zeros(C["land"].shape, bool)
    for e in tags:
        m |= ds["ITCHI"].sel(exp=e).max("time").values > 0
    m &= C["land"]
    lon2d, lat2d = np.meshgrid(ds["lon"].values, ds["lat"].values)
    if not m.any():
        m = C["land"]
    lo0 = np.floor((lon2d[m].min() - pad) / step) * step
    lo1 = np.ceil((lon2d[m].max() + pad) / step) * step
    la0 = np.floor((lat2d[m].min() - pad) / step) * step
    la1 = np.ceil((lat2d[m].max() + pad) / step) * step
    return (lo0, lo1), (la0, la1)


def _map_axis(ax, lon_ext, lat_ext, tick=4.0):
    from itchi.plotting import MapStyle, configure_map_axis

    style = MapStyle(lon_extent=lon_ext, lat_extent=lat_ext, lon_tick_spacing=tick, lat_tick_spacing=tick,
                     grid_spacing=tick, label_size=8.0, tick_size=3.0, tick_width=0.8, spine_linewidth=0.8,
                     coastline_linewidth=0.7, border_linewidth=0.6, land_alpha=0.35)
    return configure_map_axis(ax, crs=_pc(), style=style)


def draw_pixel_classes(ax, C: dict, field: np.ndarray, land_only: bool = False, **kw):
    ds = C["ds"]
    cls = siat(field).astype(float)
    cls[cls == 0] = np.nan
    if land_only:
        cls[~C["land"]] = np.nan
    return ax.pcolormesh(ds["lon"].values, ds["lat"].values, cls, cmap=CLASS_CMAP_NO0, norm=CLASS_NORM,
                         transform=_pc(), shading="auto", zorder=1, **kw)


def draw_municipal_classes(ax, C: dict, values: np.ndarray, mode: str = "poligonos", edge: str = "0.45"):
    """
    Clase municipal. mode="poligonos": polígonos del shapefile (geopandas);
    mode="malla": cada píxel toma el valor de su municipio (sin geopandas).
    """
    cls = siat(values)
    if mode == "poligonos":
        gdf = C["gdf"].set_index("CVEGEO").loc[C["codes"]]
        for k in range(0, 6):
            sub = gdf[cls == k]
            if len(sub):
                sub.plot(ax=ax, color="none" if k == 0 else SIAT_DANGER_COLORS[k], edgecolor=edge,
                         linewidth=0.08, transform=_pc(), zorder=2)
        return None
    painted = cv.paint_municipal(cls.astype(float), C["pix"], C["codes"], C["land"].shape)
    painted[painted == 0] = np.nan
    ds = C["ds"]
    return ax.pcolormesh(ds["lon"].values, ds["lat"].values, painted, cmap=CLASS_CMAP_NO0, norm=CLASS_NORM,
                         transform=_pc(), shading="auto", zorder=1)


def _draw_track(ax, C: dict, color="k"):
    tr, ds = C["track"], C["ds"]
    ax.plot(tr["lon"], tr["lat"], "-", color=color, lw=0.7, transform=_pc(), zorder=5)
    ax.plot(ds["track_lon"].values, ds["track_lat"].values, "o", color=color, ms=1.8, transform=_pc(), zorder=5)


def _draw_class_diff(ax, C, d, land_only=False, mun=False, mode="poligonos"):
    if mun:
        if mode == "poligonos":
            gdf = C["gdf"].set_index("CVEGEO").loc[C["codes"]]
            for v in sorted(set(d.tolist()) - {0}):
                gdf[d == v].plot(ax=ax, color=DIFF_CMAP(0.5 + v / 7.0), edgecolor="0.45", linewidth=0.08,
                                 transform=_pc(), zorder=2)
            gdf[d == 0].plot(ax=ax, color="none", edgecolor="0.6", linewidth=0.05, transform=_pc(), zorder=2)
            return
        d = cv.paint_municipal(d.astype(float), C["pix"], C["codes"], C["land"].shape)
    d = np.where(d == 0, np.nan, d).astype(float)
    if land_only:
        d[~C["land"]] = np.nan
    ds = C["ds"]
    ax.pcolormesh(ds["lon"].values, ds["lat"].values, d, cmap=DIFF_CMAP, vmin=-3.5, vmax=3.5, transform=_pc(),
                  shading="auto", zorder=1)


def fig_mapa_evento(C: dict, tags, ref: str | None = None, mode: str = "poligonos", land_only: bool = False,
                    title: str = ""):
    """
    Clase del máximo del evento. Filas: malla y municipio. Columnas: cada
    experimento de tags y, si ref, la diferencia de clase (último − ref).
    """
    ds = C["ds"]
    tags = list(tags)
    emax = {e: ds["ITCHI"].sel(exp=e).max("time").values for e in tags}
    mun = cv.municipal_event(C).max("time")
    ncol = len(tags) + (ref is not None)
    lon_ext, lat_ext = event_extent(C, tags=tags)
    fig, axes = plt.subplots(2, ncol, figsize=(4.6 * ncol, 8.4), subplot_kw={"projection": _pc()}, squeeze=False)
    for j, e in enumerate(tags):
        ax = _map_axis(axes[0, j], lon_ext, lat_ext)
        draw_pixel_classes(ax, C, emax[e], land_only=land_only)
        _draw_track(ax, C)
        ax.set_title(f"malla · {cv.label(e)}", fontsize=10)
        ax = _map_axis(axes[1, j], lon_ext, lat_ext)
        draw_municipal_classes(ax, C, mun.sel(exp=e).values, mode=mode)
        _draw_track(ax, C)
        ax.set_title(f"municipio (máximo) · {cv.label(e)}", fontsize=10)
    if ref is not None:
        last = tags[-1]
        d_pix = siat(emax[last]).astype(int) - siat(emax[ref]).astype(int)
        d_mun = siat(mun.sel(exp=last).values).astype(int) - siat(mun.sel(exp=ref).values).astype(int)
        for i, (d, is_mun) in enumerate(((d_pix, False), (d_mun, True))):
            ax = _map_axis(axes[i, -1], lon_ext, lat_ext)
            _draw_class_diff(ax, C, d, land_only=land_only, mun=is_mun, mode=mode)
            _draw_track(ax, C)
            ax.set_title(f"Δ clase: {cv.label(last)} − {cv.label(ref)}", fontsize=9.5)
        handles = [Patch(color=DIFF_CMAP(0.5 + v / 7.0), label=f"{v:+d}") for v in (-3, -2, -1, 1, 2, 3)]
        axes[1, -1].legend(handles=handles, loc="lower left", fontsize=7, ncol=2, title="Δ clase", title_fontsize=7)
    class_legend(axes[1, 0], loc="lower left", fontsize=7)
    fig.suptitle(title or f"{C['case']}: clase del máximo del evento (ITCHI_max)", fontsize=12, y=0.995)
    fig.subplots_adjust(wspace=0.08, hspace=0.12)
    return fig


def fig_mapa_campo(C: dict, field: str, tags, cmap="viridis", vmax=1.0, title: str = ""):
    """Máximo del evento de un campo continuo (H_P o H_W) por experimento."""
    ds = C["ds"]
    tags = list(tags)
    lon_ext, lat_ext = event_extent(C, tags=tags)
    fig, axes = plt.subplots(1, len(tags), figsize=(4.6 * len(tags), 4.6), subplot_kw={"projection": _pc()}, squeeze=False)
    for ax, e in zip(axes[0], tags, strict=True):
        f = ds[field].sel(exp=e).max("time").values
        ax = _map_axis(ax, lon_ext, lat_ext)
        m = ax.pcolormesh(ds["lon"].values, ds["lat"].values, np.where(f > 0, f, np.nan), cmap=cmap, vmin=0, vmax=vmax,
                          transform=_pc(), shading="auto", zorder=1)
        for lev in (0.2, 0.4, 0.6, 0.8):
            ax.contour(ds["lon"].values, ds["lat"].values, f, levels=[lev], colors="k", linewidths=0.4, transform=_pc(), zorder=3)
        _draw_track(ax, C)
        ax.set_title(f"{field}_max · {cv.label(e)}", fontsize=10)
        plt.colorbar(m, ax=ax, shrink=0.7, pad=0.02)
    fig.suptitle(title or f"{C['case']}: máximo del evento de {field} (contornos en los límites de clase)", fontsize=11)
    return fig


def attribution_radii(row) -> dict:
    """R_attr por cuadrante con la regla 'floor': ROCLOUD_q, o R_W06 si falta o es menor."""
    from itchi.rocloud import DEFAULT_ROCLOUD_COLUMN_MAP

    r_w06 = float(row["pa21_r2_km"])
    out = {}
    for q in QUADRANTS:
        v = row.get(DEFAULT_ROCLOUD_COLUMN_MAP[q], np.nan)
        v = float(v) if v is not None and np.isfinite(v) and v > 0 else 0.0
        out[q] = max(v, r_w06)
    return out


def fig_mapa_posicion(C: dict, t, tags, mode: str = "poligonos", half_width: float = 9.0):
    """Clase de ITCHI en una posición, en malla y por municipio, con los radios de esa posición."""
    import itchi.itchi_size_helpers as H

    ds = C["ds"]
    j = pd.DatetimeIndex(ds["time"].values).get_loc(pd.Timestamp(t))
    lon0, lat0 = float(ds["track_lon"][j]), float(ds["track_lat"][j])
    lon_ext = (np.floor(lon0 - half_width), np.ceil(lon0 + half_width))
    lat_ext = (np.floor(lat0 - 0.75 * half_width), np.ceil(lat0 + 0.75 * half_width))
    mun = cv.municipal_event(C)
    row = C["track"].set_index("time").loc[pd.Timestamp(t)]
    fig, axes = plt.subplots(2, len(tags), figsize=(4.6 * len(tags), 8.2), subplot_kw={"projection": _pc()}, squeeze=False)
    for k, e in enumerate(tags):
        for i in range(2):
            ax = _map_axis(axes[i, k], lon_ext, lat_ext)
            if i == 0:
                draw_pixel_classes(ax, C, ds["ITCHI"].sel(exp=e).isel(time=j).values)
            else:
                draw_municipal_classes(ax, C, mun.sel(exp=e).isel(time=j).values, mode=mode)
            x, y = H.quadrant_outline(lat0, lon0, H.direct_radii_km(row))
            ax.plot(x, y, "k-", lw=1.1, transform=_pc(), zorder=6)
            x, y = H.circle_outline(lat0, lon0, float(row["pa21_r2_km"]))
            ax.plot(x, y, "--", color="#D9822B", lw=1.0, transform=_pc(), zorder=6)
            x, y = H.quadrant_outline(lat0, lon0, attribution_radii(row))
            ax.plot(x, y, "m-", lw=1.2, transform=_pc(), zorder=6)
            _draw_track(ax, C)
            ax.plot(lon0, lat0, "k*", ms=9, transform=_pc(), zorder=7)
            ax.set_title(f"{'malla' if i == 0 else 'municipio'} · {cv.label(e)}", fontsize=10)
    class_legend(axes[1, 0], loc="lower left", fontsize=7)
    fig.suptitle(f"{C['case']} {pd.Timestamp(t):%Y-%m-%d %H} UTC · Vmax {float(ds['track_vmax_kt'][j]):.0f} kt "
                 "(negro: R_direct; naranja: R_W06; magenta: R_attr)", fontsize=11.5)
    return fig


def fig_conteos(tables: dict[str, pd.DataFrame], tags_labels, what: str = "municipios"):
    """Barras: número de municipios (o píxeles terrestres) por clase, por caso y experimento."""
    cases = list(tables)
    fig, axes = plt.subplots(1, len(cases), figsize=(3.6 * len(cases), 3.8), sharey=False, squeeze=False)
    width = 0.8 / len(tags_labels)
    for ax, case in zip(axes[0], cases, strict=True):
        tab = tables[case][what]
        for i, lab in enumerate(tags_labels):
            if lab not in tab:
                continue
            ax.bar(tab.index + (i - (len(tags_labels) - 1) / 2) * width, tab[lab], width=width,
                   color=[SIAT_DANGER_COLORS[k] for k in tab.index], edgecolor="k",
                   hatch=["", "///", "...", "xx", "\\\\"][i % 5], lw=0.6, label=lab)
        ax.set_title(case, fontsize=10)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("clase")
    axes[0, 0].set_ylabel(what.replace("_", " "))
    axes[0, 0].legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def fig_percentiles(C: dict):
    """P90, P95 y P99 de 6 h sobre la tierra del recorte: dónde cae el nuevo máximo (P95)."""
    ds = C["ds"]
    q90, q95, q99 = (ds[f"Q6h_{k}"].values for k in ("Q90", "Q95", "Q99"))
    land = C["land"]
    fig = plt.figure(figsize=(15, 4.6))
    lon_ext = (float(ds["lon"].min()), float(ds["lon"].max()))
    lat_ext = (float(ds["lat"].min()), float(ds["lat"].max()))
    for i, (f, lab, vmax) in enumerate(((q95, "P95 de 6 h (mm)", None), (q99 - q95, "P99 − P95 (mm)", None))):
        ax = fig.add_subplot(1, 3, i + 1, projection=_pc())
        _map_axis(ax, lon_ext, lat_ext, tick=6)
        m = ax.pcolormesh(ds["lon"].values, ds["lat"].values, np.where(land, f, np.nan), cmap="viridis",
                          transform=_pc(), shading="auto", vmin=0, vmax=np.nanpercentile(f[land], 98))
        plt.colorbar(m, ax=ax, shrink=0.7, label=lab)
        ax.set_title(lab, fontsize=10)
    ax3 = fig.add_subplot(1, 3, 3)
    for k, v, c in (("P90", q90, "#2A9D8F"), ("P95", q95, "#D9822B"), ("P99", q99, "#2A6FC9")):
        ax3.hist(v[land], bins=60, histtype="step", lw=1.8, color=c, label=f"{k} (mediana {np.nanmedian(v[land]):.1f} mm)")
    ax3.set_xlabel("mm en 6 h")
    ax3.legend(fontsize=8)
    ax3.set_title("distribución sobre México", fontsize=10)
    fig.suptitle("Percentiles de 6 h: con el máximo en P95, el tramo P95-P99 deja de distinguir clases", fontsize=11)
    fig.tight_layout()
    return fig


# ============================================================================
# PARTE C. DESLIZAMIENTOS Y ANTICIPACIÓN
# ============================================================================


def municipio_series(C: dict, code: str, tags=("v020", "p95", "ss", "p95ss")) -> dict:
    """
    Series del municipio: ITCHI, H_P y H_W municipales (máximo) en las
    posiciones; H_P continuo (P99, P95) en todos los tiempos; P6, R24 y A72
    en el píxel representativo (el del municipio con mayor P6/P95 en el
    evento) con sus percentiles locales.
    """
    ds, pix = C["ds"], C["pix"]
    flat = pix[code][0]
    ny, nx = C["land"].shape
    out = {"time": pd.DatetimeIndex(ds["time"].values), "time_all": pd.DatetimeIndex(ds["time_all"].values)}
    for fld in ("ITCHI", "H_P", "H_W"):
        out[fld] = {e: ds[fld].sel(exp=e).values.reshape(ds.sizes["time"], -1)[:, flat].max(axis=1) for e in tags}
    for name in cv.PRECIP_VARIANTS:
        key = f"H_P_cont_{name}"
        if key in ds:
            out[f"H_P_cont_{name}"] = ds[key].values.reshape(ds.sizes["time_all"], -1)[:, flat].max(axis=1)
    p6 = ds["P6"].values.reshape(ds.sizes["time_all"], -1)[:, flat]
    q95 = ds["Q6h_Q95"].values.ravel()[flat]
    best = int(np.nanargmax(np.nanmax(p6, axis=0) / np.maximum(q95, 1e-6)))
    k = flat[best]
    out["pixel"] = (float(ds["lat"].values[k // nx]), float(ds["lon"].values[k % nx]))
    for var, d in (("P6", 6), ("R24", 24), ("A72", 72)):
        out[var] = ds[var].values.reshape(ds.sizes["time_all"], -1)[:, k]
        out[f"Q_{var}"] = {name.split("_", 1)[1]: float(ds[name].values.ravel()[k])
                           for name in ds.data_vars if name.startswith(f"Q{d}h_")}
    return out


def tabla_anticipacion(C: dict, code: str, t_event, tags=("v020", "p95", "ss", "p95ss"), levels=(2, 3, 4, 5),
                       window_h: float = 120.0) -> pd.DataFrame:
    """Anticipación por nivel (casos_v021.lead_times) de ITCHI municipal y del H_P continuo."""
    s = municipio_series(C, code, tags)
    rows = []
    for e in tags:
        lt = cv.lead_times(s["time"], siat(s["ITCHI"][e]), t_event, levels, window_h)
        lt.insert(0, "serie", f"ITCHI · {cv.label(e)}")
        rows.append(lt)
    for name in cv.PRECIP_VARIANTS:
        key = f"H_P_cont_{name}"
        if key in s:
            lt = cv.lead_times(s["time_all"], siat(s[key]), t_event, levels, window_h)
            lt.insert(0, "serie", f"H_P continuo · máximo {name}")
            rows.append(lt)
    return pd.concat(rows, ignore_index=True)


def fig_serie_deslizamiento(C: dict, code: str, name: str, t_event, tags=("v020", "p95", "ss", "p95ss"),
                            t_decl=None, window_h=(96, 48)):
    """
    Cuatro paneles con el mismo eje de tiempo (UTC):
    a) P6, R24 y A72 en el píxel representativo, con sus percentiles locales;
    b) percentil más alto superado por cada variable;
    c) H_P continuo (P99, P95) y H_W, ITCHI municipales en las posiciones;
    d) clase municipal de ITCHI (escalones) y del H_P continuo.
    """
    s = municipio_series(C, code, tags)
    te = pd.Timestamp(t_event)
    t0, t1 = te - pd.Timedelta(hours=window_h[0]), te + pd.Timedelta(hours=window_h[1])
    fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True, gridspec_kw=dict(height_ratios=[1.4, 0.8, 1.2, 1.0]))

    ax = axes[0]
    colors = {"P6": "#2A6FC9", "R24": "#2A9D8F", "A72": "#8E5CC2"}
    for var in ("P6", "R24", "A72"):
        ax.step(s["time_all"], s[var], where="pre", color=colors[var], lw=1.8, label=f"{var} (mm)")
        for qn, qv in s[f"Q_{var}"].items():
            if qn in ("Q75", "Q90", "Q95", "Q99"):
                ax.axhline(qv, color=colors[var], lw=0.6, ls=":", alpha=0.8)
                ax.text(t1, qv, f" {var} {qn.replace('Q', 'P')}", color=colors[var], fontsize=7, va="center")
    ax.set_ylabel("mm")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"a) lluvia en el píxel representativo ({s['pixel'][0]:.2f}° N, {abs(s['pixel'][1]):.2f}° O) "
                 "y sus percentiles locales", fontsize=10, loc="left")

    ax = axes[1]
    ordinal = {0.0: 0, 75.0: 1, 90.0: 2, 95.0: 3, 99.0: 4, 99.9: 5}
    for i, var in enumerate(("P6", "R24", "A72")):
        lev = cv.percentile_level(s[var], s[f"Q_{var}"])
        lev = np.array([ordinal.get(float(v), np.nan) for v in lev])
        ax.step(s["time_all"], lev + (i - 1) * 0.12, where="pre", color=colors[var], lw=2, label=var)
    ax.set_yticks(range(0, 6))
    ax.set_yticklabels(["< P75", "P75", "P90", "P95", "P99", "P99.9"])
    ax.set_ylim(-0.4, 5.4)
    ax.legend(loc="upper left", fontsize=8, ncol=3)
    ax.set_title("b) percentil local más alto superado", fontsize=10, loc="left")

    ax = axes[2]
    class_bands(ax)
    for nm in cv.PRECIP_VARIANTS:
        key = f"H_P_cont_{nm}"
        if key in s:
            ax.plot(s["time_all"], s[key], color=EXP_COLOR[nm], ls=EXP_LS[nm], lw=1.2, alpha=0.8,
                    label=f"H_P continuo, máximo {nm} (sin atribución)")
    for e in tags:
        ax.plot(s["time"], s["ITCHI"][e], marker="o", color=EXP_COLOR[e], ls=EXP_LS[e], ms=4, lw=1.8,
                label=f"ITCHI · {cv.label(e)}")
    ax.plot(s["time"], s["H_W"]["ss"] if "ss" in s["H_W"] else s["H_W"][tags[0]], "^", color="0.3", ms=4,
            label="H_W · Saffir-Simpson")
    ax.set_ylim(0, 1)
    ax.set_ylabel("valor municipal (máximo)")
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax.set_title("c) peligro municipal: ITCHI en las posiciones del ciclón y H_P en todos los tiempos", fontsize=10, loc="left")

    ax = axes[3]
    for i, e in enumerate(tags):
        ax.step(s["time"], siat(s["ITCHI"][e]) + (i - 1.5) * 0.08, where="pre", color=EXP_COLOR[e], ls=EXP_LS[e], lw=2,
                label=f"ITCHI · {cv.label(e)}")
    if "H_P_cont_P99" in s:
        ax.step(s["time_all"], siat(s["H_P_cont_P99"]) - 0.25, where="pre", color="0.45", lw=1.2, label="H_P continuo P99")
    ax.set_yticks(range(0, 6))
    ax.set_yticklabels([f"{k} {CLASS_NAMES[k]}" for k in range(6)])
    ax.set_ylim(-0.5, 5.5)
    ax.legend(loc="upper left", fontsize=7.5, ncol=2)
    ax.set_title("d) clase municipal (los puntos de ITCHI solo existen donde hay posición del ciclón)", fontsize=10, loc="left")

    tr = pd.DatetimeIndex(C["track"]["time"])
    gaps = np.flatnonzero(np.diff(tr) > pd.Timedelta(hours=6))
    for ax in axes:
        ax.axvline(te, color="k", lw=1.8)
        if t_decl is not None:
            ax.axvline(pd.Timestamp(t_decl), color="#C0392B", lw=1.2, ls="--")
        for g in gaps:
            ax.axvspan(tr[g], tr[g + 1], color="0.85", alpha=0.6, lw=0, zorder=0)
        if tr.max() < t1:
            ax.axvspan(tr.max(), t1, color="0.85", alpha=0.6, lw=0, zorder=0)
        ax.grid(alpha=0.25)
    axes[0].text(te, 1.0, " deslizamiento", transform=axes[0].get_xaxis_transform(), fontsize=9, va="bottom")
    if t_decl is not None:
        axes[0].text(pd.Timestamp(t_decl), 1.0, " declaratoria", color="#C0392B",
                     transform=axes[0].get_xaxis_transform(), fontsize=8, va="bottom", ha="right")
    axes[-1].set_xlim(t0, t1)
    axes[-1].set_xlabel("tiempo (UTC); gris: sin posición del ciclón en IBTrACS")
    fig.suptitle(f"{name} ({code}) · {C['case']}: peligro antes del deslizamiento", fontsize=12, y=0.995)
    fig.tight_layout()
    return fig


def fig_mapa_deslizamiento(C: dict, code: str, times, tag: str = "v020", mode: str = "poligonos", half_width=5.0):
    """Clase municipal de ITCHI en varios tiempos, con el municipio del deslizamiento resaltado."""
    ds = C["ds"]
    mun = cv.municipal_event(C)
    g = C["gdf"].set_index("CVEGEO").loc[[code]]
    c = g.geometry.iloc[0].representative_point()
    lon_ext = (np.floor(c.x - half_width), np.ceil(c.x + half_width))
    lat_ext = (np.floor(c.y - 0.8 * half_width), np.ceil(c.y + 0.8 * half_width))
    times = [pd.Timestamp(t) for t in times]
    fig, axes = plt.subplots(2, len(times), figsize=(4.4 * len(times), 8.2), subplot_kw={"projection": _pc()}, squeeze=False)
    tindex = pd.DatetimeIndex(ds["time"].values)
    for k, t in enumerate(times):
        j = tindex.get_loc(t)
        for i in range(2):
            ax = _map_axis(axes[i, k], lon_ext, lat_ext, tick=2.0)
            if i == 0:
                draw_pixel_classes(ax, C, ds["ITCHI"].sel(exp=tag).isel(time=j).values)
            else:
                draw_municipal_classes(ax, C, mun.sel(exp=tag).isel(time=j).values, mode=mode)
            g.boundary.plot(ax=ax, color="k", linewidth=1.8, transform=_pc(), zorder=6)
            _draw_track(ax, C)
            ax.plot(float(ds["track_lon"][j]), float(ds["track_lat"][j]), "k*", ms=9, transform=_pc(), zorder=7)
            ax.set_title(f"{'malla' if i == 0 else 'municipio'} · {t:%d %b %H} UTC", fontsize=10)
    class_legend(axes[1, 0], loc="lower left", fontsize=7)
    fig.suptitle(f"{C['case']} · {cv.label(tag)}: clase de ITCHI en las posiciones previas (municipio en negro)", fontsize=11)
    return fig
