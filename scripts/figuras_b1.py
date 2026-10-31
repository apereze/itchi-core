"""
ITCHI-CORE · Figuras del Bloque 1 (notebooks/third test/09_bloque1_experimentos.ipynb)

Parte A, sin datos: qué cambia en cada experimento, con las funciones de la
librería y los percentiles medianos de México (docs/verification_v02.md, tabla 1).

    fig_a1_lluvia_esquemas   H_P contra P6 en cuatro situaciones de R24 y A72
    fig_a2_tablas            clase de H_P en el plano P6 x R24 para tres valores de A72
    fig_a3_viento            H_W contra distancia: cola hasta R_W06, corte 12 m/s, 17 kt

Parte B, con las salidas de experimento_b1.run():

    fig_b1_celdas            P(D) y n por celda P6 x R24 x A72
    fig_b2_candidatos        ΔPSS contra v0.2 y contra M0, y el costo (sesgo, POFD)
    fig_b3_roc               curvas ROC de H_P con los puntos de cada clase
    fig_b4_rejilla           impacto de cada umbral de R24 y A72 (rejilla de M1)
    fig_b5_2x2               diseño 2x2 lluvia x viento
    fig_b6_frecuencias       porcentaje de pares por clase, con y sin declaratoria
    decisiones               criterios de docs/experimento_b1.md, sección 6
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap

try:
    import itchi  # noqa: F401
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from itchi.precipitation import (
    PRECIP_MATRIX_V02,
    PRECIP_TABLES_B1,
    compute_precipitation_hazard_from_spec,
)
from itchi.wind import normalize_wind_hazard_with_tail
from itchi.wind_profiles import KNOT_TO_MS

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experimento_b1 as b1  # noqa: E402

# Percentiles medianos sobre México, mm por ventana (verification_v02.md, tabla 1)
Q_MEX = {
    6: {"Q75": 9.8, "Q90": 20.1, "Q95": 29.3, "Q99": 52.5, "Q99.9": 84.3},
    24: {"Q75": 15.0, "Q90": 31.7, "Q95": 47.0, "Q99": 87.9, "Q99.9": 142.7},
    72: {"Q75": 24.4, "Q90": 48.8, "Q95": 70.3, "Q99": 126.9, "Q99.9": 203.1},
}
SCHEMES_A = {"v0.2": PRECIP_MATRIX_V02, "M0": PRECIP_TABLES_B1["M0"],
             "M1": PRECIP_TABLES_B1["M1"], "M2": PRECIP_TABLES_B1["M2"]}
MAIN = ["v0.2", "M0", "M1", "M2", "M3 (VC por año)"]
COL = {"v0.2": "#2A6FC9", "M0": "#6B7280", "M1": "#D9822B", "M2": "#2A9D8F", "M3 (VC por año)": "#8E5CC2",
       "W0 cola hasta R_W06": "#2A6FC9", "W1 corte 12 m/s": "#D9822B", "ref corte 17 kt": "#2A9D8F",
       "sin viento": "#6B7280"}
LS = {"v0.2": "-", "M0": "--", "M1": "-", "M2": "-.", "M3 (VC por año)": ":"}
SIAT_COLORS = ["#F3F4F6", "#4C9BE8", "#4CB050", "#F2D23C", "#F08C2E", "#D7301F"]
SIAT_LABELS = ["0", "1", "2", "3", "4", "5"]
CLASES = ["<P75", "P75-P90", "P90-P95", ">=P95"]

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def _hp(spec, p6, r24, a72):
    return compute_precipitation_hazard_from_spec(
        p6, Q_MEX[6], {"R24": r24, "A72": a72}, {"R24": Q_MEX[24], "A72": Q_MEX[72]}, spec)


def _percentile_lines(ax, q, axis="x", labels=True):
    for k in ("Q75", "Q90", "Q95", "Q99"):
        if k not in q:
            continue
        line = ax.axvline if axis == "x" else ax.axhline
        line(q[k], color="0.75", lw=0.8, ls=":", zorder=0)
        if labels and axis == "x":
            ax.text(q[k], 1.02, f"P{k[1:]}", transform=ax.get_xaxis_transform(), ha="center",
                    va="bottom", fontsize=8, color="0.4")


# ============================================================================
# PARTE A
# ============================================================================


def fig_a1_lluvia_esquemas():
    """H_P contra P6 para v0.2, M0, M1 y M2 en cuatro situaciones de R24 y A72."""

    p6 = np.linspace(0.0, 70.0, 701)
    q24, q72 = Q_MEX[24], Q_MEX[72]
    cases = [
        ("Lluvia aislada\nR24 = P6, A72 = 0", p6, np.zeros_like(p6)),
        (f"Persistente, suelo seco\nR24 ≥ P95₂₄ ({q24['Q95']:.0f} mm), A72 = 0",
         np.maximum(p6, q24["Q95"]), np.zeros_like(p6)),
        (f"Persistente, suelo húmedo\nR24 ≥ P95₂₄, A72 = P75₇₂ ({q72['Q75']:.0f} mm)",
         np.maximum(p6, q24["Q95"]), np.full_like(p6, q72["Q75"])),
        (f"Suelo muy húmedo, sin persistencia\nR24 = P6, A72 = P95₇₂ ({q72['Q95']:.0f} mm)",
         p6, np.full_like(p6, q72["Q95"])),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.9), sharey=True)
    for ax, (title, r24, a72) in zip(axes, cases, strict=True):
        for y in (0.2, 0.4, 0.6, 0.8):
            ax.axhline(y, color="0.92", lw=0.8, zorder=0)
        _percentile_lines(ax, Q_MEX[6])
        for name, spec in SCHEMES_A.items():
            ax.plot(p6, _hp(spec, p6, r24, a72), LS[name], color=COL[name], lw=2, label=name)
        ax.set(xlabel="P6 (mm en 6 h)", xlim=(0, 70), ylim=(-0.02, 1.05))
        ax.set_title(title, fontsize=9.5, pad=16)
    axes[0].set_ylabel("H_P")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Peligro por lluvia de cada esquema (percentiles medianos de México); "
                 "líneas grises: percentiles de 6 h", y=1.08)
    fig.tight_layout()
    return fig


def fig_a2_tablas():
    """Clase SIAT de H_P en el plano P6 x R24, para A72 = 0, P75 y P95 (v0.2, M1, M2)."""

    p6 = np.linspace(0.0, 60.0, 241)
    r24 = np.linspace(0.0, 100.0, 241)
    P, R = np.meshgrid(p6, r24)
    a72_vals = [("A72 = 0", 0.0), ("A72 = P75₇₂", Q_MEX[72]["Q75"]), ("A72 = P95₇₂", Q_MEX[72]["Q95"])]
    cmap = ListedColormap(SIAT_COLORS)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], cmap.N)
    names = ["v0.2", "M1", "M2"]
    fig, axes = plt.subplots(len(names), 3, figsize=(12, 10.5), sharex=True, sharey=True)
    for i, name in enumerate(names):
        for j, (lab, a72) in enumerate(a72_vals):
            ax = axes[i, j]
            cl = b1.siat_class(_hp(SCHEMES_A[name], P, R, np.full_like(P, a72))).astype(float)
            cl[R < P] = np.nan                                   # R24 incluye P6
            ax.pcolormesh(P, R, cl, cmap=cmap, norm=norm, shading="auto")
            ax.fill_between(p6, 0, p6, color="0.85", hatch="//", lw=0)
            for k in ("Q75", "Q90", "Q95", "Q99"):
                ax.axvline(Q_MEX[6][k], color="k", lw=0.5, ls=":")
                ax.axhline(Q_MEX[24][k], color="k", lw=0.5, ls=":")
            ax.set_title(f"{name} · {lab}", fontsize=10)
            if i == len(names) - 1:
                ax.set_xlabel("P6 (mm)")
                ax.set_xticks([Q_MEX[6][k] for k in ("Q75", "Q90", "Q95", "Q99")],
                              ["P75", "P90", "P95", "P99"])
            if j == 0:
                ax.set_ylabel("R24 (mm)")
                ax.set_yticks([Q_MEX[24][k] for k in ("Q75", "Q90", "Q95", "Q99")],
                              ["P75₂₄", "P90₂₄", "P95₂₄", "P99₂₄"])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SIAT_COLORS]
    fig.legend(handles, [f"clase {c}" for c in SIAT_LABELS], loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Clase de peligro por lluvia según P6, R24 y A72 (gris: imposible, R24 < P6)", y=1.0)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig


def fig_a3_viento():
    """H_W contra distancia para W0 (2 m/s), W1 (12 m/s) y 17 kt; huracán y tormenta."""

    r = np.linspace(0.0, 1000.0, 2001)
    storms = [("Huracán, Vmax = 90 kt, 20° N", 90.0), ("Tormenta tropical, Vmax = 45 kt, 20° N", 45.0)]
    cfg = b1.ConfigB1()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    for ax, (title, vmax) in zip(axes, storms, strict=True):
        v = b1.wind_kt_w06(np.full(r.size, vmax), np.full(r.size, 20.0), r)
        outer = r > r[np.argmax(v)]
        r34 = r[outer & (v <= 34.0)][0] if (outer & (v <= 34.0)).any() else 0.0
        for name, v_out in cfg.viento.items():
            if v_out is None:
                continue
            # región directa radial (r <= R34), como en el pipeline
            h = normalize_wind_hazard_with_tail(v, vmax, r <= r34, max(v_out, b1.V_OUTER_MS) / KNOT_TO_MS,
                                                h34=cfg.h34)
            ax.plot(r, h, color=COL[name], lw=2.2, label=name)
        marks = [("R34", 34.0), ("12 m/s", 12.0 / KNOT_TO_MS), ("17 kt", 17.0), ("R_W06 (2 m/s)", 2.0 / KNOT_TO_MS)]
        for lab, thr in marks:
            idx = np.nonzero(v <= thr)[0]
            idx = idx[r[idx] > r[np.argmax(v)]]
            if idx.size:
                ax.axvline(r[idx[0]], color="0.6", lw=0.8, ls=":")
                ax.text(r[idx[0]], 1.02, lab, transform=ax.get_xaxis_transform(), ha="center",
                        fontsize=8, color="0.35")
        tw = ax.twinx()
        tw.plot(r, v * KNOT_TO_MS, color="0.55", lw=1, ls="--")
        tw.set_ylabel("V del perfil W06 (m/s, gris)", color="0.45")
        tw.spines["top"].set_visible(False)
        ax.set(xlabel="distancia al centro (km)", ylabel="H_W", ylim=(0, 1.05), xlim=(0, 1000))
        ax.set_title(title, pad=18)
    axes[0].legend(frameon=False, fontsize=9, loc="center right")
    fig.suptitle("Cola del peligro por viento: dónde termina en cada variante", y=1.04)
    fig.tight_layout()
    return fig


# ============================================================================
# PARTE B
# ============================================================================


def fig_b1_celdas(celdas: pd.DataFrame, n_min: int = 30):
    """P(D) % por celda; texto gris si n < n_min."""

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.3), sharey=True)
    ok = celdas["n"].to_numpy(float) >= n_min
    pmax = np.nanmax(np.where(ok, celdas["P(D) %"].to_numpy(float), np.nan))
    for ax, a72 in zip(axes, CLASES, strict=True):
        sub = celdas.xs(a72, level="A72")
        mat = sub["P(D) %"].unstack("R24").reindex(index=CLASES, columns=CLASES)
        nn = sub["n"].unstack("R24").reindex(index=CLASES, columns=CLASES)
        im = ax.imshow(mat.where(nn >= n_min).to_numpy(float), cmap="Reds", vmin=0, vmax=pmax, origin="lower")
        for i in range(4):
            for j in range(4):
                p, n = mat.iat[i, j], nn.iat[i, j]
                if pd.notna(n):
                    ax.text(j, i, f"{p:.1f} %\n({int(n):,})", ha="center", va="center", fontsize=7.5,
                            color=("0.5" if n < n_min else ("w" if p > 0.6 * pmax else "k")))
        ax.set_xticks(range(4), CLASES, rotation=30)
        ax.set_yticks(range(4), CLASES)
        ax.set(title=f"A72 {a72}", xlabel="R24")
    axes[0].set_ylabel("P6")
    fig.colorbar(im, ax=axes, shrink=0.8, label="P(declaratoria) %")
    fig.suptitle(f"Probabilidad de declaratoria por combinación de umbrales, en el pico de P6 del par "
                 f"(celdas sin color: n < {n_min}, no se interpretan)", y=0.98)
    return fig


def _bar_ci(ax, x, df, col, color, width, label=None):
    y = df[col].to_numpy(float)
    lo, hi = df[f"{col} lo"].to_numpy(float), df[f"{col} hi"].to_numpy(float)
    ax.bar(x, y, width, color=color, label=label)
    ax.errorbar(x, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], fmt="none", ecolor="k",
                elinewidth=1, capsize=3)


def fig_b2_candidatos(lluvia: pd.DataFrame):
    """ΔPSS (IC 90 %) contra v0.2 y contra M0, y el costo en alertas."""

    t = lluvia.set_index("lluvia").reindex(MAIN)
    x = np.arange(len(MAIN))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.3))
    for i, (k, c) in enumerate(zip((1, 3, 5), ("#9CA3AF", "#6B7280", "#111827"), strict=True)):
        _bar_ci(axes[0], x + (i - 1) * 0.26, t, f"dPSS>={k}", c, 0.25, label=f"clase ≥ {k}")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set(title="ΔPSS contra v0.2 (lluvia sola)", ylabel="ΔPSS")
    axes[0].set_xticks(x, MAIN, rotation=20)
    axes[0].legend(frameon=False, fontsize=8)

    sub = t.drop(index=["v0.2", "M0"])
    xs = np.arange(len(sub))
    for i, (k, c) in enumerate(zip((3, 5), ("#6B7280", "#111827"), strict=True)):
        _bar_ci(axes[1], xs + (i - 0.5) * 0.36, sub, f"dPSS>={k} vs M0", c, 0.34, label=f"clase ≥ {k}")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set(title="Aporte de R24 y A72: ΔPSS contra M0 (solo P6)", ylabel="ΔPSS")
    axes[1].set_xticks(xs, sub.index, rotation=20)
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].bar(x - 0.2, t["sesgo>=5"], 0.38, color=[COL[n] for n in MAIN], label="sesgo de frecuencia")
    ax2 = axes[2].twinx()
    ax2.plot(x + 0.2, 100 * t["POFD>=5"], "kD", label="POFD %")
    ax2.set_ylabel("POFD clase 5 (%)")
    ax2.spines["top"].set_visible(False)
    axes[2].set(title="Costo: alertas en clase 5", ylabel="sesgo (alertas / eventos)")
    axes[2].set_xticks(x, MAIN, rotation=20)
    fig.tight_layout()
    return fig


def fig_b3_roc(por_par: pd.DataFrame, names=MAIN, zoom=0.35):
    """ROC de H_P (lluvia sola) con los puntos de las clases 1 a 5."""

    d = por_par["D"].to_numpy(bool)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, lim in zip(axes, (1.0, zoom), strict=True):
        ax.plot([0, lim], [0, lim], color="0.8", lw=0.8)
        for name in names:
            s = por_par[f"HP|{name}"].to_numpy(float)
            f, h = b1.roc_curve(s, d)
            ax.plot(f, h, LS.get(name, "-"), color=COL.get(name, "k"), lw=1.8,
                    label=f"{name} (AUC {b1.auc(s, d):.3f})")
            cl = b1.siat_class(s)
            for k in range(1, 6):
                c = b1.contingency(cl >= k, d)
                ax.plot(c["POFD"], c["POD"], "o", color=COL.get(name, "k"), ms=4)
                if name == names[0] or name == "M1":
                    ax.annotate(str(k), (c["POFD"], c["POD"]), textcoords="offset points", xytext=(4, -9),
                                fontsize=7, color=COL.get(name, "k"))
        ax.set(xlim=(0, lim), ylim=(0, 1), xlabel="POFD (falsas alarmas / no declarados)",
               ylabel="POD (declarados alertados)")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_title("Curva ROC de la lluvia (números: alerta si clase ≥ k)")
    axes[1].set_title(f"Detalle POFD ≤ {zoom}")
    fig.tight_layout()
    return fig


def fig_b4_rejilla(lluvia: pd.DataFrame, k: int = 5):
    """ΔPSS contra M0 de cada combinación de umbrales de la regla M1."""

    t = lluvia.set_index("lluvia")
    rows, cols = ["P90", "P95"], ["P75", "P90", "P95"]
    val = np.full((2, 3), np.nan)
    txt = [["" for _ in cols] for _ in rows]
    for i, r in enumerate(rows):
        for j, a in enumerate(cols):
            name = "M1" if (r, a) == ("P95", "P75") else f"M1[R24>={r}, A72>={a}]"
            if name in t.index:
                val[i, j] = t.at[name, f"dPSS>={k} vs M0"]
                txt[i][j] = (f"{val[i, j]:+.3f}\n[{t.at[name, f'dPSS>={k} vs M0 lo']:+.3f}, "
                             f"{t.at[name, f'dPSS>={k} vs M0 hi']:+.3f}]")
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [3, 1.4]})
    lim = np.nanmax(np.abs(val)) if np.isfinite(val).any() else 0.01
    im = axes[0].imshow(val, cmap="RdBu", vmin=-lim, vmax=lim)
    for i in range(2):
        for j in range(3):
            axes[0].text(j, i, txt[i][j], ha="center", va="center", fontsize=8)
    axes[0].set_xticks(range(3), [f"A72 ≥ {a}" for a in cols])
    axes[0].set_yticks(range(2), [f"R24 ≥ {r}" for r in rows])
    axes[0].set_title(f"Regla M1 «P90 es máximo si...»: ΔPSS clase ≥ {k} contra M0 [IC 90 %]")
    fig.colorbar(im, ax=axes[0], shrink=0.8)
    solo = t.reindex(["M1[solo R24>=P95]", "M1[solo A72>=P75]", "M1"])
    y = np.arange(len(solo))
    axes[1].barh(y, solo[f"dPSS>={k} vs M0"], color=["#D9822B", "#2A9D8F", "#111827"])
    axes[1].errorbar(solo[f"dPSS>={k} vs M0"], y,
                     xerr=[solo[f"dPSS>={k} vs M0"] - solo[f"dPSS>={k} vs M0 lo"],
                           solo[f"dPSS>={k} vs M0 hi"] - solo[f"dPSS>={k} vs M0"]],
                     fmt="none", ecolor="k", capsize=3)
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].set_yticks(y, ["solo R24 ≥ P95", "solo A72 ≥ P75", "ambos (M1)"])
    axes[1].set(title="Ablación", xlabel="ΔPSS contra M0")
    fig.tight_layout()
    return fig


def fig_b5_2x2(diseno: pd.DataFrame, lluvias=("v0.2", "M1"),
               vientos=("W0 cola hasta R_W06", "W1 corte 12 m/s", "ref corte 17 kt")):
    """PSS por clase y AUC de cada celda lluvia x viento (barras: IC 90 % de ΔPSS)."""

    t = diseno.set_index(["lluvia", "viento"])
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    x = np.arange(len(vientos))
    for ax, k in zip(axes[:3], (1, 3, 5), strict=True):
        for name in lluvias:
            sub = t.loc[name].reindex(list(vientos))
            y = sub[f"PSS>={k}"].to_numpy(float)
            base = y - sub[f"dPSS>={k}"].to_numpy(float)
            lo, hi = base + sub[f"dPSS>={k} lo"].to_numpy(float), base + sub[f"dPSS>={k} hi"].to_numpy(float)
            ax.errorbar(x, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], marker="o", ms=7,
                        color=COL[name], ls=LS[name], lw=2, capsize=4, label=name)
        ax.set_xticks(x, [v.split(" ", 1)[1] if " " in v else v for v in vientos], rotation=15)
        ax.set(title=f"PSS, alerta si clase ≥ {k}", ylabel="PSS")
    for name in lluvias:
        sub = t.loc[name].reindex(list(vientos))
        axes[3].plot(x, sub["AUC_anual"], marker="o", ms=7, color=COL[name], ls=LS[name], lw=2, label=name)
    if ("v0.2", "sin viento") in t.index:
        axes[3].axhline(t.at[("v0.2", "sin viento"), "AUC_anual"], color="0.6", ls=":", label="v0.2 sin viento")
    axes[3].set_xticks(x, [v.split(" ", 1)[1] if " " in v else v for v in vientos], rotation=15)
    axes[3].set(title="AUC promedio por año", ylabel="AUC")
    axes[0].legend(frameon=False, fontsize=9)
    axes[3].legend(frameon=False, fontsize=8)
    fig.suptitle("Diseño 2 × 2: lluvia (v0.2 / M1) × fin de la cola de viento (R_W06 / 12 m/s; 17 kt de referencia). "
                 "Líneas paralelas = efectos aditivos", y=1.04)
    fig.tight_layout()
    return fig


def fig_b6_frecuencias(por_par: pd.DataFrame, configs=(("v0.2", "W0 cola hasta R_W06"),
                                                      ("v0.2", "W1 corte 12 m/s"),
                                                      ("M1", "W0 cola hasta R_W06"),
                                                      ("M1", "W1 corte 12 m/s"))):
    """Porcentaje de pares en cada clase SIAT, sin y con declaratoria."""

    d = por_par["D"].to_numpy(bool)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), sharey=True)
    labels = [f"{p}\n{w.split(' ', 1)[1]}" for p, w in configs]
    for ax, mask, title in ((axes[0], ~d, "Pares sin declaratoria"), (axes[1], d, "Pares con declaratoria")):
        bottom = np.zeros(len(configs))
        for c in range(6):
            frac = np.array([100 * np.mean(b1.siat_class(por_par.loc[mask, f"ITCHI|{p}|{w}"].to_numpy(float)) == c)
                             for p, w in configs])
            ax.bar(range(len(configs)), frac, bottom=bottom, color=SIAT_COLORS[c], label=f"clase {c}",
                   edgecolor="w", lw=0.5)
            bottom += frac
        ax.set_xticks(range(len(configs)), labels, fontsize=8)
        ax.set(title=title, ylim=(0, 100))
    axes[0].set_ylabel("% de pares")
    axes[1].legend(frameon=False, fontsize=8, bbox_to_anchor=(1.0, 1.0), loc="upper left")
    fig.suptitle("Clase máxima de ITCHI durante el evento: lo deseable es más clase alta a la derecha "
                 "y menos a la izquierda", y=1.03)
    fig.tight_layout()
    return fig


# ============================================================================
# CRITERIOS DE DECISIÓN (docs/experimento_b1.md, sección 6)
# ============================================================================


def decisiones(lluvia: pd.DataFrame, diseno: pd.DataFrame, tol_pod: float = 0.005) -> pd.DataFrame:
    """Aplica los criterios fijados antes de ver los resultados."""

    L = lluvia.set_index("lluvia")
    D = diseno.set_index(["lluvia", "viento"])
    out = []

    m0 = L.loc["M0"]
    dpod = m0["dPOD@POFD>=5"]
    if m0["dPSS>=5 lo"] > 0 and dpod >= -tol_pod:
        v = "se adopta por discriminación" if dpod > tol_pod else "solo desplaza el umbral: decisión operativa"
    else:
        v = "no se adopta"
    out.append({"criterio": "1. P95 como máximo (M0 contra v0.2, clase 5)",
                "evidencia": f"ΔPSS {m0['dPSS>=5']:+.3f} {m0['dPSS>=5 IC90']}; ΔPOD@POFD {dpod:+.3f}; "
                             f"sesgo {L.at['v0.2', 'sesgo>=5']:.1f} → {m0['sesgo>=5']:.1f}",
                "resultado": v})

    chosen = None
    ev = []
    for name in ("M1", "M2", "M3 (VC por año)"):
        r = L.loc[name]
        ev.append(f"{name}: {r['dPSS>=5 vs M0']:+.3f} {r['dPSS>=5 vs M0 IC90']}")
        if chosen is None and r["dPSS>=5 vs M0 lo"] > 0:
            chosen = name
    out.append({"criterio": "2. Umbrales de R24 y A72 (contra M0, clase 5; el más simple que pase)",
                "evidencia": "; ".join(ev),
                "resultado": f"se adopta {chosen}" if chosen else "R24 y A72 no aportan con P95 como máximo"})

    w0, w1 = D.loc[("v0.2", "W0 cola hasta R_W06")], D.loc[("v0.2", "W1 corte 12 m/s")]
    w17, sv = D.loc[("v0.2", "ref corte 17 kt")], D.loc[("v0.2", "sin viento")]
    ok = w1["dPSS>=1 lo"] > 0 and w1["AUC_anual"] > sv["AUC_anual"]
    domina = w17["PSS>=1"] >= w1["PSS>=1"] and w17["AUC_anual"] >= w1["AUC_anual"]
    v = "se adopta" if ok else "no se adopta"
    if ok and domina:
        v += "; 17 kt domina: 12 m/s necesita argumento físico"
    out.append({"criterio": "3. Corte de la cola en 12 m/s (contra W0, lluvia v0.2)",
                "evidencia": f"ΔPSS≥1 {w1['dPSS>=1']:+.3f} {w1['dPSS>=1 IC90']}; AUC {w1['AUC_anual']:.3f} "
                             f"(sin viento {sv['AUC_anual']:.3f}; 17 kt {w17['AUC_anual']:.3f}, "
                             f"PSS≥1 {w17['PSS>=1']:.3f})",
                "resultado": v})

    p1w0, p1w1 = D.loc[("M1", "W0 cola hasta R_W06")], D.loc[("M1", "W1 corte 12 m/s")]
    ev, aditivo = [], True
    for k in (1, 5):
        inter = (p1w1[f"PSS>={k}"] - w1[f"PSS>={k}"]) - (p1w0[f"PSS>={k}"] - w0[f"PSS>={k}"])
        half = max(p1w0[f"dPSS>={k} hi"] - p1w0[f"dPSS>={k} lo"], w1[f"dPSS>={k} hi"] - w1[f"dPSS>={k} lo"]) / 2
        aditivo &= abs(inter) <= max(half, 1e-9)
        ev.append(f"clase ≥ {k}: {inter:+.3f} (semiancho IC {half:.3f})")
    out.append({"criterio": "4. Interacción lluvia × viento",
                "evidencia": "; ".join(ev),
                "resultado": "efectos aditivos: se deciden por separado" if aditivo
                else "hay interacción: decidir P y W juntos"})
    return pd.DataFrame(out)
