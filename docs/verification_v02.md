# ITCHI v0.2 verification against disaster declarations

This document summarizes how ITCHI v0.2 was calibrated and evaluated, what the
results are, what they do not show yet, and which decisions are open for v0.3.
The formulas are in [`methodology.md`](methodology.md) ("ITCHI v0.2
Extensions"); the declaration processing is in [`declaratorias.md`](declaratorias.md).

All numbers below come from the scripts and notebooks listed in
[Reproduction](#reproduction). None of the data or derived products are
versioned (see `.gitignore`).

---

## 1. Data

| Source | Use |
|---|---|
| MSWEP V3.16, 0.1°, hourly → 6-h accumulations | P6, R24 and A72 at each synoptic time |
| Local percentiles of 6, 24 and 72 h (May–November, 2000–2025, wet windows ≥ 1 mm, `p_wet_raw`) | Thresholds P75, P90, P95, P99, P99.9 of each pixel |
| IBTrACS v04r01 (NA, EP) | Track, Vmax, R34 by quadrant, RMW |
| W06 profile with the PA21 parameterization | Attribution radius R_W06 = radius where V = 2 m/s |
| CENAPRED declarations filtered to tropical cyclones | Event label D (emergency, disaster or climatological contingency) |
| INEGI municipalities 2023 (`mun23gw`) | Pixels of each municipality |

Median percentiles over Mexico (mm per wet window):

| Duration | P75 | P90 | P95 | P99 | P99.9 | Wet frequency |
|---|--:|--:|--:|--:|--:|--:|
| 6 h (P6) | 9.8 | 20.1 | 29.3 | 52.5 | 84.3 | 0.08 |
| 24 h (R24) | 15.0 | 31.7 | 47.0 | 87.9 | 142.7 | 0.21 |
| 72 h (A72) | 24.4 | 48.8 | 70.3 | 126.9 | 203.1 | 0.41 |

Rolling 24-h and 72-h windows overlap; the effective sample size is n/(d/6)
and the sufficiency filter uses it. P99 and P99.9 of 72 h are not robust and
are not used.

Rain windows at a synoptic time T: P6 = (T − 6 h, T] (intensity),
R24 = (T − 24 h, T] (persistence, includes P6), A72 = (T − 78 h, T − 6 h]
(antecedent wetness, excludes P6).

---

## 2. Verification design

- **Unit:** municipality–cyclone pair. A pair exists if any pixel of the
  municipality is inside R_W06 of the cyclone at some synoptic time.
  381 929 pairs, 2000–2025; median R_W06 = 743 km.
- **Label:** D = 1 if the municipality had a declaration associated with that
  cyclone: 7 462 pairs (base rate 2.0 %). Declarations are associated with
  cyclones by date.
- **Rain:** peak 6-h rain of the municipality during the cyclone; P6, R24 and
  A72 are classified against the pixel's own percentiles at that peak.
- **Alert:** rain level ≥ k (k = 1…4), or SIAT class ≥ k for full ITCHI.

### Metrics

From the 2 × 2 table (a hits, b false alarms, c misses, d correct negatives):

| Metric | Formula | Question it answers | Depends on base rate |
|---|---|---|---|
| POD | a / (a + c) | Of the pairs with a declaration, how many were alerted? | No |
| POFD | b / (b + d) | Of the pairs without a declaration, how many were alerted? | No |
| PSS (Peirce, Hanssen–Kuipers, TSS) | POD − POFD | How well does the threshold separate both groups? 0 = no skill | No |
| AUC | area under POD vs POFD | Probability that a pair with a declaration has a higher level than one without | No |
| FAR | b / (a + b) | Of the alerts, how many had no declaration? | Yes |
| CSI | a / (a + b + c) | Hits among everything that was an alert or an event | Yes |
| Frequency bias | (a + b) / (a + c) | Alerts per observed event | — |

PSS and AUC are used to compare schemes because only 2 % of the pairs are
events and declarations are under-reported. FAR is fixed mostly by the base
rate: FAR = 1 / [1 + (s / (1 − s)) · POD / POFD]. AUC is averaged by year
because the base rate ranges from 0.2 % to 9 % between years.

**Caveat for rare events.** With a 2 % base rate the PSS rewards
over-forecasting (POFD barely penalizes extra alerts). Base-rate-dependent
scores move the other way (see 3.4). Both are reported.

---

## 3. Results: precipitation

### 3.1 Rain at which declarations are issued (positives only)

| Type | n | < 1 mm | < P75 | ≥ P95 | ≥ P99 |
|---|--:|--:|--:|--:|--:|
| Emergency | 5 794 | 18.0 % | 37.5 % | 46.8 % | 28.5 % |
| Disaster | 4 388 | 0.9 % | 3.8 % | 85.0 % | 59.0 % |
| Climatological contingency | 255 | 0.0 % | 1.2 % | 89.4 % | 74.1 % |

P75 covers almost all disaster declarations, so it is used as the early-alert
threshold. 18 % of emergencies were issued with < 1 mm (preventive or other
hazards); no rain threshold can capture them.

### 3.2 Probability of a declaration by cell (P6 class × A72 group)

| P6 \ A72 | < P75 | P75–P95 | ≥ P95 |
|---|--:|--:|--:|
| < P75 | 0.5 % | 0.4 % | 0.8 % |
| P75–P90 | 0.9 % | 1.1 % | 1.1 % |
| P90–P95 | 1.5 % | 1.9 % | 2.6 % |
| P95–P99 | 2.9 % | 5.8 % | 8.9 % |
| P99–P99.9 | 7.1 % | 14.2 % | 24.0 % |
| ≥ P99.9 | 11.2 % | 24.1 % | 40.1 % |

Antecedent rain matters only for P6 ≥ P95.

### 3.3 Persistence inside P95–P99

| R24 \ A72 | < P75 | P75–P95 | ≥ P95 |
|---|--:|--:|--:|
| R24 < P95 | 2.1 % | 1.6 % | 1.7 % |
| R24 ≥ P95 | 4.0 % | 7.4 % | 10.3 % |

Without persistence the antecedent adds nothing; with persistence it
multiplies the probability. This gives the v0.2 rule:
**P6 in P95–P99 with R24 ≥ P95 and A72 ≥ P75 → maximum level.**

### 3.4 Schemes compared on the same pairs

| Scheme | AUC/year | PSS ≥ 1 | PSS ≥ 2 | PSS ≥ 3 | PSS ≥ 4 | POD ≥ 4 | POFD ≥ 4 | PSS ≥ 4, disasters |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| No escalation (P6 only) | 0.864 | 0.446 | 0.533 | 0.547 | 0.413 | 0.465 | 0.052 | 0.550 |
| Maximum of 6/24/72-h classes | 0.864 | 0.376 | 0.500 | 0.538 | 0.460 | 0.523 | 0.062 | 0.613 |
| Escalate if A72 ≥ P95 | 0.864 | 0.446 | 0.533 | 0.547 | 0.447 | 0.507 | 0.061 | 0.589 |
| Escalate if R24 ≥ P99 or A72 ≥ P95 | 0.864 | 0.446 | 0.533 | 0.547 | 0.469 | 0.534 | 0.066 | 0.619 |
| **v0.2: R24 ≥ P95 and A72 ≥ P75** | 0.862 | 0.446 | 0.533 | 0.547 | **0.514** | **0.596** | 0.082 | **0.670** |

The escalation changes only the maximum level. Contingency table of v0.2 at
level ≥ 4: a = 4 444, b = 30 539, c = 3 018, d = 343 928.

All scores at level ≥ 4:

| Score | No escalation | v0.2 |
|---|--:|--:|
| POD | 0.465 | 0.596 |
| POFD | 0.052 | 0.082 |
| PSS | 0.413 | 0.514 |
| SEDI | 0.627 | 0.699 |
| FAR | 0.849 | 0.873 |
| CSI | 0.129 | 0.117 |
| ETS | 0.114 | 0.101 |
| HSS | 0.205 | 0.183 |
| Frequency bias | 3.07 | 4.69 |

v0.2 trades false alarms for detection: PSS and SEDI (Ferro and Stephenson,
2011) improve; CSI, ETS and HSS, which penalize each false alarm and depend on
the base rate, decrease. A more frequent maximum was chosen on purpose because
declarations under-report impacts; this is a choice of purpose, not a
statistical result.

**Extending the escalation to P75–P95** (one level up when R24 ≥ P95 and
A72 ≥ P75) changes PSS ≥ 3 by +0.006 (90 % bootstrap interval over cyclones:
+0.003 to +0.009). Using A72 alone lowers PSS ≥ 2 from 0.533 to 0.518. Not
adopted.

**AUC by year** ranges 0.79–0.95, except 2000 (0.46), most likely a
declaration–cyclone association problem.

---

## 4. Results: wind outside R34

Approximate evaluation on the pairs (H_P by class, W06 wind at the peak-rain
step, H = 1 − (1 − H_W)(1 − H_P), SIAT classes of 0.2):

| Variant | AUC | Pairs with wind > 0 | PSS ≥ 1 | PSS ≥ 3 | PSS ≥ 5 | POFD ≥ 1 |
|---|--:|--:|--:|--:|--:|--:|
| Rain only | 0.821 | 0 % | 0.446 | 0.547 | 0.514 | 0.411 |
| Tail to R_W06, h34 0.2 (**v0.2 default**) | 0.856 | 96 % | **0.028** | 0.578 | 0.537 | 0.972 |
| Tail to 17 kt, h34 0.2 | 0.848 | 10 % | 0.490 | 0.577 | 0.537 | 0.439 |
| Tail to 17 kt, h34 0.3 | 0.850 | 10 % | 0.490 | 0.579 | 0.549 | 0.439 |
| Tail to 24 kt, h34 0.2 | 0.833 | 4 % | 0.460 | 0.569 | 0.537 | 0.419 |

Wind outside R34 is informative (AUC 0.82 → 0.85–0.86), but the default tail
gives class 1 to almost every pair, so class 1 stops discriminating. Cutting
the tail at 17 kt keeps the information and restores class 1. To be confirmed
with full ITCHI.

---

## 5. Case studies: Cristóbal 2020 and Odile 2014

Only v0.1 and v0.2 are compared, with the same attribution radius (W06 PA21).

| | Cristóbal 2020 | Odile 2014 |
|---|---|---|
| Landfall in Mexico | 3 Jun, Campeche, 50 kt | 15 Sep, Cabo San Lucas, 110 kt |
| Snapshots | 43 | 35 |
| Municipalities in the zone | 1 679 | 1 660 |
| With a declaration | 125 (7.4 %) | 6 (0.4 %) |

Event-level municipal PSS (class of ITCHI_max):

| Alert if class ≥ | 1 | 2 | 3 | 4 | 5 |
|---|--:|--:|--:|--:|--:|
| Cristóbal v0.1 | 0.852 | 0.864 | 0.871 | 0.881 | 0.899 |
| Cristóbal v0.2 | 0.000 | 0.835 | 0.864 | 0.880 | 0.888 |
| Odile v0.1 | 0.571 | 0.628 | 0.688 | 0.777 | 0.856 |
| Odile v0.2 | 0.000 | 0.526 | 0.646 | 0.733 | 0.816 |

- POD = 1 in both versions and both cases: every municipality with a
  declaration reaches a high class at some time, so only POFD can change.
  The cases illustrate behaviour; they cannot rank versions.
- Attribution by step (one change at a time, notebook 07): the wind tail
  explains the class-1 drop (−0.73 Cristóbal, −0.39 Odile); the P75 early
  alert lowers classes 1–2 (−0.12/−0.03 Cristóbal, −0.18/−0.09 Odile); the
  escalation lowers classes 4–5 only in Odile (−0.04, with n = 6); W06 inside
  R34 does not change the PSS.
- Duration of the alert discriminates better than its peak: AUC of hours in
  class ≥ 3 is 0.979 vs 0.944 for the maximum class (Cristóbal, v0.2); 0.957 vs
  0.908 (Odile).
- Timing of the first alert relative to the municipal rain peak is
  inconsistent (Cristóbal: median 42 h before; Odile: 18 h after) and must be
  checked before it is reported.

---

## 6. Open decisions for v0.3

| # | Decision | Options | Evidence | Proposal |
|---|---|---|---|---|
| D1 | Extent of the wind tail | to R_W06 (default) · to 17 kt · none | PSS ≥ 1: 0.03 · 0.49 · 0.45; AUC 0.86 · 0.85 · 0.82 | Cut at 17 kt |
| D2 | Hazard at R34 (h34) | 0.2 (default) · 0.3 | PSS class 5: 0.537 · 0.549 | 0.3, confirm with full ITCHI |
| D3 | Escalate P75–P95 too | P95–P99 only · from P75 | ΔPSS ≥ 3 = +0.006 | Keep current |
| D4 | Is ≥ P99 always the maximum? | yes · separate brief and persistent rain | P(D) in P99–P99.9: 7 % (dry) to 24 % (A72 ≥ P95) | Evaluate |
| D5 | Declaration–cyclone association | by date (current) · by storm name | 29.8 % of declared pairs outside the radius; AUC 2000 = 0.46 | By name, before publishing |
| D6 | Validation of the rule | none · cross-validation by year + bootstrap by cyclone | Rule chosen on the same pairs | Validate before fixing v0.3 |
| D7 | Event product | ITCHI_max · ITCHI_max + hours in alert | AUC 0.98 vs 0.94 | Add hours in alert |
| D8 | Soil wetness | A72 (rain) · soil moisture (e.g. ERA5-Land) | A72 is a proxy | Test soil moisture |
| D9 (candidate) | Wind normalization | relative to the storm's own Vmax (current) · fixed maximum (e.g. 137 kt) | A 40-kt storm reaches H_W = 1 at its RMW | Discuss |

---

## 7. Limitations

- Declarations are associated with cyclones by date: 3 170 declared pairs
  (29.8 %) never enter the radius of the assigned cyclone (median distance
  1 650 km) and are excluded.
- The escalation rule was selected among five candidates on the same pairs;
  there is no cross-validation or confidence interval yet. Pairs of the same
  cyclone are not independent, so intervals must resample cyclones.
- A declaration reflects hazard, exposure, management and reporting; part of
  the false alarms are unreported impacts.
- A72 is antecedent rain, not soil moisture.
- The wind-tail variants were evaluated outside the full pipeline.

---

## Reproduction

```bash
conda activate itchi
pip install -e . --no-deps

python scripts/mswep_v3_6h_accum.py                    # 6-h accumulations (paths in Config)
python scripts/percentiles_6h.py --durations 6 24 72   # local percentiles by duration
python scripts/declaratorias_6h.py                     # TC declarations vs 6/24/72-h rain
python scripts/negativos_w06.py                        # municipality–cyclone pairs (positives and negatives)
```

| Notebook (`notebooks/second test/`) | Content | Outputs |
|---|---|---|
| `05_itchi_v02_mejoras.ipynb` | Cristóbal and Odile: v0.1, R4 and v0.2; maps and municipal PSS | `outputs/figures/itchi_v02_mejoras/` |
| `06_verificacion_metricas_figuras.ipynb` | Metrics explained, national verification figures and tables (sections 1–4 of this document) | `outputs/figures/verificacion_v02/` |
| `07_itchi_v02_peligro_por_snapshot.ipynb` | Hazard by snapshot, attribution by step, duration and timing | `outputs/diagnostics/itchi_v02_snapshots/`, `outputs/figures/itchi_v02_snapshots/` |
| `08_itchi_v01_v02_mapas_por_posicion.ipynb` | v0.1 vs v0.2 maps at every position and event maps (reads the fields saved by 07) | `outputs/figures/itchi_v02_posiciones/` |

---

## References

- Jolliffe, I. T. and Stephenson, D. B. (eds.) (2012). *Forecast Verification:
  A Practitioner's Guide in Atmospheric Science*, 2nd ed. Wiley.
  ISBN 9780470660713. Definitions of POD, POFD, PSS, FAR, CSI, ETS, HSS and ROC.
- Ferro, C. A. T. and Stephenson, D. B. (2011). Extremal Dependence Indices:
  improved verification measures for deterministic forecasts of rare binary
  events. *Weather and Forecasting*, 26(5), 699–713.
  https://doi.org/10.1175/WAF-D-10-05030.1. Behaviour of scores for rare events; SEDI.
- Willoughby, H. E., Darling, R. W. R. and Rahn, M. E. (2006). Parametric
  representation of the primary hurricane vortex. Part II: A new family of
  sectionally continuous profiles. *Monthly Weather Review*, 134(4), 1102–1120.
  https://doi.org/10.1175/MWR3106.1. W06 wind profile.
- Pérez-Alarcón, A. et al. (2021). Comparative climatology of outer tropical
  cyclone size using radial wind profiles. *Weather and Climate Extremes*.
  W06 parameterization of the outer radius (PA21).
