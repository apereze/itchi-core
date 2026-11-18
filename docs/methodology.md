# ITCHI methodology (current scheme)

ITCHI (*Integrated Tropical Cyclone Hazard Index*) is a bounded index in
[0, 1] of the hazard produced by a tropical cyclone at each grid cell and
synoptic time. It combines precipitation, measured against the local
climatology, and wind, measured with absolute Saffir-Simpson thresholds. The
scheme described here is the default of `configs/default.yaml` and of
`pipeline.compute_itchi_snapshot` since v0.2.2. Earlier formulations (v0.1,
v0.2.0) and their verification are in `docs/historial/` and in the git tag
`v0.2.2`.

## 1. Regions

For each quadrant q (RNE, RSE, RSW, RNW) of the cyclone:

| Radius | Definition |
|---|---|
| R_direct_q | observed R34 (missing quadrants filled with the mean of the valid ones); without R34, the RMW for depressions or the parametric R34 |
| R_W06_q | radius where the Willoughby et al. (2006) profile falls to 2 m s-1, with the Pérez-Alarcón et al. (2021) configuration (empirical Rmax, no R34 calibration); never below R_direct_q |
| R_attr_q | ROCLOUD_q with R_W06_q as a floor: R_W06_q where ROCLOUD_q is missing, 0 (no detection) or smaller (`attribution_radius_mode: "floor"`) |

The direct region is r <= R_direct_q, the indirect region R_direct_q < r <= R_attr_q
and the exterior r > R_attr_q, where ITCHI = 0.

## 2. Precipitation hazard H_P

Three accumulations of MSWEP V3 at synoptic time T, each compared with its own
local percentiles (wet windows, May–November, `p_wet_raw`):

- P6 = precipitation in (T − 6 h, T];
- R24 = (T − 24 h, T] (persistence);
- A72 = (T − 78 h, T − 6 h] (exclusive antecedent, climatology of R72).

H_P is a ramp over the 6-h percentiles, escalated when the rain persists on
wet ground (`precipitation.PRECIP_MATRIX`):

    H_P = (1 − f) · ramp_base(P6) + f · ramp_esc(P6)

| P6 | 0 mm | P75 | P90 | P95 |
|---|--:|--:|--:|--:|
| ramp_base | 0 | 0 | 0.25 | 1 |
| ramp_esc | 0 | 0 | 0.8 | 1 |

with f = f_R24 · f_A72, f_R24 = clip((R24 − P90₂₄)/(P95₂₄ − P90₂₄), 0, 1) and
f_A72 = clip(A72 / P75₇₂, 0, 1). The maximum is the 6-h P95; the escalation
only acts on the P6 node below the maximum. A missing value or percentile
gives f = 0; a missing P6 or 6-h percentile gives H_P = 0.

## 3. Wind hazard H_W

The local wind V is the W06 profile for (Vmax, latitude, RMW), with X1 fitted
to the observed azimuthal-mean R34 and the radius scaled by quadrant so that
V = 34 kt at R_direct_q. H_W is absolute: it does not depend on the storm
Vmax, except inside the RMW, which takes the hazard of Vmax (the eyewall
crosses the cell within the 6-h window).

| Local wind (kt, 1-min) | Category | H_W | Class |
|---|---|---|---|
| 23.3–34 (12 m s-1 to 34 kt) | tail | 0–0.2 | 1 |
| 34–63 | tropical storm | 0.2–0.4 | 2 |
| 64–82 | H1 | 0.4–0.6 | 3 (moderate) |
| 83–95 | H2 | 0.6–0.8 | 4 (high) |
| ≥ 96 | H3–H5 (major hurricane) | 0.8–1 (1 from 137 kt) | 5 (maximum) |

H_W is linear between the nodes (`constants.SAFFIR_SIMPSON_HAZARD_NODES`);
below 34 kt the tail is 0.2 (V − V_out)/(34 − V_out), with
V_out = max(V(R_W06_q), 12 m s-1). Thresholds: NHC Saffir-Simpson Hurricane
Wind Scale, https://www.nhc.noaa.gov/aboutsshws.php.

## 4. Components and index

    H_dir = 1 − (1 − H_W)^α (1 − H_P)^β        (direct region)
    H_ind = 1 − (1 − H_W) (1 − H_P)            (indirect region; H_W is the tail)
    ITCHI = 1 − (1 − H_dir)^λ (1 − H_ind)^μ

with α = β = λ = μ = 1. Classes are steps of 0.2, (lower, upper]
(`plotting.classify_itchi_to_siat_levels`).

## 5. Municipal value

A municipality is represented by the 0.1° cells whose centre falls inside it
(or the closest cell to its representative point); its value is the maximum
over those cells (`itchi.municipal.municipal_values`).

## 6. Open decisions (v0.3)

The precipitation thresholds in percentiles are provisional: v0.3 replaces
them by return periods (F1, `docs/periodos_retorno.md`). The physical
justification of the 12 m s-1 tail cut and the municipal statistic are
pending (`docs/plan_v0.3.md`, section 4).

## References

- Willoughby, Darling & Rahn (2006): Parametric representation of the primary
  hurricane vortex. Part II. *Mon. Wea. Rev.*, 134, 1102–1120,
  https://doi.org/10.1175/MWR3106.1.
- Pérez-Alarcón et al. (2021): Comparative climatology of outer tropical
  cyclone size using radial wind profiles.
- Beck et al. (2019): MSWEP V2 global 3-hourly 0.1° precipitation.
- Pérez-Estrada & Dominguez (2025): A database for the outer sizes of
  tropical cyclones over the Middle Americas (ROCLOUD).
