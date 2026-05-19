# ITCHI v0.1 Methodology

## 1. Purpose

The **Integrated Tropical Cyclone Hazard Index** (**ITCHI**) is designed as a
bounded physical hazard index for tropical cyclone conditions.

ITCHI v0.1 represents the combined hazard associated with:

1. precipitation directly linked to the inner cyclone circulation;
2. wind hazard near the cyclone core;
3. precipitation associated with external rainbands or cloud structures.

The index is defined on a spatial grid and for cyclone-centered synoptic
snapshots.

The core output is:

```text
ITCHI_g,h,t ∈ [0, 1]
```

where:

| Symbol | Meaning |
|---|---|
| `g` | grid cell |
| `h` | tropical cyclone event |
| `t` | valid snapshot time |

---

## 2. General design principles

ITCHI v0.1 follows four principles:

1. **Physical interpretability**
   Each component should be linked to a physical tropical cyclone process.

2. **Boundedness**
   All hazard components and the final index must remain within `[0, 1]`.

3. **Spatial attribution**
   The index should distinguish direct cyclone-core effects from indirect
   rainband effects.

4. **Operational compatibility**
   The method should work with historical products, forecasts, hindcasts and
   downscaled precipitation fields.

---

## 3. Temporal treatment

### 3.1. Snapshot precipitation

Precipitation is treated as a **snapshot**, not as an accumulated field.

This means that precipitation represents an instantaneous or valid-time state
of the precipitation field.

Therefore, ITCHI v0.1 does not assume that precipitation from 3-hourly fields
can be directly accumulated to 6-hourly totals.

This choice is important because ITCHI is intended to remain compatible with:

- MSWEP historical snapshots;
- numerical weather prediction outputs;
- hindcasts;
- downscaled precipitation forecasts;
- bias-corrected precipitation products.

### 3.2. Synoptic times

ITCHI is evaluated at tropical cyclone synoptic times:

```text
00, 06, 12 and 18 UTC
```

These times are used because they are consistent with standard tropical cyclone
best-track information.

---

## 4. Spatial framework

ITCHI is evaluated on a regular spatial grid.

The target resolution for the initial version is:

```text
0.1° × 0.1°
```

For each grid cell, the following cyclone-centered quantities are computed:

| Variable | Description |
|---|---|
| `radius_km` | distance from cyclone center to grid cell |
| `quadrant` | relative quadrant of the cell |
| `R_direct_q` | effective direct-region radius by quadrant |
| `ROCLOUD_q` | external attribution radius by quadrant |

The internal quadrant convention is:

```text
RNE, RSE, RSW, RNW
```

where:

| Quadrant | Meaning |
|---|---|
| `RNE` | northeast |
| `RSE` | southeast |
| `RSW` | southwest |
| `RNW` | northwest |

All geometric comparisons are performed in kilometers.

---

## 5. Radius definitions

### 5.1. R34

`R34` is the radius of 34-kt winds by quadrant.

When available, `R34` is typically obtained from tropical cyclone best-track
datasets such as IBTrACS.

In many best-track products, `R34` is reported in nautical miles. ITCHI
converts it internally to kilometers using:

```text
1 nautical mile = 1.852 km
```

### 5.2. ROCLOUD

`ROCLOUD` is used as the external attribution radius for cyclone-related
cloud/rainband structures.

It defines the outer boundary of precipitation that can be attributed to the
tropical cyclone in ITCHI v0.1.

Cells outside `ROCLOUD` do not contribute to ITCHI.

### 5.3. Effective direct radius: `R_direct_q`

ITCHI distinguishes between:

| Radius | Meaning |
|---|---|
| `R34_q` | observed or reported 34-kt wind radius |
| `R_direct_q` | effective radius used to define the direct hazard region |

This distinction is necessary because some systems, especially tropical
depressions, do not physically have 34-kt winds.

The rules are:

| Case | Rule |
|---|---|
| Complete `R34` | `R_direct_q = R34_q` |
| Partial `R34` | missing quadrants are filled with the mean of available quadrants |
| No `R34` and `Vmax < 34 kt` | use `RMW` as `R_direct_q` |
| No `R34` and no `RMW` | use `fallback_direct_radius_km`, if configured |
| No valid radius | raise an error |

Thus, ITCHI does not artificially assign `R34` to tropical depressions.
Instead, it uses a physically interpretable fallback based on the radius of
maximum wind.

---

## 6. Missing quadrant radii

For `R34` and `ROCLOUD`, missing quadrant values are filled using the mean of
available quadrants, if at least one valid quadrant exists.

For example:

```python
{
    "RNE": 120.0,
    "RSE": 100.0,
    "RSW": None,
    "RNW": 140.0,
}
```

The available mean is:

```text
(120 + 100 + 140) / 3 = 120
```

The resolved radius becomes:

```python
{
    "RNE": 120.0,
    "RSE": 100.0,
    "RSW": 120.0,
    "RNW": 140.0,
}
```

The imputed quadrant is recorded in the metadata.

---

## 7. Spatial regions

ITCHI v0.1 separates the cyclone environment into three regions.

### 7.1. Direct region

The direct region is defined as:

```text
M_direct = radius_km <= R_direct_q
```

This region represents the area where wind hazard and direct precipitation
hazard may jointly contribute to ITCHI.

### 7.2. Indirect region

The indirect region is defined as:

```text
M_indirect = R_direct_q < radius_km <= ROCLOUD_q
```

This region represents precipitation associated with external cyclone
structures, such as rainbands or cloud shields.

### 7.3. Exterior region

The exterior region is defined as:

```text
M_exterior = radius_km > ROCLOUD_q
```

The exterior region does not contribute to ITCHI.

### 7.4. Mask consistency

The three masks must satisfy:

```text
M_direct ∩ M_indirect = ∅
M_direct ∩ M_exterior = ∅
M_indirect ∩ M_exterior = ∅
```

and, where all radii are valid:

```text
M_direct ∪ M_indirect ∪ M_exterior = domain
```

---

## 8. Precipitation hazard

### 8.1. Percentile-based normalization

Precipitation hazard is computed using local climatological percentiles.

The main thresholds are:

| Percentile | Interpretation |
|---|---|
| `Q90` | onset of low or occasional hazard |
| `Q95` | high hazard |
| `Q99` | maximum hazard threshold |

The precipitation hazard `H_P` is defined in `[0, 1]`.

### 8.2. Piecewise definition

For precipitation snapshot value `P`:

```text
P < Q90        → H_P = 0
Q90 ≤ P < Q95 → H_P increases from 0 to 0.5
Q95 ≤ P < Q99 → H_P increases from 0.5 to 1
P ≥ Q99       → H_P = 1
```

This formulation allows `Q95` to represent high hazard without forcing
saturation.

---

## 9. Wind hazard

### 9.1. Normalized wind hazard

Wind hazard is represented as:

```text
H_W = V*
```

where `V*` is a normalized wind hazard field in `[0, 1]`.

If a precomputed `V*` field is available, ITCHI can use it directly.

If not, ITCHI can compute `V*` internally using:

- radial distance from the cyclone center;
- maximum sustained wind `Vmax`;
- radius of maximum wind `RMW`.

### 9.2. Simple radial wind profile

The initial radial wind profile is intentionally simple.

For `r <= RMW`:

```text
V(r) = Vmax * (r / RMW)^a
```

For `r > RMW`:

```text
V(r) = Vmax * (RMW / r)^b
```

where:

| Parameter | Meaning |
|---|---|
| `a` | inner-core exponent |
| `b` | outer-decay exponent |

This is not intended to replace a full parametric wind model. It provides a
first-order profile for ITCHI v0.1 development and testing.

### 9.3. Wind normalization for systems with `Vmax > 34 kt`

For tropical storms and stronger systems:

```text
V* = clip((V(r) - 34) / (Vmax - 34), 0, 1)
```

### 9.4. Wind normalization for systems with `Vmax <= 34 kt`

For tropical depressions, the 34-kt threshold cannot be used as the lower bound.

Two modes are supported:

| Mode | Definition | Interpretation |
|---|---|---|
| `relative_to_vmax` | `V* = clip(V(r) / Vmax, 0, 1)` | relative wind hazard |
| `zero` | `V* = 0` | no wind hazard below 34 kt |

The default development mode is:

```text
relative_to_vmax
```

because tropical depressions may still produce local wind-related effects even
if they do not reach tropical-storm-force winds.

---

## 10. ITCHI components

### 10.1. Direct precipitation hazard

```text
H_Pdir = M_direct * H_P
```

### 10.2. Indirect precipitation hazard

```text
H_Pind = M_indirect * H_P
```

### 10.3. Wind hazard inside the direct region

```text
H_W = M_direct * V*
```

The direct mask is applied during component construction.

---

## 11. Direct and indirect hazard components

### 11.1. Direct component

The direct component combines wind and direct precipitation using a bounded
union formulation:

```text
H_dir = 1 - (1 - H_W)^α * (1 - H_Pdir)^β
```

where:

| Parameter | Meaning |
|---|---|
| `α` | wind sensitivity exponent |
| `β` | direct precipitation sensitivity exponent |

For the baseline version:

```text
α = 1
β = 1
```

### 11.2. Indirect component

The indirect component is currently defined as:

```text
H_ind = H_Pind
```

This means that, in ITCHI v0.1, indirect hazard is precipitation-driven.

---

## 12. Final ITCHI formulation

The final index combines direct and indirect components as:

```text
ITCHI = 1 - (1 - H_dir)^λ * (1 - H_ind)^μ
```

where:

| Parameter | Meaning |
|---|---|
| `λ` | direct-component sensitivity exponent |
| `μ` | indirect-component sensitivity exponent |

For the baseline version:

```text
λ = 1
μ = 1
```

The final index is clipped or validated to remain within:

```text
0 ≤ ITCHI ≤ 1
```

---

## 13. Event-level products

Given a time sequence of ITCHI snapshots for one tropical cyclone event:

```text
ITCHI_t
```

two event-level products are defined.

### 13.1. Maximum hazard

```text
ITCHI_max = max(ITCHI_t)
```

This product identifies the maximum hazard experienced by each grid cell during
the event.

### 13.2. Accumulated bounded hazard

```text
ITCHI_acc = 1 - product(1 - ITCHI_t)
```

This formulation increases with repeated exposure while remaining bounded in
`[0, 1]`.

---

## 14. Quality-control rules

ITCHI outputs must satisfy the following rules:

1. `ITCHI` must remain in `[0, 1]`.
2. `H_P`, `H_W`, `H_dir` and `H_ind` must remain in `[0, 1]`.
3. `R_direct_q <= ROCLOUD_q` where both values exist.
4. The exterior region must have zero ITCHI contribution.
5. Direct, indirect and exterior masks must not overlap.
6. Spatial fields must preserve dimensions and coordinates.
7. Geometric radii must be in kilometers before mask construction.

---

## 15. Current limitations

ITCHI v0.1 is still under development.

Important current limitations include:

1. The radial wind profile is a simplified first-order approximation.
2. Climatological precipitation percentiles are assumed to be precomputed.
3. Precipitation temporal alignment is not yet implemented as a dedicated module.
4. Real-case validation against impact or disaster records is not yet included.
5. Sensitivity analysis for the exponents `α`, `β`, `λ` and `μ` remains pending.
6. The current implementation prioritizes transparency and modularity over
   computational optimization.

---

## 16. Recommended next methodological developments

Future methodological work should include:

1. formal climatology module for `Q90`, `Q95` and `Q99`;
2. precipitation snapshot alignment module;
3. real-data adapters for MSWEP, IBTrACS and ROCLOUD;
4. sensitivity analysis of weighting exponents;
5. validation against independent impact datasets;
6. comparison against alternative hazard formulations;
7. uncertainty quantification for missing or imputed radii.
