# ITCHI v0.1 Methodology

> **Integrated Tropical Cyclone Hazard Index** — A bounded physical hazard index for tropical cyclone conditions combining wind and precipitation hazards on a geospatial grid.

---

## Overview

ITCHI v0.1 is designed as a bounded physical hazard index that integrates three key components of tropical cyclone hazard:

1. **Direct precipitation** — Precipitation linked to the inner cyclone circulation
2. **Wind hazard** — Wind conditions near the cyclone core
3. **Indirect precipitation** — Precipitation from external rainbands or cloud structures

The index is defined spatially on a regular grid and evaluated at cyclone-centered synoptic snapshots.

### Core Output

The fundamental ITCHI output is a bounded scalar field:

$$\text{ITCHI}_{g,h,t} \in [0, 1]$$

where:

| Symbol | Meaning |
|:-------|:--------|
| $g$ | Grid cell |
| $h$ | Tropical cyclone event |
| $t$ | Valid snapshot time |

---

## Design Principles

ITCHI v0.1 follows four foundational principles:

### 1. Physical Interpretability
Each hazard component must be linked to observable physical tropical cyclone processes, ensuring the index remains scientifically grounded and interpretable.

### 2. Boundedness
All hazard components and the final index must remain constrained within $[0, 1]$, enabling direct probability or impact interpretations.

### 3. Spatial Attribution
The index distinguishes between:
- **Direct cyclone-core effects** (wind + inner precipitation)
- **Indirect rainband effects** (external precipitation)

This separation enables targeted hazard assessment and regional impact analysis.

### 4. Operational Compatibility
The methodology is designed to work seamlessly with:
- Historical satellite/reanalysis products (e.g., MSWEP)
- Numerical weather prediction (NWP) outputs
- Hindcasts and reforecasts
- Downscaled precipitation fields
- Bias-corrected precipitation products

---

## Temporal Framework

### Snapshot Precipitation

Precipitation is treated as a **snapshot** (instantaneous or valid-time state), not as an accumulated field. This means:

- Precipitation from 3-hourly fields **cannot** be directly accumulated to 6-hourly totals
- The index preserves compatibility with diverse temporal resolutions
- Temporal alignment is handled separately from hazard quantification

This design accommodates multiple data sources while maintaining methodological consistency.

**Compatible data sources:**
- MSWEP historical snapshots
- NWP outputs
- Hindcasts
- Downscaled precipitation forecasts
- Bias-corrected precipitation products

### Synoptic Evaluation Times

ITCHI is evaluated at standard tropical cyclone synoptic times:

$$\text{00, 06, 12, 18 UTC}$$

These times align with conventional tropical cyclone best-track reporting standards (e.g., IBTrACS), enabling direct comparison with observational baselines.

---

## Spatial Framework

### Grid Resolution

ITCHI is computed on a regular spatial grid. The target resolution for v0.1 is:

$$0.1° \times 0.1°$$

### Grid Cell Attributes

For each grid cell, the following cyclone-centered quantities are computed:

| Variable | Description |
|:---------|:------------|
| `radius_km` | Distance from cyclone center to grid cell |
| `quadrant` | Relative quadrant of the cell |
| `R_direct_q` | Effective direct-region radius by quadrant |
| `ROCLOUD_q` | External attribution radius by quadrant |

### Quadrant Convention

The internal quadrant convention uses compass directions:

$$\text{RNE, RSE, RSW, RNW}$$

| Quadrant | Direction |
|:---------|:----------|
| `RNE` | Northeast |
| `RSE` | Southeast |
| `RSW` | Southwest |
| `RNW` | Northwest |

**All geometric comparisons are performed in kilometers.**

---

## Radius Definitions

### R34: 34-kt Wind Radius

`R34` is the radius of 34-knot winds by quadrant. When available, it is obtained from tropical cyclone best-track datasets (e.g., IBTrACS).

**Unit conversion:**
$$1 \text{ nautical mile} = 1.852 \text{ km}$$

Many best-track products report `R34` in nautical miles; ITCHI converts internally to kilometers.

### ROCLOUD: External Attribution Radius

`ROCLOUD` defines the outer boundary of precipitation that can be attributed to the tropical cyclone in ITCHI v0.1:

- Cells **inside** `ROCLOUD` may contribute to ITCHI
- Cells **outside** `ROCLOUD` do not contribute
- Defines the extent of the indirect (rainband) region

### R_direct_q: Effective Direct Radius

ITCHI distinguishes between two related quantities:

| Radius | Definition |
|:-------|:-----------|
| `R34_q` | Observed or reported 34-kt wind radius (by quadrant) |
| `R_direct_q` | Effective radius defining the direct hazard region |

This distinction is necessary because many systems, particularly tropical depressions, do not physically achieve 34-kt winds.

#### Rules for Computing R_direct_q

| Condition | Rule |
|:----------|:-----|
| Complete `R34` available | $R_{\text{direct}, q} = R_{34, q}$ |
| Partial `R34` (missing quadrants) | Fill missing quadrants with mean of available quadrants |
| No `R34` and $V_{\max} < 34$ kt | Use `RMW` (radius of maximum wind) as $R_{\text{direct}, q}$ |
| No `R34` and no `RMW` | Use configured `fallback_direct_radius_km` |
| No valid radius | Raise error |

**Key principle:** ITCHI does not artificially assign 34-kt wind radii to tropical depressions. Instead, it uses physically interpretable fallbacks based on the radius of maximum wind.

### Missing Quadrant Handling

When quadrant-specific radius values are missing (for `R34` or `ROCLOUD`), they are filled using the arithmetic mean of available quadrants, provided at least one valid quadrant exists.

**Example:**

Given:
```python
{
    "RNE": 120.0,
    "RSE": 100.0,
    "RSW": None,
    "RNW": 140.0,
}
```

Available mean:
$$\bar{R} = \frac{120 + 100 + 140}{3} = 120 \text{ km}$$

Resolved radius:
```python
{
    "RNE": 120.0,
    "RSE": 100.0,
    "RSW": 120.0,  # Imputed
    "RNW": 140.0,
}
```

All imputations are recorded in output metadata for traceability.

---

## Spatial Regions

ITCHI v0.1 partitions the cyclone environment into three mutually exclusive, exhaustive regions:

### Direct Region

$$M_{\text{direct}} = \{r_{\text{km}} \leq R_{\text{direct}, q}\}$$

The direct region represents the area where wind hazard and direct precipitation hazard may jointly contribute to ITCHI.

### Indirect Region

$$M_{\text{indirect}} = \{R_{\text{direct}, q} < r_{\text{km}} \leq \text{ROCLOUD}_q\}$$

The indirect region represents precipitation from external cyclone structures (rainbands, cloud shields).

### Exterior Region

$$M_{\text{exterior}} = \{r_{\text{km}} > \text{ROCLOUD}_q\}$$

The exterior region does not contribute to ITCHI.

### Mask Consistency Requirements

The three spatial masks must be mutually disjoint:

$$M_{\text{direct}} \cap M_{\text{indirect}} = \emptyset$$
$$M_{\text{direct}} \cap M_{\text{exterior}} = \emptyset$$
$$M_{\text{indirect}} \cap M_{\text{exterior}} = \emptyset$$

Where all radii are valid, the three regions cover the entire domain:

$$M_{\text{direct}} \cup M_{\text{indirect}} \cup M_{\text{exterior}} = \text{domain}$$

---

## Precipitation Hazard

### Percentile-Based Normalization

Precipitation hazard is computed using local climatological percentiles. This approach enables consistent interpretation across diverse geographical regions and precipitation regimes.

**Reference percentiles:**

| Percentile | Interpretation |
|:-----------|:---------------|
| $Q_{90}$ | Onset of low or occasional hazard |
| $Q_{95}$ | High hazard threshold |
| $Q_{99}$ | Maximum hazard threshold |

The precipitation hazard field $H_P$ is normalized to $[0, 1]$.

### Piecewise Hazard Function

For a precipitation snapshot value $P$:

$$H_P = \begin{cases}
0 & \text{if } P < Q_{90} \\
\frac{0.5}{Q_{95} - Q_{90}}(P - Q_{90}) & \text{if } Q_{90} \leq P < Q_{95} \\
0.5 + \frac{0.5}{Q_{99} - Q_{95}}(P - Q_{95}) & \text{if } Q_{95} \leq P < Q_{99} \\
1 & \text{if } P \geq Q_{99}
\end{cases}$$

This formulation:
- Reaches 0.5 at the $Q_{95}$ threshold (high hazard)
- Continues to 1.0 at $Q_{99}$ without forcing premature saturation
- Enables nuanced hazard assessment in the high-precipitation regime

---

## Wind Hazard

### Normalized Wind Field

Wind hazard is represented as a normalized wind field $V^*$ in $[0, 1]$:

$$H_W = V^*$$

ITCHI can accept precomputed $V^*$ fields, or compute them internally using:
- Radial distance from cyclone center ($r$)
- Maximum sustained wind ($V_{\max}$)
- Radius of maximum wind (RMW)

### Radial Wind Profile

The initial radial wind profile is intentionally simplified for transparency and testability.

**Inner core** ($r \leq \text{RMW}$):
$$V(r) = V_{\max} \cdot \left(\frac{r}{\text{RMW}}\right)^a$$

**Outer region** ($r > \text{RMW}$):
$$V(r) = V_{\max} \cdot \left(\frac{\text{RMW}}{r}\right)^b$$

where:

| Parameter | Meaning |
|:----------|:--------|
| $a$ | Inner-core exponent (power-law profile inside RMW) |
| $b$ | Outer-decay exponent (power-law decay outside RMW) |

**Note:** This is a first-order approximation intended for ITCHI v0.1 development and validation. It is not designed to replace full parametric wind models.

### Wind Normalization: Tropical Storms and Stronger ($V_{\max} > 34$ kt)

For systems with sustained winds exceeding tropical-storm force:

$$V^* = \text{clip}\left(\frac{V(r) - 34}{V_{\max} - 34}, 0, 1\right)$$

This normalization:
- Sets the lower threshold at 34 kt (tropical storm minimum)
- Scales linearly from 0 to 1 relative to maximum wind
- Clips values outside $[0, 1]$

### Wind Normalization: Tropical Depressions ($V_{\max} \leq 34$ kt)

For systems below tropical-storm force, the 34-kt threshold cannot serve as a lower bound. Two modes are supported:

| Mode | Definition | Interpretation |
|:-----|:-----------|:----------------|
| `relative_to_vmax` | $V^* = \text{clip}(V(r) / V_{\max}, 0, 1)$ | Relative wind hazard (0 to 100% of Vmax) |
| `zero` | $V^* = 0$ | No wind hazard below 34 kt |

**Default development mode:** `relative_to_vmax`

**Rationale:** Tropical depressions may produce local wind-related effects even if they do not reach tropical-storm-force winds. This mode enables capturing weaker but potentially impactful wind effects.

---

## ITCHI Component Fields

### Direct Precipitation Component

$$H_{P, \text{dir}} = M_{\text{direct}} \cdot H_P$$

Precipitation hazard within the direct region only.

### Indirect Precipitation Component

$$H_{P, \text{ind}} = M_{\text{indirect}} \cdot H_P$$

Precipitation hazard within the indirect region only.

### Wind Component

$$H_W = M_{\text{direct}} \cdot V^*$$

Normalized wind hazard applied only within the direct region.

---

## Composite Hazard Components

### Direct Hazard Component

The direct component combines wind and direct precipitation using a bounded union formulation:

$$H_{\text{dir}} = 1 - (1 - H_W)^\alpha \cdot (1 - H_{P, \text{dir}})^\beta$$

**Parameters:**

| Parameter | Meaning | Baseline |
|:----------|:--------|:---------|
| $\alpha$ | Wind sensitivity exponent | 1 |
| $\beta$ | Direct precipitation sensitivity exponent | 1 |

**Interpretation:** This formulation ensures that:
- Both wind and precipitation can independently trigger high hazard
- The components are combined via a bounded union (saturates at 1)
- Exponents control relative weighting and interaction

### Indirect Hazard Component

$$H_{\text{ind}} = H_{P, \text{ind}}$$

**Current formulation:** In ITCHI v0.1, indirect hazard is purely precipitation-driven.

---

## Final ITCHI Index

The final ITCHI index combines direct and indirect components:

$$\text{ITCHI} = 1 - (1 - H_{\text{dir}})^\lambda \cdot (1 - H_{\text{ind}})^\mu$$

**Parameters:**

| Parameter | Meaning | Baseline |
|:----------|:--------|:---------|
| $\lambda$ | Direct-component sensitivity exponent | 1 |
| $\mu$ | Indirect-component sensitivity exponent | 1 |

### Output Validation

The final index is validated to remain bounded:

$$0 \leq \text{ITCHI} \leq 1$$

---

## Event-Level Products

Given a time series of ITCHI snapshots for a single tropical cyclone event ($\text{ITCHI}_t$), two event-level aggregation products are defined:

### Maximum Hazard

$$\text{ITCHI}_{\max} = \max_t(\text{ITCHI}_t)$$

Identifies the peak hazard experienced by each grid cell during the entire event lifecycle.

### Accumulated Bounded Hazard

$$\text{ITCHI}_{\text{acc}} = 1 - \prod_t(1 - \text{ITCHI}_t)$$

Captures cumulative exposure while remaining bounded in $[0, 1]$. This formulation:
- Increases with repeated cyclone passages
- Remains bounded (never exceeds 1)
- Reflects integrated risk over the event duration

---

## Quality Control

ITCHI outputs must satisfy the following quality-control rules:

| Rule ID | Requirement |
|:--------|:-----------|
| QC-1 | $\text{ITCHI} \in [0, 1]$ |
| QC-2 | $H_P, H_W, H_{\text{dir}}, H_{\text{ind}} \in [0, 1]$ |
| QC-3 | $R_{\text{direct}, q} \leq \text{ROCLOUD}_q$ where both exist |
| QC-4 | Exterior region receives zero ITCHI contribution |
| QC-5 | Direct, indirect, and exterior masks are mutually disjoint |
| QC-6 | Spatial fields preserve dimensions and coordinates |
| QC-7 | All geometric radii are in kilometers before mask construction |

---

## Current Limitations

ITCHI v0.1 is an operational research prototype. Important current limitations include:

- **Wind profile:** Simplified first-order power-law approximation; not a full parametric model
- **Climatology:** Precipitation percentiles ($Q_{90}, Q_{95}, Q_{99}$) assumed to be precomputed; no dedicated climatology module yet
- **Temporal alignment:** Precipitation snapshot alignment not yet implemented as a dedicated module
- **Validation:** Real-case validation against impact or disaster records not yet conducted
- **Sensitivity analysis:** Parameter sensitivity for $\alpha, \beta, \lambda, \mu$ pending formal analysis
- **Computational efficiency:** Current implementation prioritizes transparency and modularity over optimization

---

## Recommended Future Developments

### High-Priority Methodological Items

1. **Formal climatology module** — Compute $Q_{90}, Q_{95}, Q_{99}$ from regional precipitation climatologies
2. **Precipitation alignment module** — Handle temporal misalignment between wind and precipitation fields
3. **Real-data adapters** — Validated interfaces for MSWEP, IBTrACS, and ROCLOUD products
4. **Sensitivity analysis** — Quantify parameter sensitivity and uncertainty propagation
5. **Impact validation** — Compare ITCHI against independent impact datasets and disaster records
6. **Alternative formulations** — Compare against other hazard aggregation methods
7. **Uncertainty quantification** — Formal uncertainty framework for missing or imputed radii

### Computational Optimization

- Vectorization and parallelization of grid computations
- Efficient spatial indexing and masking operations
- Performance profiling and optimization

---

## References

This methodology document defines the mathematical framework, design principles, and quality standards for ITCHI v0.1. For implementation details, see the [ITCHI Core Repository](https://github.com/apereze/itchi-core).

### Related Documentation

- [Installation Guide](./INSTALL.md)
- [API Reference](./API.md)
- [User Guide](./USER_GUIDE.md)
- [Examples](../examples/)
