# ITCHI data contracts

This document defines the minimum data contracts required by `itchi-core`.

The purpose is to make explicit what each module expects as input and what it returns as output. This is especially important before connecting real datasets such as IBTrACS, ROCLOUD and MSWEP.

ITCHI v0.1 follows a snapshot-based design:

```text
precipitation is treated as a valid-time snapshot, not as a temporal accumulation
```

Therefore, no module in the core workflow should sum precipitation over time unless a future product explicitly defines accumulated precipitation as a valid input.

---

## Core conventions

### Spatial convention

ITCHI operates on gridded fields.

Expected spatial dimensions may be:

```text
lat, lon
```

or generic grid dimensions:

```text
y, x
```

The dimensions must be internally consistent across:

```text
precipitation
lon
lat
q90
q95
q99
wind_hazard_normalized, if provided
```

For xarray inputs, dimension coordinates should be preserved whenever possible. The `snapshot_inputs.py` module may copy missing dimension coordinates from precipitation to `lon`, `lat` and precomputed wind hazard fields when dimensions and shape already match.

This is not regridding or interpolation.

---

### Temporal convention

Canonical synoptic hours are:

```text
00, 06, 12, 18 UTC
```

The preferred time coordinate for precipitation is:

```text
valid_time
```

The fallback accepted name is:

```text
time
```

Snapshot selection may be exact or nearest within an explicit tolerance.

Nearest selection must not be used without tolerance.

---

### Quadrant convention

The internal ITCHI quadrant convention is:

```text
RNE, RSE, RSW, RNW
```

Accepted aliases may include:

```text
NE, SE, SW, NW
```

but they must be normalized internally to:

```text
RNE, RSE, RSW, RNW
```

No downstream component should use mixed labels.

---

## Track table contract

The track table represents tropical cyclone center and intensity metadata.

It is handled by:

```text
src/itchi/tracks.py
```

Minimum required columns after standardization:

| Column | Type | Unit | Required | Description |
|---|---|---|---|---|
| `storm_id` | string | none | yes | Storm identifier |
| `time` | datetime64 | UTC | yes | Snapshot valid time |
| `lat` | float | degrees north | yes | Cyclone center latitude |
| `lon` | float | degrees east | yes | Cyclone center longitude |

Expected optional columns:

| Column | Type | Unit | Required | Description |
|---|---|---|---|---|
| `vmax_kt` | float | kt | conditional | Maximum sustained wind |
| `pmin_hpa` | float | hPa | no | Minimum central pressure |
| `rmw_km` | float | km | conditional | Radius of maximum wind |
| `r34_rne` | float | usually nautical miles | conditional | R34 in northeast quadrant |
| `r34_rse` | float | usually nautical miles | conditional | R34 in southeast quadrant |
| `r34_rsw` | float | usually nautical miles | conditional | R34 in southwest quadrant |
| `r34_rnw` | float | usually nautical miles | conditional | R34 in northwest quadrant |

### Wind metadata requirement

If `wind_hazard_normalized` is not provided, the pipeline must compute wind hazard internally. In that case, these fields are required:

```text
vmax_kt
rmw_km
```

If a precomputed wind hazard field is provided, `vmax_kt` and `rmw_km` may be absent.

### R34 unit

Best-track products often report wind radii in nautical miles.

The default unit for R34 in ITCHI table-driven workflows is:

```text
nm
```

The conversion to kilometers is handled downstream by the pipeline.

### Tropical depression rule

If:

```text
vmax_kt < 34
```

then R34 should not be physically interpreted as available.

The effective direct radius must be resolved by the pipeline using:

```text
RMW
```

or, if explicitly configured:

```text
fallback_direct_radius_km
```

If neither is available, the pipeline should fail.

---

## ROCLOUD table contract

The ROCLOUD table represents the outer precipitation attribution radius.

It is handled by:

```text
src/itchi/rocloud.py
```

Minimum required columns after standardization:

| Column | Type | Unit | Required | Description |
|---|---|---|---|---|
| `storm_id` | string | none | yes | Storm identifier |
| `time` | datetime64 | UTC | yes | Snapshot valid time |

Expected radius columns:

| Column | Type | Unit | Required | Description |
|---|---|---|---|---|
| `rocloud_rne` | float | km | conditional | ROCLOUD radius in northeast quadrant |
| `rocloud_rse` | float | km | conditional | ROCLOUD radius in southeast quadrant |
| `rocloud_rsw` | float | km | conditional | ROCLOUD radius in southwest quadrant |
| `rocloud_rnw` | float | km | conditional | ROCLOUD radius in northwest quadrant |

At least one valid ROCLOUD quadrant must be available.

If ROCLOUD is fully absent, ITCHI should fail because there is no external cyclone-attribution boundary.

If ROCLOUD is partially available, missing quadrants may be filled using the configured fill strategy, currently:

```text
mean_available
```

---

## Precipitation snapshot contract

Precipitation is handled by:

```text
src/itchi/precipitation_snapshots.py
```

Expected object:

```text
xarray.DataArray
```

or:

```text
xarray.Dataset
```

with an explicitly selected precipitation variable.

Expected dimensions:

```text
valid_time, y, x
```

or:

```text
valid_time, lat, lon
```

The time coordinate may be named:

```text
valid_time
```

or:

```text
time
```

The selected precipitation snapshot passed to the pipeline must no longer have the time dimension. Example:

```text
precipitation(y, x)
```

or:

```text
precipitation(lat, lon)
```

### Critical rule

The precipitation field is a snapshot:

```text
Do not sum precipitation across valid_time.
Do not convert 3-hourly fields to 6-hourly accumulations unless the source product explicitly defines valid accumulations.
```

---

## Climatology contract

Climatological precipitation thresholds are handled by:

```text
src/itchi/climatology.py
```

Required percentile fields:

| Name | Interpretation |
|---|---|
| `Q90` | onset of low or occasional hazard |
| `Q95` | high precipitation hazard |
| `Q99` | maximum precipitation hazard threshold |

These may be passed as:

```text
float
numpy.ndarray
xarray.DataArray
```

If passed as gridded `xarray.DataArray`, they should preserve the same spatial dimensions and coordinates as the precipitation snapshot.

---

## Precipitation hazard contract

The normalized precipitation hazard is:

```text
H_P
```

with:

```text
0 <= H_P <= 1
```

The piecewise interpretation is:

| Condition | Hazard |
|---|---|
| `P < Q90` | `0` |
| `Q90 <= P < Q95` | increases from `0` to `0.5` |
| `Q95 <= P < Q99` | increases from `0.5` to `1` |
| `P >= Q99` | `1` |

Important:

```text
Q95 is not saturation.
Q99 is the saturation threshold.
```

---

## Snapshot input contract

Snapshot inputs are built by:

```text
src/itchi/snapshot_inputs.py
```

A pipeline-compatible snapshot input must contain:

| Key | Description |
|---|---|
| `precipitation` | selected precipitation snapshot |
| `q90` | precipitation Q90 threshold |
| `q95` | precipitation Q95 threshold |
| `q99` | precipitation Q99 threshold |
| `lon` | longitude grid |
| `lat` | latitude grid |
| `center_lon` | cyclone center longitude |
| `center_lat` | cyclone center latitude |
| `r34_by_quadrant` | raw R34 values by quadrant |
| `rocloud_by_quadrant` | raw ROCLOUD values by quadrant |
| `r34_unit` | R34 unit |
| `rocloud_unit` | ROCLOUD unit |
| `vmax_kt` | maximum wind, if needed |
| `rmw_km` | radius of maximum wind, if needed |
| `wind_hazard_normalized` | optional precomputed wind hazard |
| `fallback_direct_radius_km` | optional fallback for direct radius |
| `radius_fill_strategy` | radius imputation strategy |
| `wind_below_threshold_mode` | weak-system wind normalization mode |
| `run_quality_control` | whether to run pipeline QC |

### Important responsibility boundary

`snapshot_inputs.py` should not resolve or impute radii.

It passes raw quadrant values to:

```text
pipeline.py
```

so that the pipeline can preserve metadata such as:

```text
R_direct_source
R_direct_filled_quadrants
R_direct_used_fallback
ROCLOUD_source
ROCLOUD_filled_quadrants
```

---

## Pipeline output contract

The snapshot-level pipeline returns both spatial fields and metadata.

Spatial fields include:

| Field | Meaning |
|---|---|
| `radius_km` | radial distance from cyclone center |
| `quadrant` | grid-cell quadrant |
| `R_direct_q` | effective direct radius by grid cell |
| `ROCLOUD_q` | external attribution radius by grid cell |
| `M_direct` | direct-region mask |
| `M_indirect` | indirect-region mask |
| `M_exterior` | exterior-region mask |
| `H_P` | precipitation hazard |
| `H_Pdir` | direct precipitation hazard |
| `H_Pind` | indirect precipitation hazard |
| `H_W` | wind hazard |
| `H_dir` | direct component |
| `H_ind` | indirect component |
| `ITCHI` | final snapshot index |

Metadata fields include:

| Field | Meaning |
|---|---|
| `R_direct_source` | source of effective direct radius |
| `R_direct_filled_quadrants` | quadrants imputed for direct radius |
| `R_direct_used_fallback` | whether fallback direct radius was used |
| `ROCLOUD_source` | source of external radius |
| `ROCLOUD_filled_quadrants` | ROCLOUD quadrants imputed |
| `wind_hazard_source` | `provided` or `radial_profile` |

---

## Compiler output contract

The event compiler returns:

```python
{
    "snapshots": {...},
    "metadata": [...],
    "event_products": {...},
}
```

### Snapshots

`snapshots` contains time-stacked spatial variables.

Example:

```text
ITCHI(time, y, x)
```

or:

```text
ITCHI(time, lat, lon)
```

### Metadata

`metadata` is a list of dictionaries, one per snapshot.

Metadata must not be stacked as spatial arrays.

### Event products

Required event-level products:

| Field | Formula |
|---|---|
| `ITCHI_max` | `max(ITCHI_t)` |
| `ITCHI_acc` | `1 - product(1 - ITCHI_t)` |

---

## Export contract

Export is handled by:

```text
src/itchi/io.py
```

Compiled events may be exported to:

```text
NetCDF
Zarr
```

The exported Dataset should contain:

```text
time-stacked snapshot fields
event-level products
serialized metadata in Dataset attrs
```

For NumPy-derived outputs, explicit dimensions should be supplied to avoid ambiguous xarray dimensions such as:

```text
dim_0
dim_1
```

Recommended generic dimensions:

```text
time, y, x
```

---

## Minimum reproducible workflows

### Synthetic direct workflow

```bash
python examples/smoke_test_synthetic.py \
  --output-path outputs/events/smoke_test_synthetic.nc
```

### Table-driven synthetic workflow

```bash
python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc
```

### Read exported event

```bash
python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_from_tables.nc
```

---

## Quality-control expectations

Snapshot quality control should validate:

1. all hazard fields are within `[0, 1]`;
2. `R_direct_q <= ROCLOUD_q`;
3. exterior-region ITCHI contribution is zero;
4. masks are mutually exclusive;
5. spatial dimensions and coordinates are aligned;
6. geometry units are kilometers.

Quality control should remain strict.

Input builders should adapt metadata and coordinates where appropriate rather than weakening physical validation.
