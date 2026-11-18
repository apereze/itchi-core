# Real-case validation checklist

This document defines the minimum requirements to validate ITCHI-core with one or two real tropical cyclone cases before moving toward v1 diagnostics.

The purpose is not to claim operational skill. The purpose is to verify that the repository can ingest real sources, build snapshot inputs and produce physically inspectable ITCHI products.

---

## Recommended first cases

Start with one or two well-documented events with clear precipitation structure and available ROCLOUD/MSWEP coverage.

Suggested criteria:

- the storm exists in IBTrACS with stable `storm_id`;
- the same `storm_id` exists in ROCLOUD database files;
- the event period has MSWEP files covering the requested domain and times;
- the storm has at least several 00, 06, 12 and 18 UTC records;
- the storm track intersects or approaches the MSWEP subset domain;
- the event is meteorologically interpretable from maps.

For example, a first validation could use:

```text
one Eastern Pacific case
one North Atlantic / Gulf / Caribbean case
```

The exact cases should be chosen after confirming ID consistency across IBTrACS and ROCLOUD.

---

## Required inputs

### Track source

Required:

```text
IBTrACS NetCDF
```

Used by:

```text
src/itchi/ibtracs.py
```

Minimum fields after parsing:

```text
storm_id
time
lat
lon
vmax_kt
rmw_km
r34_rne
r34_rse
r34_rsw
r34_rnw
```

Notes:

- R34 is assumed to be in nautical miles by default.
- If `rmw_km` is missing and wind hazard is not precomputed, the snapshot pipeline may fail.
- If R34 is unavailable for weak systems, a fallback direct radius may be required.

---

### ROCLOUD source

Required:

```text
EP_TCSize_2000_2024.dat
NA_TCSize_2000_2024.dat
```

Used by:

```text
src/itchi/rocloud.py
```

Minimum fields after parsing:

```text
storm_id
storm_name
time
lat
lon
rocloud_rne
rocloud_rnw
rocloud_rsw
rocloud_rse
rocloud_mean_km
asymmetry
dispersion
solidity
```

Notes:

- `RNO` is mapped to `rocloud_rnw`.
- `RSO` is mapped to `rocloud_rsw`.
- Missing values encoded as `-9999` are converted to `NaN`.
- Declared block counts are preserved but not validated by default.

---

### Precipitation source

Required:

```text
MSWEP NetCDF
```

Used by:

```text
src/itchi/mswep.py
```

Minimum fields:

```text
precipitation variable
time or valid_time
lat/lon or latitude/longitude
```

Critical rule:

```text
MSWEP is treated as valid-time snapshots.
No temporal accumulation is performed.
No 3-hourly fields are summed to create 6-hourly products.
```

---

### Precipitation thresholds

For initial technical validation, scalar thresholds are acceptable:

```text
q90 = 10
q95 = 20
q99 = 30
```

For scientific validation, use gridded climatological thresholds:

```text
Q90(month, synoptic_hour, lat, lon)
Q95(month, synoptic_hour, lat, lon)
Q99(month, synoptic_hour, lat, lon)
```

For the current smoke validation, the repository allows either:

```text
scalar threshold values
NetCDF threshold files
```

---

## Minimum local folder layout

Recommended local layout:

```text
data/
  ibtracs/
    ibtracs.nc
  rocloud/
    EP_TCSize_2000_2024.dat
    NA_TCSize_2000_2024.dat
  mswep/
    event_precip.nc
  climatology/
    q90.nc
    q95.nc
    q99.nc
outputs/
  events/
  figures/
```

The full data files should not be committed to Git.

---

## Validation order

### Step 1: synthetic checks

```bash
python examples/smoke_test_synthetic.py \
  --output-path outputs/events/smoke_test_synthetic.nc

python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_synthetic.nc

python examples/smoke_test_from_tables.py \
  --output-path outputs/events/smoke_test_from_tables.nc

python examples/read_compiled_event.py \
  --input-path outputs/events/smoke_test_from_tables.nc
```

### Step 2: unit tests

```bash
python -m pytest tests/
pre-commit run --all-files
```

### Step 3: inspect one real storm interactively

```bash
jupyter lab notebooks/02_real_data_template.ipynb
```

Inspect:

```text
track_df rows for the selected event window
rocloud_df rows for the selected event window
MSWEP valid_time coverage
precipitation snapshot map
storm center location relative to domain
snapshot_inputs[0]
compiled metadata
ITCHI maps
```

### Step 4: run the non-interactive template

```bash
python examples/real_data_template.py \
  --ibtracs-path data/ibtracs/ibtracs.nc \
  --rocloud-path data/rocloud/EP_TCSize_2000_2024.dat \
  --mswep-path data/mswep/event_precip.nc \
  --precipitation-variable precipitation \
  --storm-id EP182023 \
  --start-time 2023-10-24T00:00:00 \
  --end-time 2023-10-25T06:00:00 \
  --q90 10 \
  --q95 20 \
  --q99 30 \
  --output-path outputs/events/EP182023_itchi.nc
```

---

## Acceptance criteria for this pre-v1 validation

A real-case run is acceptable at this stage if:

1. the selected storm is found in IBTrACS;
2. the selected storm is found in ROCLOUD;
3. all target times are available or are matched with explicit tolerances;
4. MSWEP covers the requested event period and domain;
5. ITCHI fields are produced and remain bounded in `[0, 1]`;
6. the maximum ITCHI area is meteorologically plausible under visual inspection;
7. metadata clearly report radius filling or fallback behavior.

This is a construction validation, not yet a full statistical validation.

---

## What can wait until v1

The following diagnostics are important, but can be formalized after one or two successful real-case builds:

```text
full automated event diagnostics
batch processing of many storms
declaration/disaster validation
ROC-AUC / PR-AUC / Brier score
reliability diagrams
spatial hit/false-alarm metrics
formal comparison against emergency declarations
```
