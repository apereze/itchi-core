# MSWEP NetCDF adapter

This document describes the MSWEP NetCDF adapter implemented in:

```text
src/itchi/mswep.py
```

The adapter converts MSWEP-like NetCDF files into precipitation snapshot inputs compatible with ITCHI v0.1.

---

## Core rule

ITCHI treats MSWEP precipitation as a valid-time snapshot field.

The adapter must not:

```text
sum precipitation across time
accumulate 3-hourly fields into 6-hourly totals
resample precipitation through time
```

The adapter may only:

```text
open NetCDF files
extract precipitation variables
standardize coordinate names
filter existing synoptic snapshots
select one existing valid-time snapshot
subset existing grid cells spatially
```

---

## Supported input object

The adapter accepts:

```text
xarray.Dataset
xarray.DataArray
NetCDF file path readable by xarray.open_dataset
```

The convenience reader is:

```python
from itchi.mswep import read_mswep_precipitation

precipitation = read_mswep_precipitation(
    "data/mswep/event_precip.nc",
    precipitation_variable="precipitation",
)
```

---

## Precipitation variable detection

If no explicit variable is provided, the adapter checks common names:

```text
precipitation
precip
precipitation_amount
precipitation_flux
pr
```

If no candidate is found, it falls back to the generic precipitation-variable extractor. Multi-variable datasets should provide an explicit variable name.

---

## Time coordinate

Accepted input time coordinates are:

```text
time
valid_time
```

The canonical output name is:

```text
valid_time
```

Example:

```python
from itchi.mswep import prepare_mswep_precipitation

precipitation = prepare_mswep_precipitation(
    dataset,
    precipitation_variable="precipitation",
    synoptic_only=True,
)
```

With `synoptic_only=True`, only existing snapshots at:

```text
00, 06, 12, 18 UTC
```

are retained.

---

## Spatial coordinates

The adapter can infer latitude and longitude names from:

```text
lat, latitude, y
lon, longitude, x
```

Longitude and latitude grids can be extracted as 2-D fields:

```python
from itchi.mswep import get_mswep_lon_lat

lon, lat = get_mswep_lon_lat(precipitation)
```

This is useful for `itchi.pipeline.compute_itchi_snapshot_from_grid`, which expects spatial grids compatible with the precipitation field.

---

## Spatial subsetting

Spatial subsetting is performed with existing 1-D coordinates only:

```python
from itchi.mswep import subset_mswep_domain

subset = subset_mswep_domain(
    precipitation,
    lon_min=-110,
    lon_max=-90,
    lat_min=5,
    lat_max=30,
    pad_degrees=2,
)
```

The function does not interpolate or regrid.

If the MSWEP file uses 0-360 longitude convention, negative longitude bounds are converted when possible. Dateline-crossing subsets are intentionally not supported in the initial adapter.

---

## Snapshot selection

A single snapshot can be selected as:

```python
from itchi.mswep import select_mswep_precipitation_snapshot

snapshot = select_mswep_precipitation_snapshot(
    precipitation,
    target_time="2023-10-24T06:00:00",
    method="exact",
)
```

Nearest-neighbor selection requires explicit tolerance:

```python
snapshot = select_mswep_precipitation_snapshot(
    precipitation,
    target_time="2023-10-24T06:00:00",
    method="nearest",
    tolerance="30min",
)
```

The target time is required to be synoptic by default.
