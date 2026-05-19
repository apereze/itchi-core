# ITCHI Core API Reference

This document summarizes the main public API of `itchi-core`.

The goal is to provide a practical reference for using the package during
development, testing and scientific experimentation.

---

## 1. Package design

`itchi-core` is organized around a modular workflow:

```text
tracks / rocloud / precipitation / wind
        ↓
geometry + radii
        ↓
masks
        ↓
components
        ↓
index
        ↓
pipeline
        ↓
aggregation
        ↓
compiler
```

The recommended high-level entry points are:

| Level | Function | Module |
|---|---|---|
| Single snapshot | `compute_itchi_snapshot_from_grid` | `itchi.pipeline` |
| Event compilation | `compile_itchi_event` | `itchi.compiler` |
| Event products only | `compute_event_products` | `itchi.aggregation` |
| Save output | `write_result`, `write_dataset` | `itchi.io` |

---

## 2. Snapshot-level API

### `compute_itchi_snapshot_from_grid`

```python
from itchi.pipeline import compute_itchi_snapshot_from_grid
```

Computes ITCHI for one cyclone-centered precipitation snapshot.

### Minimal example

```python
import numpy as np

from itchi.pipeline import compute_itchi_snapshot_from_grid

lon = np.array([0.0, 0.2, 0.5, 0.9])
lat = np.array([0.0, 0.0, 0.0, 0.0])

precipitation = np.array([5.0, 15.0, 25.0, 40.0])

result = compute_itchi_snapshot_from_grid(
    precipitation=precipitation,
    q90=10.0,
    q95=20.0,
    q99=30.0,
    lon=lon,
    lat=lat,
    center_lon=0.0,
    center_lat=0.0,
    r34_by_quadrant={
        "RNE": 40.0,
        "RSE": 40.0,
        "RSW": 40.0,
        "RNW": 40.0,
    },
    r34_unit="nm",
    rocloud_by_quadrant={
        "RNE": 130.0,
        "RSE": 130.0,
        "RSW": 130.0,
        "RNW": 130.0,
    },
    rocloud_unit="km",
    wind_hazard_normalized=None,
    vmax_kt=80.0,
    rmw_km=20.0,
    run_quality_control=True,
)

itchi = result["ITCHI"]
```

### Main inputs

| Argument | Description |
|---|---|
| `precipitation` | Precipitation snapshot field |
| `q90`, `q95`, `q99` | Local climatological precipitation percentiles |
| `lon`, `lat` | Longitude and latitude grid |
| `center_lon`, `center_lat` | Cyclone center coordinates |
| `r34_by_quadrant` | R34 radii by quadrant |
| `rocloud_by_quadrant` | ROCLOUD radii by quadrant |
| `wind_hazard_normalized` | Precomputed wind hazard, or `None` |
| `vmax_kt` | Maximum sustained wind in knots |
| `rmw_km` | Radius of maximum wind in kilometers |
| `run_quality_control` | Whether to validate output consistency |

### Main outputs

| Key | Description |
|---|---|
| `radius_km` | Distance from cyclone center |
| `quadrant` | Relative cyclone quadrant |
| `R_direct_q` | Effective direct-region radius |
| `ROCLOUD_q` | External attribution radius |
| `M_direct` | Direct-region mask |
| `M_indirect` | Indirect-region mask |
| `M_exterior` | Exterior-region mask |
| `H_P` | Precipitation hazard |
| `H_W` | Wind hazard |
| `H_Pdir` | Direct precipitation hazard |
| `H_Pind` | Indirect precipitation hazard |
| `H_dir` | Direct hazard component |
| `H_ind` | Indirect hazard component |
| `ITCHI` | Final integrated index |

### Metadata outputs

| Key | Description |
|---|---|
| `R_direct_source` | Source/rule used for the direct radius |
| `R_direct_filled_quadrants` | Quadrants filled by imputation |
| `R_direct_used_fallback` | Whether fallback was used |
| `ROCLOUD_source` | Source/rule used for ROCLOUD |
| `ROCLOUD_filled_quadrants` | ROCLOUD quadrants filled by imputation |
| `wind_hazard_source` | `provided` or `radial_profile` |

---

## 3. Event-level API

### `compile_itchi_event`

```python
from itchi.compiler import compile_itchi_event
```

Runs the snapshot pipeline for multiple times and computes event-level products.

### Minimal example

```python
from itchi.compiler import compile_itchi_event

compiled = compile_itchi_event(
    snapshot_inputs=[
        snapshot_input_00,
        snapshot_input_06,
        snapshot_input_12,
    ],
    time_values=[
        "2020-09-01T00:00",
        "2020-09-01T06:00",
        "2020-09-01T12:00",
    ],
)

snapshots = compiled["snapshots"]
metadata = compiled["metadata"]
event_products = compiled["event_products"]
```

### Output structure

```python
{
    "snapshots": {
        "ITCHI": ...,
        "H_P": ...,
        "H_W": ...,
        "H_dir": ...,
        "H_ind": ...,
    },
    "metadata": [
        {
            "time": ...,
            "R_direct_source": ...,
            "wind_hazard_source": ...,
        }
    ],
    "event_products": {
        "ITCHI_max": ...,
        "ITCHI_acc": ...,
    },
}
```

---

## 4. Aggregation API

### `compute_event_products`

```python
from itchi.aggregation import compute_event_products
```

Computes standard event products from a time-dependent ITCHI field.

```python
products = compute_event_products(
    itchi=itchi_snapshots,
    dim="time",
)

itchi_max = products["ITCHI_max"]
itchi_acc = products["ITCHI_acc"]
```

### Event products

| Product | Definition |
|---|---|
| `ITCHI_max` | Maximum ITCHI value across the event |
| `ITCHI_acc` | Bounded accumulated ITCHI hazard |

The accumulated product is defined as:

```text
ITCHI_acc = 1 - product(1 - ITCHI_t)
```

This keeps the accumulated hazard bounded in `[0, 1]`.

---

## 5. Radius API

### `resolve_direct_radius`

```python
from itchi.radii import resolve_direct_radius
```

Resolves the effective direct-region radius.

Rules:

| Case | Rule |
|---|---|
| Complete R34 | Use R34 |
| Partial R34 | Fill missing quadrants with available mean |
| No R34 and `Vmax < 34 kt` | Use RMW |
| No R34 and no RMW | Use configured fallback, if provided |
| No valid radius | Raise `ValueError` |

Example:

```python
from itchi.radii import resolve_direct_radius

resolved = resolve_direct_radius(
    r34_by_quadrant={
        "RNE": 100.0,
        "RSE": 90.0,
        "RSW": None,
        "RNW": 110.0,
    },
    vmax_kt=45.0,
    rmw_km=25.0,
)

direct_radii = resolved.radii
```

---

## 6. Wind API

### `compute_wind_hazard_from_profile`

```python
from itchi.wind import compute_wind_hazard_from_profile
```

Builds a simple radial wind profile and converts it into normalized wind hazard
`V*`.

```python
wind_hazard = compute_wind_hazard_from_profile(
    radius_km=radius_km,
    vmax_kt=80.0,
    rmw_km=20.0,
)
```

For systems below 34 kt, two modes are available:

| Mode | Interpretation |
|---|---|
| `relative_to_vmax` | Normalize relative to the system maximum wind |
| `zero` | Set wind hazard to zero |

---

## 7. Track API

### `standardize_track_dataframe`

```python
from itchi.tracks import standardize_track_dataframe
```

Standardizes tropical cyclone track tables.

```python
track = standardize_track_dataframe(
    df,
    column_map={
        "storm_id": "sid",
        "time": "iso_time",
        "lat": "lat",
        "lon": "lon",
        "vmax_kt": "usa_wind",
        "pmin_hpa": "usa_pres",
    },
)
```

### `get_track_snapshot`

```python
from itchi.tracks import get_track_snapshot

row = get_track_snapshot(
    track,
    storm_id="AL012020",
    time="2020-06-01 06:00",
)
```

---

## 8. ROCLOUD API

### `extract_rocloud_radii`

```python
from itchi.rocloud import extract_rocloud_radii
```

Extracts and resolves ROCLOUD quadrant radii.

```python
resolved_rocloud = extract_rocloud_radii(row)

rocloud_radii = resolved_rocloud.radii
```

Missing quadrants are filled using the mean of available quadrants.

---

## 9. I/O API

### `write_result`

```python
from itchi.io import write_result
```

Converts an ITCHI result dictionary to an `xarray.Dataset` and writes it.

```python
write_result(
    result=result,
    path="outputs/snapshot.nc",
    dims=("lat", "lon"),
    coords={
        "lat": lat_values,
        "lon": lon_values,
    },
)
```

### `read_dataset`

```python
from itchi.io import read_dataset

dataset = read_dataset("outputs/snapshot.nc")
```

Supported formats:

| Format | Extension |
|---|---|
| NetCDF | `.nc`, `.nc4`, `.netcdf` |
| Zarr | `.zarr` |

---

## 10. Quality-control API

### `run_snapshot_quality_control`

```python
from itchi.quality_control import run_snapshot_quality_control
```

Runs standard consistency checks on one snapshot result.

Checks include:

1. ITCHI and hazard fields remain in `[0, 1]`;
2. direct radius is less than or equal to ROCLOUD;
3. exterior region has zero contribution;
4. masks do not overlap;
5. fields preserve dimensions and coordinates;
6. geometric units are kilometers.

Example:

```python
run_snapshot_quality_control(snapshot_result)
```

---

## 11. Recommended usage levels

| User need | Recommended entry point |
|---|---|
| Test one synthetic snapshot | `compute_itchi_snapshot_from_grid` |
| Run one full cyclone event | `compile_itchi_event` |
| Aggregate existing ITCHI snapshots | `compute_event_products` |
| Save outputs | `write_result` |
| Validate output | `run_snapshot_quality_control` |

---

## 12. Notes

The API is still under active development. Function signatures may change as
real-data workflows are incorporated.

For now, the most stable high-level functions are:

```python
from itchi.pipeline import compute_itchi_snapshot_from_grid
from itchi.compiler import compile_itchi_event
from itchi.aggregation import compute_event_products
```
