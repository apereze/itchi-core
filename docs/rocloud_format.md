# ROCLOUD text format

This document describes the operational ROCLOUD text files currently supported by `itchi-core`.

The parser is implemented in:

```text
src/itchi/rocloud.py
```

The supported source files are plain text/tabulated files without header, for example:

```text
NA880.dat
EP880.dat
```

The reader also accepts equivalent files with extensions:

```text
.dat
.txt
.dot
```

---

## Raw column order

The operational 17-column format is:

| Position | Raw column | Description |
|---:|---|---|
| 1 | `dd` | Day |
| 2 | `mm` | Month |
| 3 | `yy` | Year |
| 4 | `hh` | Hour, either integer hour such as `6` or compact `HHMM` such as `0600` |
| 5 | `lat` | Tropical cyclone center latitude |
| 6 | `lon` | Tropical cyclone center longitude |
| 7 | `MWS` | Maximum wind speed, kt |
| 8 | `CPSL` | Central pressure / minimum sea-level pressure, hPa |
| 9 | `RNE` | ROCLOUD maximum extent in northeastern quadrant, km |
| 10 | `RNO` | ROCLOUD maximum extent in northwestern quadrant, km |
| 11 | `RSO` | ROCLOUD maximum extent in southwestern quadrant, km |
| 12 | `RSE` | ROCLOUD maximum extent in southeastern quadrant, km |
| 13 | `Rp` | Mean ROCLOUD extent, km |
| 14 | `A` | Asymmetry metric |
| 15 | `D` | Dispersion metric |
| 16 | `S` | Solidity metric |
| 17 | `CT` | Tropical cyclone identifier |

Missing data are encoded as:

```text
-9999
```

and are converted to `NaN` by the parser.

---

## Canonical ITCHI mapping

The ITCHI internal ROCLOUD contract requires:

```text
storm_id
time
rocloud_rne
rocloud_rse
rocloud_rsw
rocloud_rnw
```

The text parser applies the following mapping:

| Raw column | ITCHI column |
|---|---|
| `CT` | `storm_id` |
| `dd`, `mm`, `yy`, `hh` | `time` |
| `RNE` | `rocloud_rne` |
| `RNO` | `rocloud_rnw` |
| `RSO` | `rocloud_rsw` |
| `RSE` | `rocloud_rse` |
| `MWS` | `vmax_kt` |
| `CPSL` | `pmin_hpa` |
| `Rp` | `rocloud_mean_km` |
| `A` | `asymmetry` |
| `D` | `dispersion` |
| `S` | `solidity` |

The quadrant mapping is intentionally explicit because the raw file uses Spanish cardinal labels:

```text
RNO -> northwestern quadrant -> rocloud_rnw
RSO -> southwestern quadrant -> rocloud_rsw
```

---

## Recommended usage

Use `rocloud_text_to_dataframe` when reading the real ROCLOUD text files:

```python
from itchi.rocloud import rocloud_text_to_dataframe

rocloud_ep = rocloud_text_to_dataframe("data/rocloud/EP880.dat")
rocloud_na = rocloud_text_to_dataframe("data/rocloud/NA880.dat")
```

To keep only canonical synoptic times:

```python
rocloud_ep = rocloud_text_to_dataframe(
    "data/rocloud/EP880.dat",
    synoptic_only=True,
)
```

`read_rocloud_table` can still be used for generic loading. For `.dat`, `.txt` and `.dot` files it returns the raw 17-column table with assigned column names; for `.csv` and `.parquet` it preserves the existing behavior.

---

## Scientific constraint

The ROCLOUD radius is the outer precipitation-attribution boundary used by ITCHI. It must remain in kilometers and should not be conflated with R34 or with any accumulated precipitation quantity.

If all four ROCLOUD quadrants are missing for a snapshot, the attribution radius cannot be resolved and the pipeline should fail. If only some quadrants are missing, the existing radius-resolution utilities may fill missing quadrants using the configured strategy, currently `mean_available`.
