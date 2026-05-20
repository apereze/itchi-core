# ROCLOUD text format

This document describes the ROCLOUD text files currently supported by `itchi-core`.

The parser is implemented in:

```text
src/itchi/rocloud.py
```

The reader accepts text files with extensions:

```text
.dat
.txt
.dot
```

Two layouts are supported:

1. a legacy flat 17-column layout;
2. the database block layout used by `NA_TCSize_2000_2024.dat` and `EP_TCSize_2000_2024.dat`.

---

## Database block layout

The current database files use one header line per tropical cyclone, followed by one or more records.

Header example:

```text
EP022000,                BUD,     18,
```

Header fields:

| Position | Field | Description |
|---:|---|---|
| 1 | `storm_id` | Tropical cyclone identifier, for example `EP022000` or `AL012000` |
| 2 | `storm_name` | Tropical cyclone name |
| 3 | `declared_entries` | Number of entries declared in the source header |

Record example:

```text
20000613  1200   13.9 -106.6  64  1000   1062.89    545.64    862.23    825.34    824.03  0.49  0.18  0.34   1111.07   1074.93    865.39    959.66   1002.76
```

Record fields:

| Position | Raw column | Description |
|---:|---|---|
| 1 | `date` | Date as `YYYYMMDD` |
| 2 | `hh` | Time as `HHMM`, for example `0000`, `0600`, `1200`, `1800` |
| 3 | `lat` | Tropical cyclone center latitude |
| 4 | `lon` | Tropical cyclone center longitude |
| 5 | `MWS` | Maximum wind speed, kt |
| 6 | `CPSL` | Central pressure / minimum sea-level pressure, hPa |
| 7 | `RNE` | ROCLOUD maximum extent in northeastern quadrant, km |
| 8 | `RNO` | ROCLOUD maximum extent in northwestern quadrant, km |
| 9 | `RSO` | ROCLOUD maximum extent in southwestern quadrant, km |
| 10 | `RSE` | ROCLOUD maximum extent in southeastern quadrant, km |
| 11 | `Rp` | Mean ROCLOUD extent, km |
| 12 | `A` | Asymmetry metric |
| 13 | `D` | Dispersion metric |
| 14 | `S` | Solidity metric |
| 15 | `RBP_RNE` | RBP maximum extent in northeastern quadrant, km |
| 16 | `RBP_RNO` | RBP maximum extent in northwestern quadrant, km |
| 17 | `RBP_RSO` | RBP maximum extent in southwestern quadrant, km |
| 18 | `RBP_RSE` | RBP maximum extent in southeastern quadrant, km |
| 19 | `RBP_mean` | Mean RBP extent, km |

Missing data are encoded as:

```text
-9999
```

and are converted to `NaN` by the parser.

### Declared entries

The parser preserves `declared_entries`, but it does not validate it by default. This is intentional because the declared number may reflect the source best-track entry count, whereas the released ROCLOUD table may include only rows with available satellite-derived geometry.

Strict validation can be requested with:

```python
from itchi.rocloud import read_rocloud_database_text_table

raw = read_rocloud_database_text_table(
    "data/rocloud/EP_TCSize_2000_2024.dat",
    validate_record_counts=True,
)
```

---

## Legacy flat 17-column layout

The parser also supports a flat 17-column layout without storm headers:

```text
dd mm yy hh lat lon MWS CPSL RNE RNO RSO RSE Rp A D S CT
```

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
| `CT` or header field 1 | `storm_id` |
| header field 2 | `storm_name` |
| header field 3 | `declared_entries` |
| `dd`, `mm`, `yy`, `hh` or `date`, `hh` | `time` |
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
| `RBP_RNE` | `rbp_rne` |
| `RBP_RNO` | `rbp_rnw` |
| `RBP_RSO` | `rbp_rsw` |
| `RBP_RSE` | `rbp_rse` |
| `RBP_mean` | `rbp_mean` |

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

rocloud_ep = rocloud_text_to_dataframe(
    "data/rocloud/EP_TCSize_2000_2024.dat",
)

rocloud_na = rocloud_text_to_dataframe(
    "data/rocloud/NA_TCSize_2000_2024.dat",
)
```

To keep only canonical synoptic times:

```python
rocloud_ep = rocloud_text_to_dataframe(
    "data/rocloud/EP_TCSize_2000_2024.dat",
    synoptic_only=True,
)
```

`read_rocloud_table` can still be used for generic loading. For `.dat`, `.txt` and `.dot` files it auto-detects the text layout. For `.csv` and `.parquet` it preserves the existing behavior.

---

## Data policy for repository tests

The full `NA_TCSize_2000_2024.dat` and `EP_TCSize_2000_2024.dat` files should not be committed as test fixtures unless the repository explicitly decides to version external research data.

Recommended practice:

```text
Use minimal representative text fixtures in tests.
Keep full datasets outside the repository or download them through a documented data-acquisition step.
```

This keeps the repository lightweight and avoids coupling unit tests to a large external data artifact.

---

## Scientific constraint

The ROCLOUD radius is the outer precipitation-attribution boundary used by ITCHI. It must remain in kilometers and should not be conflated with R34 or with any accumulated precipitation quantity.

If all four ROCLOUD quadrants are missing for a snapshot, the attribution radius cannot be resolved and the pipeline should fail. If only some quadrants are missing, the existing radius-resolution utilities may fill missing quadrants using the configured strategy, currently `mean_available`.
