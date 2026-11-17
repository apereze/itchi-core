"""
Constants used across the ITCHI core package.

ITCHI v0.1 is designed as a physical tropical cyclone hazard index
bounded between 0 and 1.
"""

from __future__ import annotations

# ============================================================
# Package metadata
# ============================================================

PROJECT_NAME: str = "itchi-core"
ITCHI_VERSION: str = "0.2.2"


# ============================================================
# Index bounds
# ============================================================

ITCHI_MIN_VALUE: float = 0.0
ITCHI_MAX_VALUE: float = 1.0


# ============================================================
# Synoptic times
# ============================================================

SYNOPTIC_HOURS_UTC: tuple[int, int, int, int] = (0, 6, 12, 18)


# ============================================================
# Precipitation percentiles
# ============================================================

PRECIP_PERCENTILE_LOW: int = 90
PRECIP_PERCENTILE_HIGH: int = 95
PRECIP_PERCENTILE_MAXIMUM: int = 99

PRECIP_PERCENTILES: tuple[int, int, int] = (
    PRECIP_PERCENTILE_LOW,
    PRECIP_PERCENTILE_HIGH,
    PRECIP_PERCENTILE_MAXIMUM,
)


# ============================================================
# Cyclone quadrants
# ============================================================

QUADRANTS: tuple[str, str, str, str] = ("RNE", "RSE", "RSW", "RNW")


# ============================================================
# Wind thresholds
# ============================================================

TROPICAL_STORM_WIND_KT: float = 34.0

# Saffir-Simpson lower bounds (1-min sustained wind, kt; NHC): tropical
# storm and hurricane categories 1-5. 96 kt is the major-hurricane cut.
SAFFIR_SIMPSON_KT: dict[str, float] = {
    "TS": 34.0,
    "H1": 64.0,
    "H2": 83.0,
    "H3": 96.0,
    "H4": 113.0,
    "H5": 137.0,
}

# Absolute wind hazard homologated with Saffir-Simpson, default since
# v0.2.2 (agreed after the v0.2.1 progress report). (wind kt, H_W) nodes,
# linear between nodes, constant above the last one. The SIAT-style classes
# (steps of 0.2) then mean:
#   class 1 (0-0.2]   tail below 34 kt (from the tail-end wind, 12 m/s)
#   class 2 (0.2-0.4] tropical storm, 34-63 kt
#   class 3 (0.4-0.6] moderate: hurricane category 1, 64-82 kt
#   class 4 (0.6-0.8] high: hurricane category 2, 83-95 kt
#   class 5 (0.8-1]   maximum: major hurricane (categories 3-5), >= 96 kt;
#                     H_W keeps growing to 1 at 137 kt (H5) inside class 5
# The classes are (lower, upper]: a wind exactly on a bound (e.g. 96 kt,
# H_W = 0.8) falls in the lower class; IBTrACS Vmax (multiples of 5 kt)
# never equals 34, 64, 83 or 96 kt.
SAFFIR_SIMPSON_HAZARD_NODES: tuple[tuple[float, float], ...] = (
    (34.0, 0.2),
    (64.0, 0.4),
    (83.0, 0.6),
    (96.0, 0.8),
    (137.0, 1.0),
)

# v0.2.1 progress-report proposal, kept to reproduce it
# (configs/experiments/v021/ss.yaml and p95_ss.yaml):
# tail -> 1, TS -> 2, H1-H2 -> 3, H3 -> 4, H4-H5 -> 5.
SAFFIR_SIMPSON_HAZARD_NODES_V021: tuple[tuple[float, float], ...] = (
    (34.0, 0.2),
    (64.0, 0.4),
    (96.0, 0.6),
    (113.0, 0.8),
    (137.0, 1.0),
)


# ============================================================
# Numerical stability
# ============================================================

DEFAULT_EPSILON: float = 1.0e-6
