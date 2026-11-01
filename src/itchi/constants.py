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
ITCHI_VERSION: str = "0.2.1"


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

# v0.2.1 proposal: absolute wind hazard homologated with Saffir-Simpson.
# (wind kt, H_W) nodes, linear between nodes, constant above the last one.
# The SIAT-style classes (steps of 0.2) then mean:
#   class 1 (0-0.2]   tail below 34 kt (from the tail-end wind, e.g. 12 m/s)
#   class 2 (0.2-0.4] tropical storm, 34-63 kt
#   class 3 (0.4-0.6] hurricane categories 1-2, 64-95 kt
#   class 4 (0.6-0.8] category 3, 96-112 kt
#   class 5 (0.8-1]   categories 4-5, >= 113 kt (H_W = 1 from 137 kt)
SAFFIR_SIMPSON_HAZARD_NODES: tuple[tuple[float, float], ...] = (
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
