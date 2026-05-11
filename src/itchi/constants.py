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
ITCHI_VERSION: str = "0.1.0"


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


# ============================================================
# Numerical stability
# ============================================================

DEFAULT_EPSILON: float = 1.0e-6
