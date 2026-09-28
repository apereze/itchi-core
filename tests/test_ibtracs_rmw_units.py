"""RMW unit handling in itchi.ibtracs (IBTrACS usa_rmw is in nautical miles)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from itchi.ibtracs import ibtracs_to_track_dataframe

SID = "2018293N13262"


def _synthetic_ibtracs() -> xr.Dataset:
    dims = ("storm", "date_time")
    return xr.Dataset(
        {
            "sid": (("storm",), np.array([SID.encode()])),
            "iso_time": (
                dims,
                np.array([[b"2018-10-24 00:00:00", b"2018-10-24 06:00:00"]]),
            ),
            "lat": (dims, [[22.0, 22.8]]),
            "lon": (dims, [[-106.2, -105.6]]),
            "usa_wind": (dims, [[100.0, 70.0]]),
            "usa_pres": (dims, [[965.0, 985.0]]),
            "usa_rmw": (dims, [[15.0, 20.0]]),
            "usa_r34": (dims + ("quadrant",), np.full((1, 2, 4), 100.0)),
        }
    )


def test_rmw_converted_from_nautical_miles() -> None:
    track = ibtracs_to_track_dataframe(_synthetic_ibtracs(), SID)
    np.testing.assert_allclose(track["rmw_km"], [15.0 * 1.852, 20.0 * 1.852])


def test_rmw_unit_override() -> None:
    track = ibtracs_to_track_dataframe(_synthetic_ibtracs(), SID, rmw_unit="km")
    np.testing.assert_allclose(track["rmw_km"], [15.0, 20.0])


def test_r34_left_in_source_units() -> None:
    # R34 conversion stays downstream (pipeline r34_unit="nm")
    track = ibtracs_to_track_dataframe(_synthetic_ibtracs(), SID)
    np.testing.assert_allclose(track["r34_rne"], [100.0, 100.0])
