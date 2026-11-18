"""
Tests for itchi.cases and itchi.municipal on a synthetic case: track, rain
fields, percentiles and municipalities built in memory (no MSWEP, IBTrACS or
shapefile).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from itchi import cases as cv
from itchi import municipal as mu


def _synthetic_case(nt: int = 6):
    lat = np.arange(15.0, 25.0, 0.1)
    lon = np.arange(-105.0, -95.0, 0.1)
    t_all = pd.date_range("2014-09-10 00:00", periods=nt + 16, freq="6h")
    rng = np.random.default_rng(3)
    p6 = rng.gamma(0.6, 6.0, (t_all.size, lat.size, lon.size))
    # banda de lluvia intensa que se mueve con el ciclón
    lon2d, lat2d = np.meshgrid(lon, lat)
    track_t = t_all[12:12 + nt]
    c_lon = np.linspace(-103.0, -99.0, nt)
    c_lat = np.linspace(17.0, 20.0, nt)
    for i, t in enumerate(track_t):
        j = t_all.get_loc(t)
        d2 = (lon2d - c_lon[i]) ** 2 + (lat2d - c_lat[i]) ** 2
        p6[j] += 60.0 * np.exp(-d2 / 0.5)
    da = xr.DataArray(p6, dims=("time", "lat", "lon"), coords={"time": t_all, "lat": lat, "lon": lon})
    r24 = cv.compute_antecedent_precipitation(da, hours=24, exclude_current=False)
    a72 = cv.compute_antecedent_precipitation(da, hours=72, exclude_current=True)
    q6 = {"Q75": np.full(lon2d.shape, 8.0), "Q90": np.full(lon2d.shape, 18.0), "Q95": np.full(lon2d.shape, 28.0),
          "Q99": np.full(lon2d.shape, 50.0)}
    q24 = {"Q75": np.full(lon2d.shape, 20.0), "Q90": np.full(lon2d.shape, 40.0), "Q95": np.full(lon2d.shape, 55.0)}
    q72 = {"Q75": np.full(lon2d.shape, 30.0), "Q90": np.full(lon2d.shape, 60.0)}
    vmax = np.array([45.0, 70.0, 100.0, 120.0, 60.0, 30.0])[:nt]
    track = pd.DataFrame({
        "time": track_t, "lon": c_lon, "lat": c_lat, "vmax_kt": vmax, "rmw_km": np.full(nt, 30.0),
        "r34_rne": [60.0, 80, 90, 100, 70, np.nan][:nt], "r34_rse": [50.0, 70, 80, 90, 60, np.nan][:nt],
        "r34_rsw": [40.0, 50, 60, 70, 40, np.nan][:nt], "r34_rnw": [45.0, 60, 70, 80, 50, np.nan][:nt],
    })
    for q, col in cv.DEFAULT_ROCLOUD_COLUMN_MAP.items():
        track[col] = np.nan if q == "RSW" else 500.0
    # Municipios: bloques de 1° x 1°; uno de ellos de un solo píxel (n = 0)
    pix, nx = {}, lon.size
    for a in range(3):
        for b in range(4):
            jj = np.arange(20 + 10 * a, 30 + 10 * a)
            ii = np.arange(20 + 10 * b, 30 + 10 * b)
            J, I = np.meshgrid(jj, ii, indexing="ij")
            pix[f"{a}{b:04d}"] = ((J * nx + I).ravel(), J.size)
    pix["99999"] = (np.array([5 * nx + 5]), 0)
    return track, da.sel(time=slice(None)), r24, a72, q6, q24, q72, pix


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    track, p6, r24, a72, q6, q24, q72, pix = _synthetic_case()
    for col in cv.DEFAULT_R34_COLUMN_MAP.values():
        assert col in track
    overlay = tmp_path_factory.mktemp("cfg") / "cola_rw06.yaml"
    overlay.write_text("wind:\n  tail_end_wind_ms: null\n", encoding="utf-8")
    settings = cv.experiment_settings(cv.CaseConfig(experiments={"itchi": None, "cola_rw06": str(overlay)}))
    ds = cv.compute_case_fields(track, p6, r24, a72, q6, q24, q72, settings, verbose=False)
    codes = list(pix)
    land = mu.land_mask(pix, (p6["lat"].size, p6["lon"].size))
    return dict(case="SINTETICO", ds=ds, track=track, pix=pix, codes=codes, land=land)


def test_default_experiment() -> None:
    s = cv.experiment_settings()
    assert list(s) == ["itchi"]
    assert s["itchi"]["precip_matrix"]["p6_nodes"][-1] == "Q95"
    assert s["itchi"]["wind_tail_end_kt"] == pytest.approx(12.0 / 0.514444)


def test_fields_shapes_and_physics(case) -> None:
    ds = case["ds"]
    assert ds["ITCHI"].dims == ("exp", "time", "lat", "lon")
    assert ds.sizes["time"] == len(case["track"])
    w12, w = ds["H_W"].sel(exp="itchi"), ds["H_W"].sel(exp="cola_rw06")
    assert int((w12 > 0).sum()) < int((w > 0).sum())               # el corte en 12 m/s reduce el área de la cola
    np.testing.assert_allclose(ds["H_P"].sel(exp="itchi"), ds["H_P"].sel(exp="cola_rw06"))   # el viento no toca la lluvia
    # Tormenta de 45 kt (primera posición): clase 2; huracán de 100 kt (tercera): clase 5
    assert 0.2 < float(w12.isel(time=0).max()) <= 0.4
    assert float(w12.isel(time=2).max()) > 0.8
    # Dentro de la región de atribución ITCHI es la unión de lluvia y viento: nunca menor que cada uno
    region = (ds["M_direct"] | ds["M_indirect"]).values.astype(bool)
    it = ds["ITCHI"].sel(exp="itchi").values
    assert (it[region] - np.maximum(w12.values, ds["H_P"].sel(exp="itchi").values)[region]).min() >= -1e-6
    assert (it[~region] == 0.0).all()
    # H_P continuo igual a H_P del pipeline donde hay región de atribución
    j = list(pd.DatetimeIndex(ds["time_all"].values)).index(pd.Timestamp(ds["time"].values[2]))
    reg = (ds["M_direct"].isel(time=2) | ds["M_indirect"].isel(time=2)).values.astype(bool)
    np.testing.assert_allclose(ds["H_P_cont"].sel(exp="itchi").isel(time_all=j).values[reg],
                               ds["H_P"].sel(exp="itchi").isel(time=2).values[reg], atol=1e-6)


def test_municipal_values(case) -> None:
    ds, pix, codes = case["ds"], case["pix"], case["codes"]
    f = ds["ITCHI"].sel(exp="itchi").values
    mx = mu.municipal_values(f, pix, codes, "max")
    mean = mu.municipal_values(f, pix, codes, "mean")
    p90 = mu.municipal_values(f, pix, codes, "p90")
    assert mx.shape == (len(codes), ds.sizes["time"])
    assert np.all(mx >= p90 - 1e-9) and np.all(mx >= mean - 1e-6)      # el máximo acota a los demás estadísticos
    k = codes.index("10001")
    flat = pix["10001"][0]
    np.testing.assert_allclose(mx[k], f.reshape(f.shape[0], -1)[:, flat].max(axis=1))
    # pintar y volver a leer da el mismo valor municipal
    painted = mu.paint_municipal(mx[:, 2], pix, codes, f.shape[1:])
    np.testing.assert_allclose(painted.ravel()[flat], mx[k, 2])


def test_class_counts_and_event(case) -> None:
    mun = cv.municipal_event(case)
    assert mun.dims == ("exp", "mun", "time")
    tab = cv.class_counts(case)
    assert set(tab.columns.get_level_values(0)) == {"municipios", "pixeles_tierra"}
    # cada experimento cuenta a lo más todos los municipios
    assert int(tab["municipios"].sum().max()) <= len(case["codes"])


def test_lead_times() -> None:
    times = pd.date_range("2013-09-14 00:00", periods=8, freq="6h")
    cls = np.array([3, 1, 2, 3, 3, 4, 5, 5])
    out = cv.lead_times(times, cls, "2013-09-15 14:30", levels=(3, 4, 5)).set_index("nivel")
    # nivel 3: hubo un episodio aislado a las 00 UTC del 14; el último empieza a las 18 UTC
    assert out.loc[3, "primer_tiempo"] == pd.Timestamp("2013-09-14 00:00")
    assert out.loc[3, "inicio_episodio"] == pd.Timestamp("2013-09-14 18:00")
    assert out.loc[3, "anticipacion_h"] == pytest.approx(20.5)
    assert out.loc[3, "horas_desde_fin"] == pytest.approx(2.5)
    assert out.loc[4, "anticipacion_h"] == pytest.approx(8.5)
    assert out.loc[5, "anticipacion_h"] == pytest.approx(2.5)
    # el paso de las 18 UTC está después del evento: no cuenta
    late = cv.lead_times(times, np.array([0, 0, 0, 0, 0, 0, 0, 5]), "2013-09-15 14:30", levels=(5,))
    assert np.isnan(late["anticipacion_h"].iloc[0])
    # serie interrumpida (sin posición del ciclón): el episodio terminó 14.5 h antes
    gap = cv.lead_times(times[:5], np.array([0, 2, 3, 3, 0]), "2013-09-15 14:30", levels=(3,))
    assert gap["horas_desde_fin"].iloc[0] == pytest.approx(20.5)
    # la ventana excluye lo anterior a window_h
    win = cv.lead_times(times, cls, "2013-09-15 14:30", levels=(3,), window_h=24.0)
    assert win["primer_tiempo"].iloc[0] == pd.Timestamp("2013-09-14 18:00")


def test_percentile_level() -> None:
    q = {"Q75": 8.0, "Q90": 18.0, "Q95": 28.0, "Q99": 50.0}
    np.testing.assert_array_equal(cv.percentile_level(np.array([1.0, 10.0, 30.0, 60.0]), q), [0, 75, 95, 99])


def test_select_positions() -> None:
    track = pd.DataFrame({"lat": [18.0, 30.0], "lon": [-100.0, -60.0], "pa21_r2_km": [400.0, 400.0]})
    for col in cv.DEFAULT_ROCLOUD_COLUMN_MAP.values():
        track[col] = np.nan
    kept = cv.select_positions(track, np.array([19.0]), np.array([-100.0]))
    assert len(kept) == 1 and kept["lat"].iloc[0] == 18.0


def test_tabla_ordinal_and_find_municipio(case) -> None:
    tab = cv.tabla_ordinal(case, decl_codes={"10001", "10002"})
    assert set(tab["clase"]) == set(range(6))
    assert (tab["con_declaratoria"] <= tab["municipios"]).all()
    assert int(tab.loc[tab["exp"] == "itchi", "con_declaratoria"].sum()) <= 2
    gdf = pd.DataFrame({"CVEGEO": ["12011", "21071", "12001"], "NOMGEO": ["Atoyac de Álvarez", "Huauchinango", "Acapulco de Juárez"]})
    assert mu.find_municipio(gdf, "atoyac de alvarez", "12") == "12011"
    assert mu.find_municipio(gdf, "Huauchinango", 21) == "21071"
    with pytest.raises(ValueError):
        mu.find_municipio(gdf, "Atoyac de Álvarez", "14")
    # acento mal decodificado (UTF-8 leído como latin-1) y respaldo por CVEGEO
    bad = gdf.assign(NOMGEO=gdf["NOMGEO"].map(lambda x: x.encode("utf-8").decode("latin-1")))
    assert "Ã" in bad["NOMGEO"].iloc[0]
    assert mu.find_municipio(bad, "Atoyac de Álvarez", "12") == "12011"
    other = gdf.assign(NOMGEO=["Atoyac", "Huauchinango", "Acapulco de Juárez"])
    with pytest.warns(UserWarning):
        assert mu.find_municipio(other, "Atoyac de Álvarez", "12", cvegeo="12011") == "12011"


def test_declaratoria_inicio() -> None:
    decl = pd.DataFrame({"CVEGEO": ["12011", "12011", "12001"], "CT_pico": ["EP132013", "EP132013", "AL102013"],
                         "fecha_inicio": ["2013-09-17", "2013-09-15", "2013-09-14"]})
    out = cv.declaratoria_inicio(decl, "EP132013")
    assert out.loc["12011"] == pd.Timestamp("2013-09-15 06:00")
    assert "12001" not in out.index
    assert cv.declaratoria_inicio(None, "EP132013").empty
