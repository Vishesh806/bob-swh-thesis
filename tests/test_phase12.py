"""Phase 1 + 2 tests on synthetic ERA5 and buoy data (no downloads needed).

Run from the project root:  python -m pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bobswh import config
from bobswh.alignment import NearestNodeAligner, haversine_km
from bobswh.buoy import BuoyReader
from bobswh.era5 import ERA5PointExtractor
from bobswh.features import (FeatureEngineer, angular_difference, cyclical_encode, drag_coefficient,
                             wind_speed_direction, wind_stress)
from bobswh.pipeline import Phase12Pipeline
from bobswh.preprocess import Preprocessor, fill_short_gaps, regularise
from bobswh.windowing import ChronologicalSplitter, SlidingWindowGenerator, WindowSpec

TIMES = pd.date_range("2020-01-01", "2020-01-31 23:00", freq="1h")
SITE = (15.1, 88.13)
SITE_BD08 = (config.BUOYS["BD08"].lat, config.BUOYS["BD08"].lon)


# ---------------------------------------------------------------- synthetic ERA5
def _wave_value(t_hours, lat, lon):
    return 1.5 + 0.5 * np.sin(2 * np.pi * t_hours / 120.0) + 0.01 * lat + 0.001 * lon


def write_era5(folder):
    """Two new-CDS-style files: atmosphere on 0.25 deg, waves on 0.5 deg with land (NaN) west of 82 E.
    Latitude is stored north -> south as in real ERA5."""
    folder.mkdir(parents=True, exist_ok=True)
    th = np.arange(len(TIMES), dtype=float)

    lat = np.arange(20.0, 9.99, -0.25)
    lon = np.arange(80.0, 95.01, 0.25)
    shape = (len(TIMES), len(lat), len(lon))
    u = np.broadcast_to((5 + 5 * np.sin(2 * np.pi * th / 48))[:, None, None], shape).copy()
    v = np.full(shape, -2.0)
    sst = np.full(shape, 301.15)
    sst[:, :, lon < 81.0] = np.nan
    atm = xr.Dataset(
        {
            "u10": (("valid_time", "latitude", "longitude"), u),
            "v10": (("valid_time", "latitude", "longitude"), v),
            "msl": (("valid_time", "latitude", "longitude"), np.full(shape, 100800.0)),
            "t2m": (("valid_time", "latitude", "longitude"), np.full(shape, 300.15)),
            "sst": (("valid_time", "latitude", "longitude"), sst),
        },
        coords={"valid_time": TIMES, "latitude": lat, "longitude": lon, "number": 0},
    )
    atm.to_netcdf(folder / "data_stream-oper.nc", engine="scipy")

    wlat = np.arange(20.0, 9.99, -0.5)
    wlon = np.arange(80.0, 95.01, 0.5)
    T, LA, LO = np.meshgrid(th, wlat, wlon, indexing="ij")
    swh = _wave_value(T, LA, LO)
    swh[:, :, wlon < 82.0] = np.nan
    mwd = np.where(np.isnan(swh), np.nan, 355.0)
    mwp = np.where(np.isnan(swh), np.nan, 7.0)
    wav = xr.Dataset(
        {
            "swh": (("valid_time", "latitude", "longitude"), swh),
            "mwd": (("valid_time", "latitude", "longitude"), mwd),
            "mwp": (("valid_time", "latitude", "longitude"), mwp),
        },
        coords={"valid_time": TIMES, "latitude": wlat, "longitude": wlon},
    )
    wav.to_netcdf(folder / "data_stream-wave.nc", engine="scipy")


# ---------------------------------------------------------------- synthetic buoy
def buoy_frame() -> pd.DataFrame:
    th = np.arange(len(TIMES), dtype=float)
    hs = 1.2 + 0.4 * np.sin(2 * np.pi * th / 120.0)
    df = pd.DataFrame({
        "Date Time": TIMES.strftime("%Y-%m-%d %H:%M"),
        "Latitude": SITE[0], "Longitude": SITE[1],
        "SWH (m)": hs, "MWD (deg)": 350.0, "Tavg (s)": 6.5,
        "WS (m/s)": 6.0, "WD (deg)": 30.0, "SLP (hPa)": 1008.0, "SST (degC)": 28.5,
    })
    df.loc[100:101, "SWH (m)"] = np.nan          # 2 h gap  -> filled
    df.loc[200:205, "SWH (m)"] = np.nan          # 6 h gap  -> kept missing
    df.loc[300, "SWH (m)"] = -999.0              # sentinel -> 1 h gap, filled
    df.loc[400, "SWH (m)"] += 4.0                # isolated spike -> removed, filled
    df.loc[500:502, "SWH (m)"] += 3.0            # sustained peak -> kept
    df.loc[600, "SWH (m)"] = 25.0                # out of range -> removed
    df.loc[650, "SLP (hPa)"] = 999.0             # genuine low pressure -> kept
    return df


@pytest.fixture()
def project(tmp_path):
    write_era5(tmp_path / "era5")
    (tmp_path / "buoy" / "BDX").mkdir(parents=True)
    buoy_frame().to_csv(tmp_path / "buoy" / "BDX" / "bdx_2020.csv", index=False)
    return tmp_path


# ---------------------------------------------------------------- Phase 1
def test_haversine_known_distance():
    # one degree of latitude ~ 111.2 km
    assert haversine_km(0, 0, 1, 0) == pytest.approx(111.19, abs=0.05)


def test_nearest_node_skips_land():
    lat = np.arange(10.0, 20.01, 0.5)
    lon = np.arange(80.0, 95.01, 0.5)
    valid = np.ones((lat.size, lon.size), bool)
    valid[:, lon < 82.0] = False
    m = NearestNodeAligner(lat, lon, valid).query("coast", 15.0, 81.1)
    assert (m.lat[0], m.lon[0]) == (15.0, 82.0)


def test_extractor_matches_grid_values(project):
    ex = ERA5PointExtractor({"BDX": SITE})
    df = ex.extract(sorted((project / "era5").glob("*.nc")))["BDX"]
    rep = ex.report().set_index("grid_resolution_deg")
    assert (rep.loc[0.25, "node_lat"], rep.loc[0.25, "node_lon"]) == (15.0, 88.25)
    assert (rep.loc[0.5, "node_lat"], rep.loc[0.5, "node_lon"]) == (15.0, 88.0)
    th = np.arange(len(TIMES), dtype=float)
    np.testing.assert_allclose(df["era5_swh"].to_numpy(), _wave_value(th, 15.0, 88.0))
    assert df["era5_msl"].iloc[0] == pytest.approx(1008.0)        # Pa -> hPa
    assert df["era5_sst"].iloc[0] == pytest.approx(28.0)          # K -> degC
    assert len(df) == len(TIMES)


def test_buoy_reader_aliases_and_sentinels(project):
    raw = BuoyReader().read(project / "buoy" / "BDX" / "bdx_2020.csv")
    assert {"hs", "mwd", "tm", "wspd", "wdir", "mslp", "sst", "lat", "lon"} <= set(raw.columns)
    assert np.isnan(raw["hs"].iloc[300])
    assert raw["mslp"].iloc[650] == 999.0
    assert BuoyReader.position(raw) == pytest.approx(SITE)


def test_preprocessor_gap_rules(project):
    raw = BuoyReader().read(project / "buoy" / "BDX" / "bdx_2020.csv").drop(columns=["lat", "lon"])
    res = Preprocessor().run(raw)
    hs, imp = res.data["hs"], res.imputed["hs"]
    assert imp.iloc[100:102].all() and hs.iloc[100:102].notna().all()      # short gap filled
    assert hs.iloc[200:206].isna().all() and not imp.iloc[200:206].any()   # long gap left
    assert imp.iloc[300] and imp.iloc[400] and imp.iloc[600]                # sentinel, spike, out of range
    assert hs.iloc[400] < 3.0                                               # spike replaced
    assert (hs.iloc[500:503] > 3.5).all() and not imp.iloc[500:503].any()  # cyclone-like peak kept
    assert res.report.loc["hs", "n_spikes"] == 1


def test_ffill_is_causal_and_directions_wrap():
    idx = pd.date_range("2020-01-01", periods=6, freq="1h")
    df = pd.DataFrame({"hs": [1.0, 2.0, np.nan, np.nan, 5.0, 6.0], "mwd": [350.0, 350.0, np.nan, np.nan, 10.0, 10.0]}, index=idx)
    f, _ = fill_short_gaps(df, 3, "ffill")
    assert f["hs"].iloc[2:4].tolist() == [2.0, 2.0]      # never sees the 5.0 after the gap
    lin, _ = fill_short_gaps(df, 3, "linear")
    d = lin["mwd"].iloc[2:4].to_numpy()
    assert np.all((d > 340) | (d < 20))                  # bridged through north, not 180


def test_regularise_is_right_closed():
    idx = pd.to_datetime(["2020-01-01 00:00", "2020-01-01 00:30", "2020-01-01 01:00", "2020-01-01 01:30", "2020-01-01 02:00"])
    df = pd.DataFrame({"hs": [1.0, 2.0, 3.0, 4.0, 5.0], "wdir": [350.0, 10.0, 350.0, 10.0, 0.0]}, index=idx)
    r = regularise(df, "1h")
    assert r.loc["2020-01-01 01:00", "hs"] == pytest.approx(2.5)        # (00:00, 01:00] = {00:30, 01:00}
    w = r.loc["2020-01-01 01:00", "wdir"]
    assert min(w, 360 - w) == pytest.approx(0.0, abs=1e-6)              # circular mean of 10 and 350


# ---------------------------------------------------------------- Phase 2
def test_wind_direction_convention():
    _, d = wind_speed_direction([0.0, -1.0, 0.0, 1.0], [-1.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(d, [0.0, 90.0, 180.0, 270.0])           # N, E, S, W (from)


def test_wind_stress_values():
    tx, ty, tau = wind_stress([10.0, 20.0], [0.0, 0.0], 1.2, "large_pond")
    np.testing.assert_allclose(tau, [1.2 * 1.2e-3 * 100, 1.2 * (0.49 + 0.065 * 20) * 1e-3 * 400])
    np.testing.assert_allclose(tx, tau)
    np.testing.assert_allclose(ty, 0.0)
    assert drag_coefficient(40.0, "large_pond") == pytest.approx(drag_coefficient(25.0, "large_pond"))
    assert drag_coefficient(60.0, "wu1982") == pytest.approx(2.5e-3)


def test_cyclical_encoding_continuity():
    s1, c1 = cyclical_encode(1.0)
    s2, c2 = cyclical_encode(359.0)
    assert np.hypot(s1 - s2, c1 - c2) < 0.04
    assert angular_difference(10.0, 350.0) == pytest.approx(20.0)


def test_features_are_causal():
    idx = pd.date_range("2020-01-01", periods=200, freq="1h")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "era5_u10": rng.normal(5, 3, 200), "era5_v10": rng.normal(0, 3, 200),
        "era5_msl": rng.normal(1008, 2, 200), "era5_t2m": 28.0, "era5_sst": 29.0,
        "era5_swh": rng.uniform(1, 2, 200), "era5_mwp": 7.0, "era5_mwd": rng.uniform(0, 360, 200),
        "buoy_hs": rng.uniform(1, 2, 200),
    }, index=idx)
    fe = FeatureEngineer(pressure_anomaly_days=2)
    full = fe.transform(df)
    cut = 120
    head = fe.transform(df.iloc[:cut])
    # changing the future must not change any feature at or before the cut
    pd.testing.assert_frame_equal(full.iloc[:cut], head, check_freq=False)
    assert not any(c in fe.feature_columns(full) for c in ("era5_mwd", "era5_wdir10"))


def _ramp(n=500):
    idx = pd.date_range("2020-01-01", periods=n, freq="1h")
    return pd.DataFrame({"buoy_hs": np.arange(n, dtype=float), "era5_tau": np.arange(n, dtype=float) * 10}, index=idx)


def test_sequences_alignment_and_shapes():
    df = _ramp()
    spec = WindowSpec(target_col="buoy_hs", input_cols=("era5_tau",), lookback_h=72, horizons_h=(6, 12, 24, 48, 72))
    w = SlidingWindowGenerator(spec).sequences(df)
    assert w.X.shape[1:] == (73, 2) and w.Y.shape[1] == 5
    t0 = (w.anchors - df.index[0]) / pd.Timedelta(hours=1)
    np.testing.assert_allclose(w.X[:, -1, 0], t0)                 # last encoder step = origin
    np.testing.assert_allclose(w.X[:, 0, 0], t0 - 72)             # first encoder step = origin - L
    np.testing.assert_allclose(w.Y, t0.to_numpy()[:, None] + np.array([6, 12, 24, 48, 72]))
    assert w.anchors[-1] + pd.Timedelta(hours=72) <= df.index[-1]


def test_windows_drop_gaps_and_imputed_targets():
    df = _ramp()
    df.iloc[250, 0] = np.nan
    imputed = pd.DataFrame(False, index=df.index, columns=df.columns)
    imputed.iloc[300, 0] = True
    spec = WindowSpec(target_col="buoy_hs", input_cols=("era5_tau",))
    w = SlidingWindowGenerator(spec, imputed=imputed).sequences(df)
    bad = {df.index[250], df.index[300]}
    for a in w.anchors:
        inputs = pd.date_range(a - pd.Timedelta(hours=72), a, freq="1h")
        assert df.index[250] not in inputs
        targets = {a + pd.Timedelta(hours=h) for h in spec.horizons_h}
        assert not (targets & bad)
    X, Y = SlidingWindowGenerator(spec, imputed=imputed).tabular(df)
    assert list(X.index) == list(w.anchors)                        # same samples for trees and LSTM
    np.testing.assert_allclose(X["buoy_hs_lag24h"], ((X.index - df.index[0]) / pd.Timedelta(hours=1)) - 24)


def test_chronological_split_has_no_leakage():
    idx = pd.date_range("2020-01-01", "2020-03-31 23:00", freq="1h")
    bounds = {"train": ("2020-01-01", "2020-01-31 23:00"), "val": ("2020-02-01", "2020-02-29 23:00"),
              "test": ("2020-03-01", "2020-03-31 23:00")}
    sp = ChronologicalSplitter(bounds, lookback_h=72, max_horizon_h=72)
    s = sp.split(idx)
    tr, va, te = idx[s["train"]], idx[s["val"]], idx[s["test"]]
    assert tr.max() + pd.Timedelta(hours=72) < va.min() - pd.Timedelta(hours=72)
    assert va.max() + pd.Timedelta(hours=72) < te.min() - pd.Timedelta(hours=72)
    with pytest.raises(ValueError):
        ChronologicalSplitter({"a": ("2020-01-01", "2020-02-10"), "b": ("2020-02-01", "2020-03-01")}).split(idx)


# ---------------------------------------------------------------- end to end
def test_pipeline_end_to_end(project):
    sites = {"BDX": config.BuoySite("BDX", 0.0, 0.0)}     # wrong nominal position: file lat/lon must win
    pipe = Phase12Pipeline(project / "era5", project / "buoy", project / "out", sites=sites)
    path = pipe.run()["BDX"]
    df = pd.read_parquet(path)
    imputed = pd.read_parquet(project / "out" / "BDX_imputed.parquet")
    assert len(df) == len(TIMES)
    for col in ("buoy_hs", "era5_swh", "era5_tau", "era5_wdir10_sin", "era5_mwd_cos", "buoy_mwd_sin",
                "era5_msl_tend24h", "era5_wind_wave_cos", "era5_steepness", "buoy_hs_tend3h", "doy_sin"):
        assert col in df, col
    assert imputed["buoy_hs"].sum() == 5          # 2 h gap + sentinel + spike + out-of-range
    spec = WindowSpec(target_col="buoy_hs", input_cols=("era5_tau", "era5_msl", "era5_mwd_sin", "era5_mwd_cos"))
    w = SlidingWindowGenerator(spec, imputed=imputed).sequences(df)
    assert len(w) > 0 and np.isfinite(w.X).all() and np.isfinite(w.Y).all()


# ---------------------------------------------------------------- ERA5 time-series boxes
def test_site_boxes_and_timeseries_request():
    from bobswh.era5 import ERA5TimeseriesDownloader, site_box
    assert site_box(*SITE_BD08) == [19.0, 89.0, 17.5, 90.5]
    assert site_box(16.32, 87.98) == [17.0, 87.25, 15.5, 88.75]
    assert site_box(13.58, 84.11) == [14.25, 83.25, 12.75, 84.75]
    req = ERA5TimeseriesDownloader().request(*SITE_BD08)
    assert req["date"] == "2015-01-01/2024-12-31" and len(req["variable"]) == 8
    n, w, s, e = req["area"]
    assert n - s <= 2 and e - w <= 2                      # CDS maximum extent of 2 x 2 deg


def _box_file(path, lat0, lon0):
    lat = np.arange(lat0, lat0 + 1.01, 0.25)
    lon = np.arange(lon0, lon0 + 1.01, 0.25)
    t = pd.date_range("2020-01-01", periods=48, freq="1h")
    swh = np.broadcast_to((lat0 + np.arange(48) * 0.01)[:, None, None], (48, lat.size, lon.size)).copy()
    xr.Dataset({"swh": (("valid_time", "latitude", "longitude"), swh)},
               coords={"valid_time": t, "latitude": lat, "longitude": lon}).to_netcdf(path, engine="scipy")


def test_extractor_merges_per_site_boxes(tmp_path):
    _box_file(tmp_path / "ts_A.nc", 18.0, 89.0)
    _box_file(tmp_path / "ts_B.nc", 13.0, 84.0)
    files = sorted(tmp_path.glob("*.nc"))
    out = ERA5PointExtractor({"A": (18.4, 89.6), "B": (13.5, 84.2)}).extract(files)
    assert out["A"]["era5_swh"].iloc[0] == pytest.approx(18.0)      # each site read from its own box
    assert out["B"]["era5_swh"].iloc[0] == pytest.approx(13.0)
    with pytest.raises(ValueError, match="C"):
        ERA5PointExtractor({"A": (18.4, 89.6), "C": (5.0, 95.0)}).extract(files)


def test_drop_sparse_columns():
    from bobswh.preprocess import drop_sparse_columns
    df = pd.DataFrame({"a": np.arange(100.0), "b": np.r_[np.arange(5.0), np.full(95, np.nan)]})
    out, dropped = drop_sparse_columns(df, 0.9)
    assert dropped == ["b"] and list(out.columns) == ["a"]
