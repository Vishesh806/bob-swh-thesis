"""Physically based feature engineering for Bay of Bengal wave forecasting.

Every feature at time t uses data at times <= t only: differences look
backwards, and rolling windows cover (t - w, t]. Nothing here shifts a series
forward, so the feature matrix can be built once and windowed without
look-ahead.

References
----------
Large, W. G. and Pond, S. (1981). Open ocean momentum flux measurements in
    moderate to strong winds. J. Phys. Oceanogr., 11, 324-336.
Wu, J. (1982). Wind-stress coefficients over sea surface from breeze to
    hurricane. J. Geophys. Res., 87(C12), 9704-9706.
Powell, M. D., Vickery, P. J. and Reinhold, T. A. (2003). Reduced drag
    coefficient for high wind speeds in tropical cyclones. Nature, 422, 279-283.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

DRAG_SCHEMES = ("large_pond", "wu1982", "constant")


# ---------------------------------------------------------------- wind
def wind_speed_direction(u, v) -> tuple[np.ndarray, np.ndarray]:
    """Speed (m s-1) and meteorological direction (deg, the direction the wind blows FROM).

    theta = atan2(-u, -v) mod 360, so a northerly (u = 0, v < 0) is 0 deg and an
    easterly (u < 0, v = 0) is 90 deg, the same convention as ERA5 ``mwd``.
    """
    u = np.asarray(u, float)
    v = np.asarray(v, float)
    return np.hypot(u, v), np.degrees(np.arctan2(-u, -v)) % 360.0


def wind_to_10m(speed, z: float, z0: float = config.SEA_ROUGHNESS_M):
    """Reduce wind measured at height z (m) to 10 m with the neutral log law
    U10 = Uz ln(10 / z0) / ln(z / z0)."""
    return speed * np.log(10.0 / z0) / np.log(z / z0)


def air_density(mslp_hpa, t_air_c) -> np.ndarray:
    """Dry-air density from the ideal-gas law, rho = p / (R_d T), in kg m-3.

    Ignoring humidity overestimates rho by about 1 % in tropical marine air.
    """
    return np.asarray(mslp_hpa, float) * 100.0 / (config.R_DRY_AIR * (np.asarray(t_air_c, float) + 273.15))


def drag_coefficient(speed, scheme: str = "large_pond", cd_max: float = 2.5e-3) -> np.ndarray:
    """Neutral 10 m drag coefficient C_d(U10).

    large_pond: 1.2e-3 for U < 11 m s-1, (0.49 + 0.065 U) 1e-3 for 11 <= U <= 25 m s-1,
                held at its 25 m s-1 value above that (outside the fitted range).
    wu1982:     (0.8 + 0.065 U) 1e-3, capped at ``cd_max``. Field data in tropical
                cyclones show C_d levelling off at high wind speeds
                (Powell et al. 2003), so it does not keep growing linearly.
    constant:   1.3e-3.
    """
    u = np.clip(np.asarray(speed, float), 0.0, None)
    if scheme == "large_pond":
        uc = np.minimum(u, 25.0)
        return np.where(uc < 11.0, 1.2e-3, (0.49 + 0.065 * uc) * 1e-3)
    if scheme == "wu1982":
        return np.minimum((0.8 + 0.065 * u) * 1e-3, cd_max)
    if scheme == "constant":
        return np.full_like(u, 1.3e-3)
    raise ValueError(f"scheme must be one of {DRAG_SCHEMES}, got {scheme!r}")


def wind_stress(u10, v10, rho_air=config.RHO_AIR_FALLBACK, scheme: str = "large_pond") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bulk surface wind stress (N m-2): tau = rho_a C_d(|U|) |U| U.

    Returns (tau_x, tau_y, |tau|). tau_x and tau_y point in the direction the
    wind blows TOWARDS (the sense of momentum transfer to the sea surface).
    """
    u = np.asarray(u10, float)
    v = np.asarray(v10, float)
    speed = np.hypot(u, v)
    k = np.asarray(rho_air, float) * drag_coefficient(speed, scheme) * speed
    return k * u, k * v, k * speed


# ---------------------------------------------------------------- directions and waves
def cyclical_encode(degrees) -> tuple[np.ndarray, np.ndarray]:
    """(sin theta, cos theta) of a direction. Removes the 359 -> 0 deg discontinuity:
    1 deg and 359 deg map to points 0.035 apart instead of 358 units apart."""
    rad = np.deg2rad(np.asarray(degrees, float))
    return np.sin(rad), np.cos(rad)


def angular_difference(a_deg, b_deg) -> np.ndarray:
    """Signed smallest difference a - b, in (-180, 180] degrees."""
    d = (np.asarray(a_deg, float) - np.asarray(b_deg, float) + 180.0) % 360.0 - 180.0
    return np.where(d == -180.0, 180.0, d)


def wave_steepness(hs, period) -> np.ndarray:
    """Deep-water steepness Hs / L0, with L0 = g T^2 / (2 pi). NaN where T <= 0."""
    t = np.asarray(period, float)
    l0 = config.GRAVITY * t ** 2 / (2.0 * np.pi)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(t > 0, np.asarray(hs, float) / l0, np.nan)


def step_hours(index: pd.DatetimeIndex) -> float:
    """Sampling step of a regular index in hours; raises if the index is irregular."""
    if len(index) < 2:
        raise ValueError("need at least two timestamps")
    steps = np.unique(np.diff(index.values))
    if steps.size != 1:
        raise ValueError("index is not regular; run preprocess.regularise first")
    return pd.Timedelta(steps[0]) / pd.Timedelta(hours=1)


# ---------------------------------------------------------------- feature engineer
@dataclass
class FeatureEngineer:
    """Add physically motivated, strictly causal features to a merged ERA5 + buoy frame.

    Features are added only when their source columns exist, so the same
    object serves the ERA5-only and the ERA5 + buoy configurations.
    """

    u_col: str = "era5_u10"
    v_col: str = "era5_v10"
    mslp_col: str = "era5_msl"
    t2m_col: str = "era5_t2m"
    sst_col: str = "era5_sst"
    wave_dir_col: str = "era5_mwd"
    model_hs_col: str = "era5_swh"
    model_period_col: str = "era5_mwp"
    hs_cols: tuple[str, ...] = ("buoy_hs", "era5_swh")
    drag_scheme: str = "large_pond"
    stress_windows_h: tuple[int, ...] = (6, 12, 24)
    pressure_tendency_h: tuple[int, ...] = (3, 12, 24)
    pressure_anomaly_days: int = 30
    hs_tendency_h: tuple[int, ...] = (3, 6, 12)
    calendar: bool = True

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        dt = step_hours(df.index)

        def steps(hours: float) -> int:
            return max(int(round(hours / dt)), 1)

        out = df.copy()
        new: dict[str, pd.Series | np.ndarray] = {}

        if self.u_col in out and self.v_col in out:
            speed, wdir = wind_speed_direction(out[self.u_col], out[self.v_col])
            new["era5_wspd10"] = speed
            new["era5_wdir10"] = wdir
            if self.mslp_col in out and self.t2m_col in out:
                rho = air_density(out[self.mslp_col], out[self.t2m_col])
            else:
                rho = np.full(len(out), config.RHO_AIR_FALLBACK)
            tx, ty, tau = wind_stress(out[self.u_col], out[self.v_col], rho, self.drag_scheme)
            new.update({"era5_rho_air": rho, "era5_tau_x": tx, "era5_tau_y": ty, "era5_tau": tau})
            tau_s = pd.Series(tau, index=out.index)
            spd_s = pd.Series(speed, index=out.index)
            for w in self.stress_windows_h:
                n = steps(w)
                new[f"era5_tau_mean{w}h"] = tau_s.rolling(n, min_periods=n).mean()
                new[f"era5_wspd10_max{w}h"] = spd_s.rolling(n, min_periods=n).max()

        if self.mslp_col in out:
            p = out[self.mslp_col]
            for h in self.pressure_tendency_h:
                new[f"era5_msl_tend{h}h"] = p.diff(steps(h))          # p(t) - p(t - h)
            n = steps(self.pressure_anomaly_days * 24)
            new["era5_msl_anom"] = p - p.rolling(n, min_periods=n // 2).mean()

        if self.t2m_col in out and self.sst_col in out:
            new["era5_air_sea_dt"] = out[self.t2m_col] - out[self.sst_col]

        if self.model_hs_col in out and self.model_period_col in out:
            new["era5_steepness"] = wave_steepness(out[self.model_hs_col], out[self.model_period_col])

        for hs in self.hs_cols:
            if hs in out:
                for h in self.hs_tendency_h:
                    new[f"{hs}_tend{h}h"] = out[hs].diff(steps(h))

        out = pd.concat([out, pd.DataFrame(new, index=out.index)], axis=1)

        circ: dict[str, np.ndarray] = {}
        for col in [c for c in out.columns if config.is_circular(c)]:
            circ[f"{col}_sin"], circ[f"{col}_cos"] = cyclical_encode(out[col])
        if "era5_wdir10" in out and self.wave_dir_col in out:
            # cos of the wind-wave angle: +1 for a wind sea aligned with the wind,
            # <= 0 for swell crossing or opposing the local wind
            circ["era5_wind_wave_cos"] = np.cos(np.deg2rad(angular_difference(out["era5_wdir10"], out[self.wave_dir_col])))
        if self.calendar:
            doy = out.index.dayofyear.to_numpy() - 1 + out.index.hour.to_numpy() / 24.0
            ang = 2.0 * np.pi * doy / 365.25
            circ["doy_sin"], circ["doy_cos"] = np.sin(ang), np.cos(ang)
        return pd.concat([out, pd.DataFrame(circ, index=out.index)], axis=1)

    @staticmethod
    def feature_columns(df: pd.DataFrame) -> list[str]:
        """Numeric model inputs: every column except raw directions in degrees."""
        return [c for c in df.columns if not config.is_circular(c) and pd.api.types.is_numeric_dtype(df[c])]
