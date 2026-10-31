"""
Return periods of precipitation for ITCHI (climatology of rainfall rarity).

The hazard of a rainfall amount is expressed as how often it is reached at the
same pixel, not as a percentile of wet windows. Two estimators per pixel and
duration d (6, 24, 72 h):

1. Annual (seasonal) maxima + GEV (block maxima)
   M_y = max of R_d in season y (May-November). T(x) = 1 / P(M > x): the
   return period in years. Reliable for T >= 2 years; with 26 seasons the
   empirical estimate is usable up to ~10 years and the GEV extrapolates.

2. Declustered peaks over threshold + GPD (POT)
   Independent events: peaks of R_d separated by at least ``min_separation``
   steps. The K largest peaks of the record define the threshold u (the
   (K+1)-th peak) and the excesses are fitted with a generalized Pareto
   distribution. lambda(x) = mean number of events per season with peak >= x;
   R(x) = 1 / lambda(x) is the mean recurrence interval in years, valid also
   below one year (e.g. R = 0.33 years: three events per season).

   For a Poisson number of events, the two scales are related by
   T = 1 / (1 - exp(-1 / R)) (annual_return_period).

Conventions
-----------
- Arrays have time (or sample) on axis 0 and any number of trailing pixel axes.
- Shape parameter xi follows Coles (2001): xi > 0 heavy tail, xi < 0 bounded.
  Hosking's k = -xi is used internally by the L-moment estimators.
- Pure NumPy (no SciPy, no xarray) so it runs inside dask blocks and in tests.

References
----------
Coles, S. (2001). An Introduction to Statistical Modeling of Extreme Values.
    Springer. https://doi.org/10.1007/978-1-4471-3675-0
Hosking, J. R. M. (1990). L-moments: analysis and estimation of distributions
    using linear combinations of order statistics. J. R. Stat. Soc. B, 52(1),
    105-124. https://doi.org/10.1111/j.2517-6161.1990.tb01775.x
Hosking, J. R. M. and Wallis, J. R. (1987). Parameter and quantile estimation
    for the generalized Pareto distribution. Technometrics, 29(3), 339-349.
    https://doi.org/10.1080/00401706.1987.10488243
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

EULER_GAMMA: float = 0.5772156649015329
_gamma = np.vectorize(math.gamma, otypes=[float])


# ============================================================
# Series operations (axis 0 = time, contiguous 6-h steps)
# ============================================================


def running_sum(x: np.ndarray, n: int) -> np.ndarray:
    """
    R(t) = x(t) + x(t-1) + ... + x(t-n+1) along axis 0.

    NaN if any of the n steps is NaN or if t < n - 1 (start of the series).
    """
    x = np.asarray(x, dtype=float)
    if n < 1:
        raise ValueError(f"n must be >= 1: {n}")
    if n == 1:
        return x.copy()
    out = np.full_like(x, np.nan)
    if x.shape[0] >= n:
        acc = x[n - 1:].copy()
        for k in range(1, n):
            acc += x[n - 1 - k:x.shape[0] - k]
        out[n - 1:] = acc
    return out


def declustered_peaks(x: np.ndarray, min_separation: int, threshold: float = 1.0) -> np.ndarray:
    """
    Independent event peaks along axis 0.

    A step t is a peak if x(t) >= threshold and x(t) is the largest value in
    [t - min_separation, t + min_separation]; on ties the first occurrence
    wins. Two peaks are therefore at least min_separation + 1 steps apart.

    Parameters
    ----------
    x : numpy.ndarray
        Series (time, ...) of one season; NaN = missing.
    min_separation : int
        Half-width of the window in steps (e.g. 8 steps = 48 h for 6-h data).
    threshold : float, default=1.0
        Minimum peak value (wet window).

    Returns
    -------
    numpy.ndarray
        Same shape as x: the peak value at peaks, NaN elsewhere.
    """
    x = np.asarray(x, dtype=float)
    v = np.where(np.isfinite(x), x, -np.inf)
    nt = v.shape[0]
    before = np.full_like(v, -np.inf)       # max over [t-s, t-1]
    after = np.full_like(v, -np.inf)        # max over [t+1, t+s]
    for s in range(1, min(min_separation, nt - 1) + 1):
        before[s:] = np.maximum(before[s:], v[:nt - s])
        after[:nt - s] = np.maximum(after[:nt - s], v[s:])
    peak = np.isfinite(x) & (x >= threshold) & (v > before) & (v >= after)
    return np.where(peak, x, np.nan)


def update_top_k(top: np.ndarray | None, candidates: np.ndarray, k: int) -> np.ndarray:
    """
    Keep the k largest values along axis 0, sorted in descending order.

    NaN means "no value"; the result is NaN-padded where fewer than k values
    exist.
    """
    c = np.asarray(candidates, dtype=float)
    pool = c if top is None else np.concatenate([np.asarray(top, dtype=float), c], axis=0)
    filled = np.where(np.isfinite(pool), pool, -np.inf)
    if filled.shape[0] > k:
        filled = -np.partition(-filled, k - 1, axis=0)[:k]
    filled = -np.sort(-filled, axis=0)
    if filled.shape[0] < k:
        pad = np.full((k - filled.shape[0],) + filled.shape[1:], -np.inf)
        filled = np.concatenate([filled, pad], axis=0)
    return np.where(np.isfinite(filled), filled, np.nan)


def count_at_or_above(peaks: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Number of peaks >= each threshold along axis 0.

    Parameters
    ----------
    peaks : numpy.ndarray
        (time, ...) peak values, NaN where there is no peak.
    thresholds : numpy.ndarray
        (n_thresholds, ...) thresholds per pixel; NaN gives a count of 0
        (mask afterwards).

    Returns
    -------
    numpy.ndarray
        (n_thresholds, ...) integer counts.
    """
    p = np.asarray(peaks, dtype=float)
    thr = np.asarray(thresholds, dtype=float)
    with np.errstate(invalid="ignore"):
        return (p[None, ...] >= thr[:, None, ...]).sum(axis=1)


def seasonal_maximum(x: np.ndarray, min_valid_fraction: float = 0.9) -> np.ndarray:
    """
    Maximum along axis 0; NaN if less than min_valid_fraction of the steps
    are valid (a truncated season would bias the maximum low).
    """
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(x).mean(axis=0)
    with np.errstate(invalid="ignore"):
        m = np.max(np.where(np.isfinite(x), x, -np.inf), axis=0)
    return np.where((valid >= min_valid_fraction) & np.isfinite(m), m, np.nan)


# ============================================================
# Sample L-moments (NaN-aware, axis 0)
# ============================================================


def sample_lmoments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Unbiased sample L-moments l1, l2, l3 and sample size n along axis 0.

    Uses the probability-weighted moments b0, b1, b2 of the ascending order
    statistics (Hosking, 1990); NaN values are ignored pixel by pixel.
    """
    x = np.asarray(x, dtype=float)
    s = np.sort(x, axis=0)                       # NaN last
    n = np.isfinite(x).sum(axis=0).astype(float)
    i = np.arange(1, x.shape[0] + 1, dtype=float).reshape((-1,) + (1,) * (x.ndim - 1))
    ok = i <= n
    xs = np.where(ok, s, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b0 = xs.sum(axis=0) / n
        b1 = (xs * (i - 1)).sum(axis=0) / (n * (n - 1))
        b2 = (xs * (i - 1) * (i - 2)).sum(axis=0) / (n * (n - 1) * (n - 2))
    l1 = b0
    l2 = 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    bad = n < 3
    return (np.where(bad, np.nan, l1), np.where(bad, np.nan, l2),
            np.where(bad, np.nan, l3), n)


# ============================================================
# GEV (seasonal maxima)
# ============================================================


def fit_gev_lmoments(
    maxima: np.ndarray,
    min_n: int = 15,
    xi_bounds: tuple[float, float] = (-0.5, 0.5),
) -> dict[str, np.ndarray]:
    """
    GEV parameters from seasonal maxima by L-moments (Hosking, 1990).

    Parameters
    ----------
    maxima : numpy.ndarray
        Seasonal maxima (n_seasons, ...).
    min_n : int, default=15
        Minimum number of valid seasons; NaN parameters below it.
    xi_bounds : tuple, default=(-0.5, 0.5)
        Bounds of the shape parameter. With ~26 seasons the at-site shape is
        very uncertain; the bounds avoid absurd extrapolations. The location
        and scale are recomputed for the bounded shape.

    Returns
    -------
    dict
        "xi", "sigma", "mu", "n", "xi_clipped" (bool).
    """
    l1, l2, l3, n = sample_lmoments(maxima)
    with np.errstate(invalid="ignore", divide="ignore"):
        t3 = l3 / l2
        c = 2.0 / (3.0 + t3) - math.log(2.0) / math.log(3.0)
        k = 7.8590 * c + 2.9554 * c ** 2                 # Hosking's k = -xi
    k_clip = np.clip(k, -xi_bounds[1], -xi_bounds[0])
    clipped = np.isfinite(k) & (k_clip != k)
    small = np.abs(k_clip) < 1e-6
    k_safe = np.where(small, 1.0, k_clip)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        g = _gamma(1.0 + k_safe)
        alpha = np.where(small, l2 / math.log(2.0), l2 * k_safe / ((1.0 - 2.0 ** (-k_safe)) * g))
        zeta = np.where(small, l1 - EULER_GAMMA * alpha, l1 - alpha * (1.0 - g) / k_safe)
    ok = (n >= min_n) & np.isfinite(alpha) & (alpha > 0)
    return {
        "xi": np.where(ok, -k_clip, np.nan),
        "sigma": np.where(ok, alpha, np.nan),
        "mu": np.where(ok, zeta, np.nan),
        "n": n,
        "xi_clipped": ok & clipped,
    }


def gev_return_level(T: float | np.ndarray, xi, sigma, mu) -> np.ndarray:
    """
    Value exceeded with probability 1/T per season (T in years, T > 1).
    """
    T = np.asarray(T, dtype=float)
    if np.any(T <= 1.0):
        raise ValueError("GEV return periods must be > 1 year.")
    y = -np.log(1.0 - 1.0 / T)
    xi, sigma, mu = (np.asarray(a, dtype=float) for a in (xi, sigma, mu))
    small = np.abs(xi) < 1e-6
    xi_safe = np.where(small, 1.0, xi)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        return np.where(small, mu - sigma * np.log(y), mu + sigma / xi_safe * (y ** (-xi_safe) - 1.0))


def gev_return_period(x, xi, sigma, mu) -> np.ndarray:
    """
    T(x) = 1 / P(seasonal maximum > x), in years (inf beyond an upper bound).
    """
    x, xi, sigma, mu = (np.asarray(a, dtype=float) for a in (x, xi, sigma, mu))
    z = (x - mu) / sigma
    small = np.abs(xi) < 1e-6
    xi_safe = np.where(small, 1.0, xi)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        base = 1.0 + xi_safe * z
        t = np.where(small, np.exp(-z), np.where(base > 0, base ** (-1.0 / xi_safe), 0.0))
        t = np.where((~small) & (xi_safe > 0) & (base <= 0), np.inf, t)   # below the lower bound
        p_exceed = -np.expm1(-t)                                          # 1 - exp(-t)
        T = 1.0 / p_exceed
    return np.where(np.isfinite(x) & np.isfinite(sigma), T, np.nan)


# ============================================================
# GPD on declustered peaks (POT)
# ============================================================


def fit_gpd_top_k(
    top: np.ndarray,
    n_seasons: float | np.ndarray,
    xi_bounds: tuple[float, float] = (-0.5, 0.5),
    min_excesses: int = 20,
) -> dict[str, np.ndarray]:
    """
    GPD fitted to the K largest declustered peaks above the (K+1)-th.

    Parameters
    ----------
    top : numpy.ndarray
        Peaks sorted in descending order (K+1, ...), NaN-padded.
    n_seasons : float or numpy.ndarray
        Number of seasons of the record (per pixel if it varies).
    xi_bounds : tuple, default=(-0.5, 0.5)
        Bounds of the shape parameter (scale recomputed if bounded).
    min_excesses : int, default=20
        Minimum number of excesses; NaN parameters below it.

    Returns
    -------
    dict
        "u" (threshold, mm), "sigma", "xi", "rate" (events per season above
        u), "n_exc", "xi_clipped" (bool).
    """
    top = np.asarray(top, dtype=float)
    finite = np.isfinite(top)
    n_peaks = finite.sum(axis=0)
    idx_u = np.clip(n_peaks - 1, 0, None)
    u = np.take_along_axis(top, idx_u[None, ...], axis=0)[0]
    n_exc = np.maximum(n_peaks - 1, 0)
    rows = np.arange(top.shape[0]).reshape((-1,) + (1,) * (top.ndim - 1))
    exc = np.where(finite & (rows < n_exc), top - u, np.nan)

    l1, l2, _, _ = sample_lmoments(exc)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = l1 / l2 - 2.0                               # Hosking's k = -xi
    xi = np.clip(-k, xi_bounds[0], xi_bounds[1])
    clipped = np.isfinite(k) & (xi != -k)
    sigma = (1.0 - xi) * l1                             # mean excess = sigma / (1 - xi)
    ok = (n_exc >= min_excesses) & np.isfinite(sigma) & (sigma > 0)
    rate = n_exc / np.asarray(n_seasons, dtype=float)
    return {
        "u": np.where(ok, u, np.nan),
        "sigma": np.where(ok, sigma, np.nan),
        "xi": np.where(ok, xi, np.nan),
        "rate": np.where(ok, rate, np.nan),
        "n_exc": n_exc,
        "xi_clipped": ok & clipped,
    }


def pot_return_level(R: float | np.ndarray, u, sigma, xi, rate) -> np.ndarray:
    """
    Value reached on average once every R years (lambda = 1 / R events per
    season). NaN where 1 / R > rate (more frequent than the threshold: use
    percentiles there).
    """
    R = np.asarray(R, dtype=float)
    u, sigma, xi, rate = (np.asarray(a, dtype=float) for a in (u, sigma, xi, rate))
    ratio = rate * R                                    # lambda_u / lambda
    small = np.abs(xi) < 1e-6
    xi_safe = np.where(small, 1.0, xi)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        x = np.where(small, u + sigma * np.log(ratio), u + sigma / xi_safe * (ratio ** xi_safe - 1.0))
    return np.where(ratio >= 1.0, x, np.nan)


def pot_recurrence_interval(x, u, sigma, xi, rate) -> np.ndarray:
    """
    Mean recurrence interval R(x) = 1 / lambda(x), in years.

    lambda(x) = rate * [1 + xi (x - u) / sigma]^(-1/xi) for x >= u. Below the
    threshold the GPD does not apply: returns NaN (R < 1 / rate). Above an
    upper bound (xi < 0) returns inf.
    """
    x, u, sigma, xi, rate = (np.asarray(a, dtype=float) for a in (x, u, sigma, xi, rate))
    z = (x - u) / sigma
    small = np.abs(xi) < 1e-6
    xi_safe = np.where(small, 1.0, xi)
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        base = 1.0 + xi_safe * z
        surv = np.where(small, np.exp(-z), np.where(base > 0, base ** (-1.0 / xi_safe), 0.0))
        R = 1.0 / (rate * surv)
    return np.where(np.isfinite(x) & (x >= u) & np.isfinite(sigma), R, np.nan)


def empirical_pot_level(top: np.ndarray, n_seasons: float, R: Sequence[float]) -> np.ndarray:
    """
    Empirical value reached once every R years: the (R^-1 * n_seasons)-th
    largest peak, interpolated linearly between ranks (rank m <-> lambda =
    m / n_seasons). NaN where the rank exceeds the stored peaks or is < 1.
    """
    top = np.asarray(top, dtype=float)
    out = []
    for r in np.atleast_1d(np.asarray(R, dtype=float)):
        m = n_seasons / r                     # fractional rank, 1-based
        if m < 1.0 or m > top.shape[0]:
            out.append(np.full(top.shape[1:], np.nan))
            continue
        lo = int(math.floor(m)) - 1
        hi = min(lo + 1, top.shape[0] - 1)
        w = m - math.floor(m)
        out.append((1.0 - w) * top[lo] + w * top[hi])
    return np.stack(out, axis=0)


# ============================================================
# Conversions
# ============================================================


def annual_return_period(R) -> np.ndarray:
    """
    Return period of the seasonal maximum from the mean recurrence interval of
    events (Poisson events): T = 1 / (1 - exp(-1/R)). R = 1 -> T = 1.58.
    """
    R = np.asarray(R, dtype=float)
    with np.errstate(divide="ignore"):
        return 1.0 / -np.expm1(-1.0 / R)


def recurrence_from_annual(T) -> np.ndarray:
    """Inverse of annual_return_period: R = -1 / ln(1 - 1/T), T > 1."""
    T = np.asarray(T, dtype=float)
    return -1.0 / np.log1p(-1.0 / T)


def events_per_season_above_percentile(q: float, wet_frequency, windows_per_season: float,
                                       duration_h: int = 6) -> np.ndarray:
    """
    Rough number of independent events per season above the wet-window
    percentile q: exceedances (1 - q) * wet_frequency * windows, divided by
    the overlap of consecutive d-hour windows (d / 6). It ignores the
    clustering of exceedances within one storm, so it is an upper bound.
    """
    return (1.0 - q) * np.asarray(wet_frequency, dtype=float) * windows_per_season / (duration_h / 6)
