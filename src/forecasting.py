"""Demand forecasting methods and accuracy metrics (NumPy only, no ML libraries).

Every forecaster has the same signature::

    forecaster(y, dates, h, **params) -> np.ndarray of length h

``y`` is the observed daily demand up to (not including) the first forecast day and
``dates`` is the matching DatetimeIndex. Forecasts are floored at zero.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def ses_level(x: np.ndarray, alpha: float) -> float:
    """Final level of simple exponential smoothing, computed in closed form.

    Equivalent to l_t = alpha * x_t + (1 - alpha) * l_{t-1} started from l_{-1} = x_0.
    Only the last 500 points matter for the alphas used here.
    """
    x = np.asarray(x, dtype=float)[-500:]
    m = len(x)
    if m == 0:
        return 0.0
    weights = alpha * (1.0 - alpha) ** np.arange(m - 1, -1, -1)
    return float(weights @ x + (1.0 - alpha) ** m * x[0])


def seasonal_indices(y: np.ndarray, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Weekday index (7) from the last 52 weeks and month-of-year index (12) from complete years.

    An index of 1.2 means "20% above the average day / month".
    """
    n = len(y)
    m = min(n, 364)
    yy, dd = y[-m:], dates.dayofweek.to_numpy()[-m:]
    overall = yy.mean()
    if overall <= 0:
        return np.ones(7), np.ones(12)
    dow = np.array([yy[dd == k].mean() if (dd == k).any() else overall for k in range(7)]) / overall

    years, months = dates.year.to_numpy(), dates.month.to_numpy()
    acc, cnt = np.zeros(12), 0
    for yr in np.unique(years):
        in_year = years == yr
        if in_year.sum() < 365:  # use complete calendar years only
            continue
        year_mean = y[in_year].mean()
        if year_mean <= 0:
            continue
        for mth in range(1, 13):
            acc[mth - 1] += y[in_year & (months == mth)].mean() / year_mean
        cnt += 1
    month = acc / cnt if cnt else np.ones(12)
    return dow, month


# --------------------------------------------------------------------------------------
# forecasters
# --------------------------------------------------------------------------------------
def f_seasonal_naive(y, dates, h):
    """Repeat the last 7 days. A simple benchmark any real model should beat."""
    last = y[-7:]
    return np.array([last[i % 7] for i in range(h)], dtype=float)


def f_moving_average(y, dates, h, window=28):
    """Flat forecast = mean of the last ``window`` days."""
    return np.full(h, y[-window:].mean())


def f_ses(y, dates, h, alpha=0.1):
    """Simple exponential smoothing: flat forecast at the smoothed level."""
    return np.full(h, max(ses_level(y, alpha), 0.0))


def f_holt_damped(y, dates, h, alpha=0.1, beta=0.05, phi=0.98):
    """Holt's linear method with a damped trend (stops runaway extrapolation)."""
    x = np.asarray(y, dtype=float)[-365:]
    level, trend = x[:7].mean(), 0.0
    for v in x:
        prev = level
        level = alpha * v + (1 - alpha) * (prev + phi * trend)
        trend = beta * (level - prev) + (1 - beta) * phi * trend
    steps = np.cumsum(phi ** np.arange(1, h + 1))
    return np.maximum(level + trend * steps, 0.0)


def f_croston_sba(y, dates, h, alpha=0.1):
    """Croston's method with the Syntetos-Boylan bias correction, for intermittent demand.

    Smooths demand *size* and the *interval* between demands separately.
    """
    x = np.asarray(y, dtype=float)[-730:]
    nz = np.flatnonzero(x > 0)
    if len(nz) == 0:
        return np.zeros(h)
    sizes = x[nz]
    intervals = np.diff(np.concatenate([[-1], nz])).astype(float)
    z, p = ses_level(sizes, alpha), ses_level(intervals, alpha)
    return np.full(h, (1 - alpha / 2) * z / max(p, 1.0))


def f_seasonal_ses(y, dates, h, alpha=0.1):
    """Exponential smoothing on de-seasonalised demand, re-seasonalised for the forecast days.

    Removes the weekday pattern and the annual (monsoon / peak-season) pattern, smooths the
    remaining level, then multiplies the seasonal indices back in for each future day.
    """
    dow, month = seasonal_indices(y, dates)
    d_idx = dates.dayofweek.to_numpy()
    m_idx = dates.month.to_numpy() - 1
    factor = np.maximum(dow[d_idx] * month[m_idx], 1e-6)
    level = ses_level(np.asarray(y, dtype=float) / factor, alpha)

    future = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=h, freq="D")
    f_factor = dow[future.dayofweek.to_numpy()] * month[future.month.to_numpy() - 1]
    return np.maximum(level * f_factor, 0.0)


@dataclass
class MethodSpec:
    fn: callable
    grid: list[dict] = field(default_factory=lambda: [{}])  # candidate parameter sets, tuned per SKU


METHODS: dict[str, MethodSpec] = {
    "Seasonal Naive": MethodSpec(f_seasonal_naive),
    "Moving Avg (7d)": MethodSpec(f_moving_average, [{"window": 7}]),
    "Moving Avg (28d)": MethodSpec(f_moving_average, [{"window": 28}]),
    "Exp. Smoothing (SES)": MethodSpec(f_ses, [{"alpha": a} for a in (0.03, 0.06, 0.1, 0.2, 0.4)]),
    "Holt (damped trend)": MethodSpec(f_holt_damped, [{"alpha": a} for a in (0.03, 0.06, 0.1, 0.2)]),
    "Croston-SBA": MethodSpec(f_croston_sba, [{"alpha": a} for a in (0.05, 0.1, 0.2)]),
    "Seasonal SES (weekday+annual)": MethodSpec(f_seasonal_ses, [{"alpha": a} for a in (0.03, 0.06, 0.1, 0.2, 0.4)]),
}


# --------------------------------------------------------------------------------------
# accuracy metrics
# --------------------------------------------------------------------------------------
def wape(actual, forecast) -> float:
    """Weighted absolute percentage error = sum|A-F| / sum(A). Safe with zero-demand days (unlike MAPE)."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    denom = a.sum()
    return float(np.abs(a - f).sum() / denom) if denom > 0 else float("nan")


def bias(actual, forecast) -> float:
    """sum(F-A)/sum(A). Positive = over-forecasting (excess stock), negative = under-forecasting (stockouts)."""
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    denom = a.sum()
    return float((f - a).sum() / denom) if denom > 0 else float("nan")


def mae(actual, forecast) -> float:
    return float(np.mean(np.abs(np.asarray(actual, float) - np.asarray(forecast, float))))


def rmse(actual, forecast) -> float:
    return float(np.sqrt(np.mean((np.asarray(actual, float) - np.asarray(forecast, float)) ** 2)))
