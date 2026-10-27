"""Rolling-origin backtest: tune on validation, select a method per SKU, score on the hold-out test.

Protocol (all time-ordered, no leakage):
  1. For every SKU x method, pick the smoothing parameter with the lowest WAPE on the validation window.
  2. Pick the best method per SKU from those validation scores ("auto-select").
  3. Score every method (and the auto-selected one) on the untouched test window.

Forecasts are re-issued every ORIGIN_STEP days (weekly replanning) using only data available at that date.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .forecasting import METHODS, bias, mae, rmse, wape


@dataclass
class BacktestResult:
    metrics: pd.DataFrame            # one row per SKU x method (validation + test scores)
    chosen: pd.DataFrame             # one row per SKU: auto-selected method + params + test scores
    test_forecasts: dict             # {sku: {method: {origin_idx: np.ndarray(FORECAST_HORIZON)}}}
    dates: pd.DatetimeIndex
    val_start: int                   # integer positions of the windows on the daily axis
    test_start: int
    n_days: int


def pivot_sales(sales: pd.DataFrame) -> pd.DataFrame:
    """Long sales table -> wide matrix (date x sku) of daily units sold."""
    wide = sales.pivot(index="date", columns="sku_id", values="units_sold").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide


def forecast_origins(y, dates, fn, params, start, end, h=config.FORECAST_HORIZON, step=config.ORIGIN_STEP):
    """Issue a forecast of length ``h`` at each origin in [start, end) using only y[:origin]."""
    return {o: fn(y[:o], dates[:o], h, **params) for o in range(start, end, step)}


def score_origins(y, fcs: dict, end: int, horizon=config.EVAL_HORIZON):
    """Pool (actual, forecast) pairs from every origin, clipped to ``horizon`` days and to the window end."""
    act, fc = [], []
    for o, f in fcs.items():
        k = min(horizon, end - o)
        if k > 0:
            act.append(y[o : o + k])
            fc.append(f[:k])
    return np.concatenate(act), np.concatenate(fc)


def lead_time_demand_cv(y, dates, fn, params, start, end, lead_days: int) -> float:
    """Relative error of lead-time demand: RMSE(sum of actuals - sum of forecasts over L days) / mean(sum of actuals).

    Summing forecast errors over the lead time (instead of assuming independent daily errors) captures the fact
    that forecast errors are positively correlated from day to day. Evaluated at EVERY day in [start, end - L].
    """
    errs, tot = [], []
    for o in range(start, end - lead_days + 1):
        f = fn(y[:o], dates[:o], lead_days, **params)
        a = y[o : o + lead_days].sum()
        errs.append(a - f.sum())
        tot.append(a)
    mean_ltd = np.mean(tot)
    return float(np.sqrt(np.mean(np.square(errs))) / mean_ltd) if mean_ltd > 0 else 0.0


def run_backtest(wide: pd.DataFrame, verbose: bool = True) -> BacktestResult:
    dates = wide.index
    n = len(dates)
    val_start = int(dates.get_loc(pd.Timestamp(config.VAL_START)))
    test_start = int(dates.get_loc(pd.Timestamp(config.TEST_START)))

    rows, chosen_rows, test_fc = [], [], {}
    for i, sku in enumerate(wide.columns, 1):
        y = wide[sku].to_numpy(dtype=float)
        test_fc[sku] = {}
        sku_rows = []
        for name, spec in METHODS.items():
            # 1) tune parameter on the validation window
            best_w, best_params = np.inf, spec.grid[0]
            for params in spec.grid:
                fcs = forecast_origins(y, dates, spec.fn, params, val_start, test_start)
                a, f = score_origins(y, fcs, end=test_start)
                w = wape(a, f)
                if np.isfinite(w) and w < best_w:
                    best_w, best_params = w, params
            # 2) score on the untouched test window with the tuned parameter
            fcs = forecast_origins(y, dates, spec.fn, best_params, test_start, n)
            test_fc[sku][name] = fcs
            a, f = score_origins(y, fcs, end=n)
            sku_rows.append(
                dict(
                    sku_id=sku, method=name, params=str(best_params) if best_params else "-",
                    val_wape=best_w, test_wape=wape(a, f), test_mae=mae(a, f),
                    test_rmse=rmse(a, f), test_bias=bias(a, f),
                )
            )
        sku_df = pd.DataFrame(sku_rows)
        best_name = sku_df.loc[sku_df["val_wape"].idxmin(), "method"]
        sku_df["selected"] = sku_df["method"] == best_name
        rows.append(sku_df)
        chosen_rows.append(sku_df[sku_df["selected"]].iloc[0])
        if verbose and (i % 10 == 0 or i == len(wide.columns)):
            print(f"  backtest: {i}/{len(wide.columns)} SKUs")

    metrics = pd.concat(rows, ignore_index=True)
    chosen = pd.DataFrame(chosen_rows).reset_index(drop=True)
    return BacktestResult(metrics, chosen, test_fc, dates, val_start, test_start, n)
