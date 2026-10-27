"""Forward-looking replenishment plan: EOQ, safety stock, reorder point and stockout risk per SKU."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .backtest import BacktestResult, lead_time_demand_cv
from .forecasting import METHODS
from .inventory import (
    abc_classification, eoq, lead_time_stats, reorder_point, safety_stock, stockout_probability, z_score,
)


def parse_params(s: str) -> dict:
    return {} if s in ("-", "", None) else eval(s, {"__builtins__": {}})  # noqa: S307 - own repr of a small dict


def build_reorder_plan(
    wide: pd.DataFrame,
    sales: pd.DataFrame,
    master: pd.DataFrame,
    lead_hist: pd.DataFrame,
    snapshot: pd.DataFrame,
    bt: BacktestResult,
) -> pd.DataFrame:
    """Plan as of the last day of data, using each SKU's auto-selected forecasting method.

    Lead-time-demand error is measured on the most recent window (the test window) so the safety stock reflects
    current forecast quality. Stock on hand comes from the inventory snapshot.
    """
    dates = wide.index
    n = len(dates)
    lt = lead_time_stats(lead_hist, before="9999")  # all history up to the plan date
    last_year = sales[sales["date"] >= (dates[-1] - pd.Timedelta(days=364)).strftime("%Y-%m-%d")]
    revenue = last_year.groupby("sku_id")["revenue"].sum()
    abc = abc_classification(revenue)

    m = master.set_index("sku_id")
    snap = snapshot.set_index("sku_id")
    chosen = bt.chosen.set_index("sku_id")

    rows = []
    for sku in wide.columns:
        y = wide[sku].to_numpy(dtype=float)
        method = chosen.loc[sku, "method"]
        params = parse_params(chosen.loc[sku, "params"])
        fn = METHODS[method].fn

        fc = fn(y, dates, config.PLAN_HORIZON, **params)
        d = float(fc.mean())  # forecast average daily demand over the next PLAN_HORIZON days
        L, sd_L = float(lt.loc[sku, "lt_mean"]), float(lt.loc[sku, "lt_sd"])
        Lc = int(np.ceil(L))
        ltd = float(fc[:Lc].sum() * L / Lc)

        cv = lead_time_demand_cv(y, dates, fn, params, bt.test_start, n, Lc)
        z = z_score(config.SERVICE_LEVELS[abc[sku]])
        sigma_ltd = cv * ltd
        ss = safety_stock(z, sigma_ltd, d, sd_L)
        rop = reorder_point(ltd / L, L, ss)

        H = m.loc[sku, "unit_cost"] * m.loc[sku, "holding_cost_rate"]
        q = eoq(d * 365, m.loc[sku, "ordering_cost"], H)

        on_hand, on_order = float(snap.loc[sku, "on_hand"]), float(snap.loc[sku, "on_order"])
        position = on_hand + on_order
        sd_total = float(np.sqrt(sigma_ltd**2 + (d * sd_L) ** 2))
        risk = stockout_probability(position, ltd, sd_total)

        rows.append(
            dict(
                sku_id=sku, sku_name=m.loc[sku, "sku_name"], category=m.loc[sku, "category"], abc_class=abc[sku],
                method=method, forecast_daily_demand=d, lead_time_mean_days=L, lead_time_sd_days=sd_L,
                service_level=config.SERVICE_LEVELS[abc[sku]], lead_time_demand=ltd, safety_stock=ss,
                reorder_point=rop, eoq=q, on_hand=on_hand, on_order=on_order, inventory_position=position,
                days_of_cover=on_hand / d if d > 0 else np.inf, stockout_risk=risk,
                reorder_now=bool(position <= rop), suggested_order_qty=float(np.ceil(q)) if position <= rop else 0.0,
            )
        )
    plan = pd.DataFrame(rows)
    plan["risk_band"] = pd.cut(plan["stockout_risk"], [-0.01, 0.10, 0.40, 1.01], labels=["Low", "Medium", "High"])
    return plan.sort_values(["stockout_risk", "abc_class"], ascending=[False, True]).reset_index(drop=True)
