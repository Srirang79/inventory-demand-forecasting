"""Inventory policy maths: EOQ, safety stock, reorder point, ABC classes and stockout risk."""
from __future__ import annotations

from math import sqrt
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import config

_NORMAL = NormalDist()


def z_score(service_level: float) -> float:
    """Standard-normal z for a target cycle service level (e.g. 0.95 -> 1.645)."""
    return _NORMAL.inv_cdf(service_level)


def eoq(annual_demand: float, order_cost: float, holding_cost_per_unit: float) -> float:
    """Economic Order Quantity = sqrt(2 * D * S / H). Minimises ordering + holding cost."""
    if annual_demand <= 0 or holding_cost_per_unit <= 0:
        return 0.0
    return sqrt(2.0 * annual_demand * order_cost / holding_cost_per_unit)


def total_annual_cost(q, annual_demand, order_cost, holding_cost_per_unit):
    """Ordering cost + average cycle-stock holding cost for order size ``q``."""
    q = np.asarray(q, dtype=float)
    return annual_demand / q * order_cost + q / 2.0 * holding_cost_per_unit


def safety_stock(z: float, sigma_ltd: float, daily_demand: float, sigma_lead_time: float) -> float:
    """Safety stock covering BOTH demand-forecast error and supplier lead-time variability.

    SS = z * sqrt( sigma_LTD^2 + (d * sigma_L)^2 )

    sigma_LTD : std-dev of lead-time-demand forecast error (units)
    d         : average daily demand,  sigma_L : std-dev of lead time (days)
    """
    return z * sqrt(sigma_ltd**2 + (daily_demand * sigma_lead_time) ** 2)


def reorder_point(daily_demand: float, lead_time: float, ss: float) -> float:
    """ROP = expected demand during lead time + safety stock."""
    return daily_demand * lead_time + ss


def stockout_probability(inventory_position: float, mean_ltd: float, sd_ltd: float) -> float:
    """P(demand during lead time > stock available), assuming normally distributed lead-time demand."""
    if sd_ltd <= 0:
        return float(inventory_position < mean_ltd)
    return float(1.0 - _NORMAL.cdf((inventory_position - mean_ltd) / sd_ltd))


def abc_classification(revenue_by_sku: pd.Series, cutoffs=config.ABC_CUTOFFS) -> pd.Series:
    """Classify SKUs by cumulative revenue share: A = top ~70%, B = next ~20%, C = the rest."""
    ranked = revenue_by_sku.sort_values(ascending=False)
    share_before = ranked.cumsum() / ranked.sum() - ranked / ranked.sum()
    cls = np.where(share_before < cutoffs[0], "A", np.where(share_before < cutoffs[1], "B", "C"))
    return pd.Series(cls, index=ranked.index).reindex(revenue_by_sku.index)


def lead_time_stats(lead_hist: pd.DataFrame, before: str) -> pd.DataFrame:
    """Mean and std-dev of realised lead times per SKU, using only purchase orders placed before ``before``."""
    h = lead_hist[lead_hist["order_date"] < before]
    out = h.groupby("sku_id")["lead_time_days"].agg(lt_mean="mean", lt_sd="std")
    out["lt_sd"] = out["lt_sd"].fillna(0.0)
    return out
