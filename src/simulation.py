"""Replenishment policy simulation over the hold-out test window.

Three (s, Q) reorder-point policies are replayed day by day against the ACTUAL demand of the test window,
with randomly drawn supplier lead times (30 Monte-Carlo replications, same lead-time draws for every policy):

  A. Naive            ROP = lead-time demand from the trailing 30-day average, NO safety stock, Q = 30 days of demand
  B. Statistical SS   textbook: trailing 30-day mean/std, safety stock z*sqrt(L*sd_d^2 + d^2*sd_L^2), Q = EOQ
  C. Forecast-driven  weekly re-forecast from the auto-selected model, safety stock from forecast-error, Q = EOQ

Stock-outs are treated as lost sales. Total cost = holding + ordering + lost margin on unmet demand.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config
from .backtest import BacktestResult, lead_time_demand_cv
from .forecasting import METHODS
from .inventory import abc_classification, eoq, lead_time_stats, z_score
from .planning import parse_params

POLICIES = ["A. Naive (no safety stock)", "B. Statistical safety stock", "C. Forecast-driven (this project)"]


@dataclass
class SkuSim:
    sku: str
    y_test: np.ndarray
    rop: dict            # policy -> array(T) reorder point
    q: dict              # policy -> array(T) order quantity
    init_stock: float
    lt_hist: np.ndarray
    unit_cost: float
    unit_price: float
    order_cost: float
    hold_rate: float
    abc: str


def build_policies(wide, sales, master, lead_hist, bt: BacktestResult):
    """Compute, for each SKU, the reorder point / order quantity each policy would use on every test day."""
    dates, t0, n = wide.index, bt.test_start, bt.n_days
    T = n - t0
    train_lt = lead_time_stats(lead_hist, before=config.TEST_START)
    train_rev = sales[sales["date"] < config.TEST_START].groupby("sku_id").apply(
        lambda g: g[g["date"] >= (pd.Timestamp(config.TEST_START) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")]["revenue"].sum(),
        include_groups=False,
    )
    abc = abc_classification(train_rev)
    m = master.set_index("sku_id")
    chosen = bt.chosen.set_index("sku_id")

    sims = []
    for sku in wide.columns:
        y = wide[sku].to_numpy(dtype=float)
        L, sd_L = float(train_lt.loc[sku, "lt_mean"]), float(train_lt.loc[sku, "lt_sd"])
        Lc = int(np.ceil(L))
        z = z_score(config.SERVICE_LEVELS[abc[sku]])
        H = m.loc[sku, "unit_cost"] * m.loc[sku, "holding_cost_rate"]
        S = m.loc[sku, "ordering_cost"]

        method = chosen.loc[sku, "method"]
        params = parse_params(chosen.loc[sku, "params"])
        # error of the chosen model measured ONLY on the validation window (no test leakage)
        cv = lead_time_demand_cv(y, dates, METHODS[method].fn, params, bt.val_start, t0, Lc)
        fcs = bt.test_forecasts[sku][method]
        origins = np.array(sorted(fcs))

        csum = np.concatenate([[0.0], np.cumsum(y)])
        rop = {p: np.zeros(T) for p in POLICIES}
        q = {p: np.zeros(T) for p in POLICIES}
        for i in range(T):
            t = t0 + i  # decisions are taken at the end of day t
            d30 = (csum[t + 1] - csum[t - 29]) / 30.0
            sd30 = y[t - 29 : t + 1].std(ddof=1)

            rop[POLICIES[0]][i] = d30 * L
            q[POLICIES[0]][i] = max(1.0, 30 * d30)

            ss_b = z * np.sqrt(L * sd30**2 + (d30 * sd_L) ** 2)
            rop[POLICIES[1]][i] = d30 * L + ss_b
            q[POLICIES[1]][i] = max(1.0, eoq(d30 * 365, S, H))

            o = origins[origins <= t].max()
            f, off = fcs[o], t + 1 - o
            ltd = f[off : off + Lc].sum() * L / Lc
            d_f = ltd / L
            ss_c = z * np.sqrt((cv * ltd) ** 2 + (d_f * sd_L) ** 2)
            rop[POLICIES[2]][i] = ltd + ss_c
            q[POLICIES[2]][i] = max(1.0, eoq(f[off : off + 14].mean() * 365, S, H))

        lt_h = lead_hist[lead_hist["sku_id"] == sku]["lead_time_days"].to_numpy()
        sims.append(
            SkuSim(
                sku, y[t0:], rop, q, float(csum[t0 + 1] - csum[t0 - 29]) / 30.0 * 20.0, lt_h,
                float(m.loc[sku, "unit_cost"]), float(m.loc[sku, "unit_price"]), float(S),
                float(m.loc[sku, "holding_cost_rate"]), abc[sku],
            )
        )
    return sims


def run_policy(sim: SkuSim, policy: str, lead_times: np.ndarray, return_path: bool = False):
    """Replay one policy for one SKU. Returns a dict of KPIs (and the daily on-hand path if requested)."""
    y, rop, q = sim.y_test, sim.rop[policy], sim.q[policy]
    T = len(y)
    on_hand, on_order = sim.init_stock, 0.0
    arrivals: dict[int, float] = {}
    n_orders = 0
    sold = lost = 0.0
    stockout_days = 0
    inv_sum = 0.0
    path = np.zeros(T)
    for t in range(T):
        got = arrivals.pop(t, 0.0)
        on_hand += got
        on_order -= got
        s = min(on_hand, y[t])
        on_hand -= s
        sold += s
        if y[t] > s:
            lost += y[t] - s
            stockout_days += 1
        inv_sum += on_hand
        path[t] = on_hand
        guard = 0
        while on_hand + on_order <= rop[t] and guard < 5:  # order until position is above the reorder point
            qty = np.ceil(q[t])
            lt = int(lead_times[n_orders % len(lead_times)])
            arrivals[t + lt] = arrivals.get(t + lt, 0.0) + qty
            on_order += qty
            n_orders += 1
            guard += 1

    demand = y.sum()
    avg_inv = inv_sum / T
    holding = avg_inv * sim.unit_cost * sim.hold_rate * T / 365.0
    ordering = n_orders * sim.order_cost
    stockout_cost = lost * (sim.unit_price - sim.unit_cost)
    out = dict(
        demand=demand, sold=sold, lost=lost, fill_rate=sold / demand if demand > 0 else 1.0,
        stockout_days=stockout_days, avg_inventory_units=avg_inv, avg_inventory_value=avg_inv * sim.unit_cost,
        orders=n_orders, holding_cost=holding, ordering_cost=ordering, stockout_cost=stockout_cost,
        total_cost=holding + ordering + stockout_cost,
    )
    return (out, path) if return_path else out


def simulate_all(sims: list[SkuSim], reps: int = config.SIM_REPLICATIONS, seed: int = config.SEED) -> pd.DataFrame:
    """Monte-Carlo over lead times; every policy sees identical lead-time draws (common random numbers)."""
    rng = np.random.default_rng(seed)
    rows = []
    for sim in sims:
        acc = {p: [] for p in POLICIES}
        for _ in range(reps):
            lead_times = rng.choice(sim.lt_hist, size=60)
            for p in POLICIES:
                acc[p].append(run_policy(sim, p, lead_times))
        for p in POLICIES:
            mean = pd.DataFrame(acc[p]).mean().to_dict()
            rows.append(dict(sku_id=sim.sku, abc_class=sim.abc, policy=p, **mean))
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Portfolio-level KPIs per policy (units-weighted fill rate, summed costs)."""
    g = results.groupby("policy")
    s = pd.DataFrame(
        {
            "fill_rate": g["sold"].sum() / g["demand"].sum(),
            "avg_stockout_days_per_sku": g["stockout_days"].mean(),
            "avg_inventory_value_inr": g["avg_inventory_value"].sum(),
            "orders_per_sku": g["orders"].mean(),
            "holding_cost_inr": g["holding_cost"].sum(),
            "ordering_cost_inr": g["ordering_cost"].sum(),
            "stockout_cost_inr": g["stockout_cost"].sum(),
            "total_cost_inr": g["total_cost"].sum(),
        }
    )
    return s.loc[POLICIES]
