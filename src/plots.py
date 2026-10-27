"""Matplotlib figures saved to outputs/figures/."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config
from .backtest import BacktestResult
from .forecasting import METHODS
from .inventory import eoq, total_annual_cost
from .planning import parse_params
from .simulation import POLICIES, SkuSim, run_policy

C_NAIVE, C_STAT, C_FCST = "#8d949e", "#3b6ea8", "#e8710a"
C_ACTUAL, C_TRAIN = "#1f2933", "#b8c0cc"
POLICY_COLORS = dict(zip(POLICIES, [C_NAIVE, C_STAT, C_FCST]))
SHORT = {POLICIES[0]: "A. Naive", POLICIES[1]: "B. Statistical SS", POLICIES[2]: "C. Forecast-driven"}

plt.rcParams.update(
    {
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
        "axes.titleweight": "bold", "axes.titlesize": 11,
    }
)


def _save(fig, name):
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = config.FIG_DIR / name
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_demand_overview(sales: pd.DataFrame, master: pd.DataFrame):
    s = sales.merge(master[["sku_id", "category"]], on="sku_id")
    s["date"] = pd.to_datetime(s["date"])
    monthly = s.groupby(s["date"].dt.to_period("M"))["revenue"].sum() / 1e7  # INR crore
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    ax[0].plot(monthly.index.to_timestamp(), monthly.values, color=C_STAT, lw=2)
    ax[0].set_title("Monthly revenue across all 60 SKUs")
    ax[0].set_ylabel("INR crore")
    for yr in range(2022, 2026):
        ax[0].axvspan(pd.Timestamp(f"{yr}-06-01"), pd.Timestamp(f"{yr}-09-30"), color="#cfd8e3", alpha=0.45, lw=0)
    ax[0].text(0.01, 0.04, "shaded = monsoon months", transform=ax[0].transAxes, fontsize=8, color="#52606d")

    cat = s.groupby(["category", s["date"].dt.month])["units_sold"].sum().unstack(0)
    idx = cat / cat.mean()
    for c, col in [("Aggregates", C_FCST), ("Electrical", C_STAT), ("Paints & Waterproofing", "#2a9d8f")]:
        ax[1].plot(idx.index, idx[c], marker="o", ms=3, lw=1.8, label=c, color=col)
    ax[1].axhline(1, color=C_NAIVE, lw=1, ls="--")
    ax[1].set_xticks(range(1, 13), list("JFMAMJJASOND"))
    ax[1].set_title("Seasonality differs by category")
    ax[1].set_ylabel("Month index (1.0 = average)")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return _save(fig, "01_demand_overview.png")


def fig_forecast_vs_actual(wide: pd.DataFrame, bt: BacktestResult, master: pd.DataFrame, skus: list[str]):
    names = master.set_index("sku_id")["sku_name"]
    chosen = bt.chosen.set_index("sku_id")
    t0, n = bt.test_start, bt.n_days
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.4))
    for ax, sku in zip(axes.ravel(), skus):
        y = wide[sku].to_numpy()
        method = chosen.loc[sku, "method"]
        fcs = bt.test_forecasts[sku][method]
        stitched = np.full(n, np.nan)
        for o, f in fcs.items():
            k = min(config.ORIGIN_STEP, n - o)
            stitched[o : o + k] = f[:k]
        lo = t0 - 90
        d = bt.dates
        ax.plot(d[lo:t0], y[lo:t0], color=C_TRAIN, lw=1, label="History")
        ax.plot(d[t0:], y[t0:], color=C_ACTUAL, lw=1, label="Actual (test)")
        ax.plot(d[t0:], stitched[t0:], color=C_FCST, lw=2, label="Forecast")
        w = chosen.loc[sku, "test_wape"]
        ax.set_title(f"{names[sku]}\n{method} | test WAPE {w:.0%}", fontsize=9)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Weekly-refreshed forecasts vs actual daily demand on the hold-out quarter (Oct-Dec 2025)", y=1.02, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "02_forecast_vs_actual.png")


def fig_method_comparison(bt: BacktestResult):
    m = bt.metrics.groupby("method")["test_wape"].mean().sort_values()
    auto = bt.chosen["test_wape"].mean()
    labels = list(m.index) + ["Auto-select per SKU"]
    vals = list(m.values) + [auto]
    colors = [C_NAIVE] * len(m) + [C_FCST]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.barh(labels, vals, color=colors)
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{v:.1%}" for v in vals], padding=3, fontsize=9)
    ax.set_xlabel("Mean WAPE on hold-out test (lower is better)")
    ax.set_title("Forecast accuracy by method (60 SKUs, 14-day horizon, weekly origins)")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(axis="y", visible=False)
    return _save(fig, "03_method_comparison.png")


def fig_inventory_trajectory(sim: SkuSim, name: str, seed: int = config.SEED):
    rng = np.random.default_rng(seed)
    lead_times = rng.choice(sim.lt_hist, size=60)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True, sharey=True)
    x = np.arange(len(sim.y_test))
    for ax, p in zip(axes, POLICIES):
        kpi, path = run_policy(sim, p, lead_times, return_path=True)
        ax.fill_between(x, path, color=POLICY_COLORS[p], alpha=0.35, lw=0)
        ax.plot(x, path, color=POLICY_COLORS[p], lw=1.6)
        ax.plot(x, sim.rop[p], color=C_ACTUAL, lw=1, ls="--", label="Reorder point")
        out = np.flatnonzero(path <= 0)
        ax.scatter(out, np.zeros(len(out)), marker="v", color="#c0392b", s=28, zorder=5, label="Stocked out", clip_on=False)
        ax.set_title(f"{SHORT[p]}: fill rate {kpi['fill_rate']:.1%}, stockout days: {kpi['stockout_days']}, orders: {kpi['orders']}", fontsize=10)
        ax.set_ylabel("Units on hand")
        ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Day of test quarter (Oct 1 = 0)")
    fig.suptitle(f"Same demand, same supplier delays, three policies - {name}", fontweight="bold", y=1.0)
    fig.tight_layout()
    return _save(fig, "04_inventory_simulation.png")


def fig_policy_comparison(summary: pd.DataFrame):
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    lab = [SHORT[p] for p in summary.index]
    cols = [POLICY_COLORS[p] for p in summary.index]
    ax[0].bar(lab, summary["fill_rate"] * 100, color=cols)
    ax[0].set_ylim(95, 100)
    ax[0].set_title("Fill rate (%)")
    for i, v in enumerate(summary["fill_rate"] * 100):
        ax[0].text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)
    ax[1].bar(lab, summary["avg_stockout_days_per_sku"], color=cols)
    ax[1].set_title("Stockout days per SKU (92-day quarter)")
    for i, v in enumerate(summary["avg_stockout_days_per_sku"]):
        ax[1].text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)
    bottom = np.zeros(len(summary))
    for col, label, c in [("holding_cost_inr", "Holding", "#9db4d0"), ("ordering_cost_inr", "Ordering", "#5b7fa8"), ("stockout_cost_inr", "Lost margin", "#c0392b")]:
        v = summary[col].to_numpy() / 1e5
        ax[2].bar(lab, v, bottom=bottom, color=c, label=label)
        bottom += v
    for i, v in enumerate(bottom):
        ax[2].text(i, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
    ax[2].set_title("Total cost (INR lakh)")
    ax[2].legend(frameon=False, fontsize=8)
    for a in ax:
        a.tick_params(axis="x", labelsize=8)
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, "05_policy_comparison.png")


def fig_eoq_curve(row: pd.Series, master: pd.DataFrame):
    m = master.set_index("sku_id").loc[row["sku_id"]]
    H = m["unit_cost"] * m["holding_cost_rate"]
    D = row["forecast_daily_demand"] * 365
    S = m["ordering_cost"]
    q_star = eoq(D, S, H)
    q = np.linspace(q_star * 0.15, q_star * 3, 300)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(q, D / q * S, color=C_STAT, lw=1.6, label="Ordering cost (falls as Q grows)")
    ax.plot(q, q / 2 * H, color=C_NAIVE, lw=1.6, label="Holding cost (rises as Q grows)")
    ax.plot(q, total_annual_cost(q, D, S, H), color=C_FCST, lw=2.4, label="Total annual cost")
    ax.axvline(q_star, color=C_ACTUAL, ls="--", lw=1)
    ax.annotate(f"EOQ = {q_star:,.0f} units", (q_star, total_annual_cost(q_star, D, S, H)), xytext=(10, 28), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Order quantity (units)")
    ax.set_ylabel("INR per year")
    ax.set_title(f"EOQ trade-off - {m['sku_name']}")
    ax.legend(frameon=False, fontsize=8)
    return _save(fig, "06_eoq_cost_curve.png")


def fig_stockout_risk(plan: pd.DataFrame, top: int = 15):
    d = plan.head(top).iloc[::-1]
    colors = d["stockout_risk"].map(lambda r: "#c0392b" if r > 0.4 else (C_FCST if r > 0.1 else C_STAT))
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(d["sku_name"] + "  [" + d["abc_class"] + "]", d["stockout_risk"] * 100, color=colors)
    ax.set_xlabel("Probability of stockout before next delivery arrives (%)")
    ax.set_title(f"Highest stockout risk right now - top {top} SKUs (as of {config.DATA_END})")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 100)
    return _save(fig, "07_stockout_risk.png")
