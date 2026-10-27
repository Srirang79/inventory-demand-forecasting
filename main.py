"""Run the full pipeline:  data -> forecast backtest -> reorder plan -> policy simulation -> charts.

Usage:
    python main.py                 # use data/*.csv (generated automatically if missing)
    python main.py --regenerate    # rebuild the synthetic dataset first
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from src import backtest, config, data_generation, plots, planning, simulation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--regenerate", action="store_true", help="re-create the synthetic dataset in data/")
    ap.add_argument("--seed", type=int, default=config.SEED, help="seed for dataset generation")
    args = ap.parse_args()
    t_start = time.time()
    config.TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # 1) data ------------------------------------------------------------------------------------
    if args.regenerate or not (config.DATA_DIR / "sales_history.csv").exists():
        print("[1/5] Generating synthetic dataset ...")
        data_generation.generate(seed=args.seed)
    else:
        print("[1/5] Loading dataset from data/ ...")
    read = lambda name: pd.read_csv(config.DATA_DIR / f"{name}.csv")  # noqa: E731
    sales, master = read("sales_history"), read("sku_master")
    lead_hist, snapshot = read("lead_time_history"), read("inventory_snapshot")
    wide = backtest.pivot_sales(sales)
    print(f"      {len(sales):,} daily rows | {wide.shape[1]} SKUs | {wide.index[0].date()} to {wide.index[-1].date()}")

    # 2) forecasting backtest ----------------------------------------------------------------------
    print("[2/5] Forecast backtest (tune on validation, score on hold-out test) ...")
    bt = backtest.run_backtest(wide)
    bt.metrics.to_csv(config.TABLE_DIR / "forecast_accuracy_by_sku_and_method.csv", index=False)
    by_method = (
        bt.metrics.groupby("method")[["val_wape", "test_wape", "test_mae", "test_rmse", "test_bias"]].mean()
        .sort_values("test_wape")
    )
    by_method.loc["Auto-select per SKU"] = bt.chosen[["val_wape", "test_wape", "test_mae", "test_rmse", "test_bias"]].mean()
    by_method.round(4).to_csv(config.TABLE_DIR / "forecast_accuracy_summary.csv")
    print(by_method.round(3).to_string())

    # 3) reorder plan ---------------------------------------------------------------------------------
    print("[3/5] Building reorder plan (EOQ, safety stock, reorder point, stockout risk) ...")
    plan = planning.build_reorder_plan(wide, sales, master, lead_hist, snapshot, bt)
    plan.round(3).to_csv(config.TABLE_DIR / "reorder_plan.csv", index=False)
    print(f"      {int(plan['reorder_now'].sum())} SKUs at/below reorder point | "
          f"{int((plan['risk_band'] == 'High').sum())} high stockout risk")

    # 4) policy simulation ------------------------------------------------------------------------------
    print(f"[4/5] Simulating 3 replenishment policies ({config.SIM_REPLICATIONS} Monte-Carlo runs) ...")
    sims = simulation.build_policies(wide, sales, master, lead_hist, bt)
    results = simulation.simulate_all(sims)
    summary = simulation.summarise(results)
    results.round(3).to_csv(config.TABLE_DIR / "simulation_results_by_sku.csv", index=False)
    summary.round(4).to_csv(config.TABLE_DIR / "simulation_policy_summary.csv")
    print(summary.round(3).T.to_string())

    # 5) charts -----------------------------------------------------------------------------------------
    print("[5/5] Saving charts ...")
    top_a = plan[plan["abc_class"] == "A"].sort_values("forecast_daily_demand", ascending=False)
    sample = list(top_a["sku_id"].head(2)) + list(plan[plan["abc_class"] == "C"]["sku_id"].head(1))
    slow = master.loc[master["sku_name"].str.contains("Water Tank|Ready-Mix|Marble"), "sku_id"].tolist()
    sample += [s for s in slow if s not in sample][:1]
    plots.fig_demand_overview(sales, master)
    plots.fig_forecast_vs_actual(wide, bt, master, sample[:4])
    plots.fig_method_comparison(bt)
    star = next(s for s in sims if s.sku == top_a["sku_id"].iloc[0])
    plots.fig_inventory_trajectory(star, master.set_index("sku_id").loc[star.sku, "sku_name"])
    plots.fig_policy_comparison(summary)
    plots.fig_eoq_curve(top_a.iloc[0], master)
    plots.fig_stockout_risk(plan)
    print(f"Done in {time.time() - t_start:.0f}s. Tables -> outputs/tables, charts -> outputs/figures")


if __name__ == "__main__":
    main()
