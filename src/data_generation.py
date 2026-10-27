"""Synthetic SKU-level dataset for a construction-materials distributor.

The data is SIMULATED (no real company data is used). It is built to contain the
patterns a real replenishment team has to deal with:

* a stable underlying level with year-on-year growth (trend)
* strong annual seasonality (monsoon slowdown Jun-Sep, peaks Oct-Mar)
* day-of-week effects (Sundays are slow)
* promotions that lift demand for a few days
* over-dispersed noise (variance larger than a Poisson model would give)
* occasional bulk project orders (outlier spikes)
* intermittent demand for slow-moving, high-ticket items
* supplier lead times that vary and get worse in the monsoon

Output files (written to ``data/``):
    sku_master.csv          one row per SKU: category, cost, price, ordering & holding cost
    sales_history.csv       daily units sold per SKU (long format)
    lead_time_history.csv   realised supplier lead times for past purchase orders
    inventory_snapshot.csv  stock position on the last day of the data
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# (name, unit_cost in INR, demand profile)  profile: fast | std | slow
CATALOG: dict[str, list[tuple[str, float, str]]] = {
    "Cement & Mortar": [
        ("OPC 53 Grade Cement 50kg", 385, "fast"),
        ("PPC Cement 50kg", 365, "fast"),
        ("Premium PPC Cement 50kg", 410, "std"),
        ("White Cement 5kg", 165, "std"),
        ("Tile Adhesive 20kg", 540, "std"),
        ("Block Jointing Mortar 25kg", 310, "std"),
    ],
    "Steel & Rebar": [
        ("TMT Fe500D 8mm 12m bar", 365, "fast"),
        ("TMT Fe500D 10mm 12m bar", 570, "fast"),
        ("TMT Fe500D 12mm 12m bar", 820, "fast"),
        ("TMT Fe500D 16mm 12m bar", 1460, "std"),
        ("TMT Fe500D 20mm 12m bar", 2280, "std"),
        ("Binding Wire 1kg coil", 78, "std"),
        ("MS Angle 40x40mm 6m", 1180, "slow"),
        ("Welded Steel Mesh Sheet", 2650, "slow"),
    ],
    "Bricks & Blocks": [
        ("Red Clay Brick", 8, "fast"),
        ("Fly Ash Brick", 6, "fast"),
        ("AAC Block 600x200x100mm", 58, "fast"),
        ("AAC Block 600x200x200mm", 104, "std"),
        ("Concrete Hollow Block 8in", 42, "std"),
        ("Interlocking Paver Block", 34, "std"),
    ],
    "Aggregates": [
        ("M-Sand per tonne", 1150, "fast"),
        ("20mm Aggregate per tonne", 980, "fast"),
        ("40mm Aggregate per tonne", 940, "std"),
        ("GSB Mix per tonne", 720, "std"),
        ("Ready-Mix Concrete M25 per m3", 5600, "slow"),
    ],
    "Tiles & Flooring": [
        ("Vitrified Tile 600x600 box", 820, "std"),
        ("Vitrified Tile 800x800 box", 1650, "std"),
        ("Ceramic Wall Tile 300x450 box", 610, "std"),
        ("Anti-Skid Floor Tile 300x300 box", 540, "std"),
        ("Granite Slab per sqft", 165, "slow"),
        ("Marble Slab per sqft", 240, "slow"),
        ("Tile Spacers Pack", 45, "fast"),
    ],
    "Plumbing": [
        ("CPVC Pipe 1in 3m", 440, "std"),
        ("UPVC Pipe 4in 3m", 760, "std"),
        ("PVC Elbow 4in", 58, "fast"),
        ("Brass Ball Valve 1in", 390, "std"),
        ("Overhead Water Tank 1000L", 6900, "slow"),
        ("CP Basin Faucet", 720, "std"),
        ("Concealed Flush Tank", 3100, "slow"),
        ("GI Fitting Set", 260, "std"),
    ],
    "Electrical": [
        ("FR Wire 1.5 sqmm 90m", 1450, "fast"),
        ("FR Wire 2.5 sqmm 90m", 2350, "fast"),
        ("FR Wire 4 sqmm 90m", 3650, "std"),
        ("MCB 32A Single Pole", 290, "std"),
        ("Modular Switch 6A", 72, "fast"),
        ("LED Panel Light 18W", 540, "std"),
        ("PVC Conduit Pipe 25mm 3m", 68, "fast"),
        ("Distribution Board 8-Way", 2100, "slow"),
    ],
    "Paints & Waterproofing": [
        ("Exterior Emulsion 20L", 5200, "std"),
        ("Interior Emulsion 20L", 4300, "std"),
        ("Wall Primer 20L", 2350, "std"),
        ("Wall Putty 40kg", 1180, "fast"),
        ("Synthetic Enamel 1L", 430, "std"),
        ("Waterproofing Compound 20L", 3900, "std"),
        ("PU Sealant Cartridge", 310, "std"),
    ],
    "Hardware & Fasteners": [
        ("Anchor Bolt Pack", 190, "std"),
        ("Wood Screw Box 100pc", 150, "std"),
        ("Door Lock Set", 1350, "slow"),
        ("Nails 1kg", 92, "fast"),
        ("Door Hinge Pair", 120, "std"),
    ],
}

# lt: mean supplier lead time (days) | S: cost per purchase order (INR) | h: annual holding-cost rate
# margin: selling margin over cost | seas: seasonality strength | scale: volume multiplier
CATEGORY_PARAMS: dict[str, dict[str, float]] = {
    "Cement & Mortar":        dict(lt=3,  S=1500, h=0.22, margin=0.14, seas=0.9, scale=1.0),
    "Steel & Rebar":          dict(lt=6,  S=3000, h=0.15, margin=0.12, seas=0.7, scale=1.0),
    "Bricks & Blocks":        dict(lt=4,  S=2000, h=0.18, margin=0.20, seas=1.2, scale=4.0),
    "Aggregates":             dict(lt=2,  S=1800, h=0.14, margin=0.22, seas=1.4, scale=1.5),
    "Tiles & Flooring":       dict(lt=9,  S=2200, h=0.20, margin=0.25, seas=0.8, scale=1.0),
    "Plumbing":               dict(lt=7,  S=1200, h=0.22, margin=0.24, seas=0.8, scale=1.0),
    "Electrical":             dict(lt=8,  S=1000, h=0.22, margin=0.22, seas=0.6, scale=1.0),
    "Paints & Waterproofing": dict(lt=6,  S=1100, h=0.25, margin=0.28, seas=1.0, scale=0.6),
    "Hardware & Fasteners":   dict(lt=10, S=700,  h=0.20, margin=0.26, seas=0.5, scale=1.0),
}

BASE_DEMAND_RANGE = {"fast": (40, 120), "std": (8, 35), "slow": (0.6, 2.5)}

# Construction activity by calendar month (Jan..Dec): monsoon dip, festive/fiscal-year-end peaks
MONTH_PROFILE = np.array([1.10, 1.12, 1.18, 1.05, 0.98, 0.80, 0.72, 0.74, 0.85, 1.05, 1.12, 1.10])
# Mon..Sun: Sundays are slow
DOW_PROFILE = np.array([1.10, 1.08, 1.05, 1.05, 1.00, 1.05, 0.57])
DOW_PROFILE = DOW_PROFILE / DOW_PROFILE.mean()


def _build_master(rng: np.random.Generator):
    rows, hidden = [], []
    n = 0
    for category, items in CATALOG.items():
        p = CATEGORY_PARAMS[category]
        for name, cost, profile in items:
            n += 1
            lo, hi = BASE_DEMAND_RANGE[profile]
            base = rng.uniform(lo, hi) * (1.0 if profile == "slow" else p["scale"])
            lt_mean = max(1.0, p["lt"] * rng.uniform(0.85, 1.15))
            rows.append(
                dict(
                    sku_id=f"SKU{n:03d}",
                    sku_name=name,
                    category=category,
                    unit_cost=float(cost),
                    unit_price=round(cost * (1 + p["margin"] * rng.uniform(0.9, 1.1)), 2),
                    ordering_cost=round(p["S"] * rng.uniform(0.9, 1.1)),
                    holding_cost_rate=p["h"],
                    supplier_lead_time_days=round(lt_mean, 1),
                )
            )
            hidden.append(
                dict(
                    profile=profile,
                    base=base,
                    growth=rng.normal(0.08, 0.05),
                    uplift=rng.uniform(1.25, 1.6),
                    seas=p["seas"],
                    lt_mean=lt_mean,
                    lt_sd=lt_mean * rng.uniform(0.2, 0.4),
                    category=category,
                )
            )
    return pd.DataFrame(rows), hidden


def _simulate_sales(master: pd.DataFrame, hidden: list[dict], dates: pd.DatetimeIndex, rng):
    T, n = len(dates), len(master)
    t_years = np.arange(T) / 365.25
    month_idx = dates.month.to_numpy() - 1
    dow = dates.dayofweek.to_numpy()

    units = np.zeros((T, n), dtype=int)
    promo = np.zeros((T, n), dtype=bool)
    n_years = dates[-1].year - dates[0].year + 1

    for j, h in enumerate(hidden):
        season = MONTH_PROFILE ** h["seas"]
        season = season / season.mean()
        mu = h["base"] * np.exp(h["growth"] * t_years) * season[month_idx] * DOW_PROFILE[dow]
        if h["category"] == "Paints & Waterproofing":  # festive-season repainting
            mu = mu * np.where(np.isin(month_idx, [9, 10]), 1.15, 1.0)

        # promotions: ~4 windows per year, 5-10 days each
        mask = np.zeros(T, dtype=bool)
        for _ in range(4 * n_years):
            start = int(rng.integers(0, T - 10))
            mask[start : start + int(rng.integers(5, 11))] = True
        promo[:, j] = mask
        mu = mu * np.where(mask, h["uplift"], 1.0)

        # over-dispersed counts (gamma-Poisson = negative binomial); slow items are lumpier
        k = 3.0 if h["profile"] == "slow" else 14.0
        lam = mu * rng.gamma(k, 1.0 / k, size=T)
        d = rng.poisson(lam)

        # rare bulk project orders
        bulk = rng.random(T) < 0.004
        d = np.where(bulk, np.maximum(d, 1) * rng.integers(3, 5, size=T), d)
        units[:, j] = d

    sales = pd.DataFrame(
        {
            "date": np.repeat(dates.strftime("%Y-%m-%d"), n),
            "sku_id": np.tile(master["sku_id"].to_numpy(), T),
            "units_sold": units.ravel(),
            "promo_flag": promo.ravel().astype(int),
        }
    )
    price = master.set_index("sku_id")["unit_price"]
    disc = np.where(sales["promo_flag"] == 1, 0.90, 1.0)
    sales["unit_price"] = (sales["sku_id"].map(price).to_numpy() * disc).round(2)
    sales["revenue"] = (sales["units_sold"] * sales["unit_price"]).round(2)
    return sales


def _simulate_lead_times(master: pd.DataFrame, hidden: list[dict], dates: pd.DatetimeIndex, rng):
    rows = []
    for r, h in zip(master.itertuples(), hidden):
        order_date = dates[0]
        while True:
            order_date = order_date + pd.Timedelta(days=int(rng.integers(20, 45)))
            if order_date > dates[-1] - pd.Timedelta(days=15):
                break
            monsoon_delay = 1.0 if order_date.month in (6, 7, 8, 9) else 0.0
            lt = max(1, int(round(rng.normal(h["lt_mean"] + monsoon_delay, h["lt_sd"]))))
            rows.append(dict(sku_id=r.sku_id, order_date=order_date.strftime("%Y-%m-%d"), lead_time_days=lt))
    return pd.DataFrame(rows)


def _snapshot(master: pd.DataFrame, sales: pd.DataFrame, dates: pd.DatetimeIndex, rng):
    recent = sales[sales["date"] >= (dates[-1] - pd.Timedelta(days=59)).strftime("%Y-%m-%d")]
    avg = recent.groupby("sku_id")["units_sold"].mean()
    rows = []
    for sku in master["sku_id"]:
        cover_days = rng.uniform(2, 35)
        on_hand = int(round(avg[sku] * cover_days))
        on_order = int(round(avg[sku] * 14)) if rng.random() < 0.25 else 0
        rows.append(dict(sku_id=sku, snapshot_date=dates[-1].strftime("%Y-%m-%d"), on_hand=on_hand, on_order=on_order))
    return pd.DataFrame(rows)


def generate(out_dir: Path | str = config.DATA_DIR, seed: int = config.SEED) -> dict[str, pd.DataFrame]:
    """Generate and save all dataset files. Deterministic for a given seed."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.date_range(config.DATA_START, config.DATA_END, freq="D")

    master, hidden = _build_master(rng)
    sales = _simulate_sales(master, hidden, dates, rng)
    lead = _simulate_lead_times(master, hidden, dates, rng)
    snap = _snapshot(master, sales, dates, rng)

    tables = {
        "sku_master": master,
        "sales_history": sales,
        "lead_time_history": lead,
        "inventory_snapshot": snap,
    }
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    return tables


if __name__ == "__main__":
    t = generate()
    for k, v in t.items():
        print(f"{k:22s} {len(v):>7,d} rows")
