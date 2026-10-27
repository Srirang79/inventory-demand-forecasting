"""Unit tests for the formulas and forecasters. Run with:  python -m pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.forecasting import (
    METHODS, bias, f_croston_sba, f_moving_average, f_seasonal_naive, f_seasonal_ses, f_ses, ses_level, wape,
)
from src.inventory import abc_classification, eoq, reorder_point, safety_stock, stockout_probability, z_score


# ---- inventory formulas ---------------------------------------------------------------
def test_eoq_textbook_value():
    # D=1200, S=100, H=6  ->  sqrt(2*1200*100/6) = 200
    assert eoq(1200, 100, 6) == pytest.approx(200.0)


def test_eoq_zero_demand():
    assert eoq(0, 100, 6) == 0.0


def test_z_scores():
    assert z_score(0.95) == pytest.approx(1.645, abs=1e-3)
    assert z_score(0.50) == pytest.approx(0.0, abs=1e-9)


def test_safety_stock_and_rop():
    ss = safety_stock(z=1.645, sigma_ltd=30, daily_demand=10, sigma_lead_time=2)  # sqrt(900 + 400) = 36.06
    assert ss == pytest.approx(1.645 * 36.0555, rel=1e-3)
    assert reorder_point(10, 5, ss) == pytest.approx(50 + ss)


def test_safety_stock_grows_with_service_level_and_variability():
    base = safety_stock(z_score(0.90), 20, 5, 1)
    assert safety_stock(z_score(0.98), 20, 5, 1) > base
    assert safety_stock(z_score(0.90), 20, 5, 3) > base


def test_stockout_probability_is_50pct_when_stock_equals_expected_demand():
    assert stockout_probability(100, 100, 20) == pytest.approx(0.5)
    assert stockout_probability(200, 100, 20) < 0.001


def test_abc_classes_follow_revenue_rank():
    rev = pd.Series({"a": 80, "b": 12, "c": 5, "d": 3})
    cls = abc_classification(rev)
    assert cls["a"] == "A" and cls["d"] == "C"


# ---- forecasters ----------------------------------------------------------------------
def _series(n=800, level=50.0):
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return np.full(n, level), dates


def test_ses_level_of_constant_series_is_constant():
    assert ses_level(np.full(100, 7.0), 0.2) == pytest.approx(7.0)


@pytest.mark.parametrize("name", list(METHODS))
def test_every_method_returns_nonnegative_forecast_of_right_length(name):
    y, dates = _series()
    y = y + np.random.default_rng(0).poisson(5, len(y))
    spec = METHODS[name]
    f = spec.fn(y, dates, 14, **spec.grid[0])
    assert f.shape == (14,) and (f >= 0).all()


def test_constant_demand_is_forecast_exactly():
    y, dates = _series(level=40.0)
    for fn in (f_moving_average, f_ses, f_seasonal_naive):
        assert fn(y, dates, 5) == pytest.approx(np.full(5, 40.0))


def test_seasonal_ses_learns_weekly_pattern():
    dates = pd.date_range("2022-01-01", periods=1200, freq="D")
    y = np.where(dates.dayofweek == 6, 10.0, 100.0)  # Sundays are 10x lower
    f = f_seasonal_ses(y, dates, 7)
    future = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=7)
    assert f[list(future.dayofweek).index(6)] < 0.3 * f[list(future.dayofweek).index(2)]


def test_croston_handles_intermittent_and_all_zero():
    y = np.zeros(200)
    y[::10] = 5
    f = f_croston_sba(y, pd.date_range("2024-01-01", periods=200), 3)
    assert 0 < f[0] < 5
    assert (f_croston_sba(np.zeros(50), pd.date_range("2024-01-01", periods=50), 3) == 0).all()


# ---- metrics --------------------------------------------------------------------------
def test_wape_and_bias():
    a, f = np.array([10, 10, 10]), np.array([12, 8, 10])
    assert wape(a, f) == pytest.approx(4 / 30)
    assert bias(a, f) == pytest.approx(0.0)
    assert bias(a, a + 3) == pytest.approx(0.3)
    assert np.isnan(wape(np.zeros(3), np.ones(3)))


# ---- data -----------------------------------------------------------------------------
def test_dataset_generation_is_deterministic_and_complete(tmp_path):
    from src.data_generation import generate

    a = generate(tmp_path / "a", seed=1)
    b = generate(tmp_path / "b", seed=1)
    assert a["sales_history"].equals(b["sales_history"])
    sales = a["sales_history"]
    n_days = len(pd.date_range(config.DATA_START, config.DATA_END))
    assert len(sales) == n_days * a["sku_master"].shape[0]
    assert (sales["units_sold"] >= 0).all()
