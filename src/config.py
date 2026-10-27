"""Central configuration: paths, date windows and modelling constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
TABLE_DIR = OUT_DIR / "tables"

SEED = 42

# ---- Dataset span -----------------------------------------------------------
DATA_START = "2022-01-01"
DATA_END = "2025-12-31"

# ---- Time-based split (no shuffling: forecasting must respect time order) ----
#   train      : DATA_START .. day before VAL_START   (fit models)
#   validation : VAL_START  .. day before TEST_START  (pick method + smoothing params per SKU)
#   test       : TEST_START .. DATA_END               (honest hold-out, never used for selection)
VAL_START = "2025-07-01"
TEST_START = "2025-10-01"

# ---- Forecast evaluation ----------------------------------------------------
EVAL_HORIZON = 14       # days ahead scored in the rolling-origin backtest
FORECAST_HORIZON = 21   # days ahead each backtest origin forecasts (used by the policy simulation)
PLAN_HORIZON = 28       # days ahead for the forward-looking replenishment plan
ORIGIN_STEP = 7         # a new forecast origin every 7 days (weekly replanning)

# ---- Inventory policy -------------------------------------------------------
# Target cycle service level by ABC class (A = highest-revenue SKUs)
SERVICE_LEVELS = {"A": 0.98, "B": 0.95, "C": 0.90}
ABC_CUTOFFS = (0.70, 0.90)  # cumulative revenue share: A <= 70%, B <= 90%, C = rest

# ---- Policy simulation ------------------------------------------------------
SIM_REPLICATIONS = 30   # Monte-Carlo repetitions over random supplier lead times
