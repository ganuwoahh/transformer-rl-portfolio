"""Central configuration. Every tunable lives here so experiments are reproducible."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# On Colab these are redirected to Google Drive via environment variables so that
# cached data and checkpoints survive session disconnects.
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))

# --- Universe ---------------------------------------------------------------
ASSETS = [
    "AAPL", "MSFT", "NVDA", "AMZN",  # tech / growth
    "JPM",                            # financials
    "XOM",                            # energy
    "JNJ", "PG",                      # defensives
    "QQQ",                            # equity index
    "GLD", "TLT",                     # gold, long bonds
    "BTC-USD",                        # crypto
]
BENCHMARK = "SPY"        # benchmark only, never traded by the agent
CALENDAR_TICKER = "SPY"  # all series are aligned to this ticker's trading days
MACRO_TICKERS = {"vix": "^VIX", "tnx": "^TNX", "dxy": "DX-Y.NYB"}
RF_TICKER = "^IRX"       # 13-week T-bill yield (annualised %): cash return + Sharpe risk-free

# --- Dates ------------------------------------------------------------------
# Start is early enough that, after the ~1 year feature warm-up, usable data
# begins around the start of 2016.
DATA_START = "2014-10-01"
DATA_END = None          # None = up to today
TRAIN_END = "2020-12-31"
VAL_END = "2022-12-31"   # test = everything after this date

# --- Features ---------------------------------------------------------------
TRADING_DAYS = 252
VOL_WINDOWS = (14, 30)
SHARPE_MOM_WINDOW = 60
RSI_WINDOW = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOLUME_WINDOW = 20
ZSCORE_WINDOW = 252      # trailing window for causal z-scoring
ZSCORE_CLIP = 5.0
