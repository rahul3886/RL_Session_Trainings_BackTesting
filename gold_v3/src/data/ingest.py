"""
Data ingestion from Dukascopy API with CSV fallback.
Primary source: dukascopy-python library (institutional tick data).
Fallback: pre-downloaded CSVs in data/raw/.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


import MetaTrader5 as mt5

# ─── MT5 timeframe mapping ──────────────────────────────────────
TIMEFRAME_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}


def connect_mt5(login=None, password=None, server=None) -> bool:
    """
    Initialise MT5 terminal connection.
    Credentials are optional — if terminal is already open and
    logged in, mt5.initialize() with no args works fine.
    Reads MT5_LOGIN, MT5_PASSWORD, MT5_SERVER from env vars
    as override if not passed directly.
    """
    login    = login    or os.environ.get("MT5_LOGIN")
    password = password or os.environ.get("MT5_PASSWORD")
    server   = server   or os.environ.get("MT5_SERVER")

    kwargs = {}
    if login:    kwargs["login"] = int(login)
    if password: kwargs["password"] = password
    if server:   kwargs["server"] = server

    if not mt5.initialize(**kwargs):
        raise ConnectionError(
            f"MT5 initialisation failed: {mt5.last_error()}\n"
            f"Make sure MetaTrader5 terminal is open and logged in."
        )
    return True


def disconnect_mt5():
    mt5.shutdown()


def fetch_mt5(
    instrument: str,
    start: str,
    end: str,
    timeframe: str,
    raw_dir: str,
) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV from MT5 terminal.

    IMPORTANT — the MT5 180-day limit workaround:
    MT5 brokers often restrict copy_rates_range to ~180 days on
    live accounts. To get 3 years:
      1. Open the XAUUSD chart in MT5 manually
      2. Scroll back to January 2022 and hold until bars load
      3. MT5 caches the data locally once loaded
      4. copy_rates_range will then return the full cached range
    """
    tf_const = TIMEFRAME_MAP.get(timeframe.upper())
    if tf_const is None:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt   = datetime.strptime(end,   "%Y-%m-%d")

    connect_mt5()
    try:
        # Normalise symbol — MT5 brokers may use XAUUSD or XAUUSDm etc.
        symbol = instrument.upper()
        symbols_available = [s.name for s in mt5.symbols_get()]

        if symbol not in symbols_available:
            # Try common variants
            for variant in [symbol + "m", symbol + ".", "GOLD"]:
                if variant in symbols_available:
                    symbol = variant
                    logger.info(f"Symbol remapped to: {symbol}")
                    break

        rates = mt5.copy_rates_range(symbol, tf_const, start_dt, end_dt)

        if rates is None or len(rates) == 0:
            logger.error(
                f"MT5 returned no data for {symbol} {timeframe} "
                f"{start}→{end}. Error: {mt5.last_error()}\n"
                f"Tip: Open the chart in MT5, scroll back to {start}, "
                f"wait for bars to load, then retry."
            )
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.set_index("time")
        df.index.name = "datetime"
        if "tick_volume" in df.columns:
            df = df.rename(columns={"tick_volume": "volume"})
        elif "real_volume" in df.columns:
            df = df.rename(columns={"real_volume": "volume"})
        df = df[["open", "high", "low", "close", "volume"]]

        # Save raw CSV for future fallback
        os.makedirs(raw_dir, exist_ok=True)
        fname = f"{instrument}_{timeframe}_{start}_{end}_mt5.csv"
        df.to_csv(os.path.join(raw_dir, fname))
        logger.info(
            f"MT5 fetched {len(df)} bars for {symbol} {timeframe} "
            f"({start} → {end}). Saved to {fname}"
        )
        return df

    finally:
        disconnect_mt5()


def load_csv(raw_dir: str, instrument: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Load pre-downloaded CSV files from raw directory.
    Auto-detects files matching instrument and timeframe pattern.
    """
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.warning(f"Raw directory not found: {raw_dir}")
        return None

    # Find matching files
    pattern = f"{instrument.lower()}*{timeframe.lower()}*"
    candidates = list(raw_path.glob(f"*{instrument.upper()}*{timeframe}*")) + \
                 list(raw_path.glob(f"*{instrument.lower()}*{timeframe.lower()}*"))

    if not candidates:
        # Try any CSV in directory
        candidates = list(raw_path.glob("*.csv"))

    if not candidates:
        logger.warning(f"No CSV files found in {raw_dir}")
        return None

    logger.info(f"Found {len(candidates)} CSV file(s) in {raw_dir}")

    # Load and concatenate all matching files
    dfs = []
    for fpath in candidates:
        try:
            df = pd.read_csv(fpath, parse_dates=True)
            df = _standardise_columns(df)
            dfs.append(df)
            logger.info(f"  Loaded {fpath.name}: {len(df)} bars")
        except Exception as e:
            logger.warning(f"  Failed to load {fpath.name}: {e}")

    if not dfs:
        return None

    combined = pd.concat(dfs, ignore_index=False)
    combined = combined.sort_index().drop_duplicates()
    return combined


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names to lowercase OHLCV format."""
    # Common column name mappings
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("open", "o"):
            col_map[col] = "open"
        elif cl in ("high", "h"):
            col_map[col] = "high"
        elif cl in ("low", "l"):
            col_map[col] = "low"
        elif cl in ("close", "c"):
            col_map[col] = "close"
        elif cl in ("volume", "vol", "v", "tickvolume"):
            col_map[col] = "volume"
        elif cl in ("timestamp", "time", "date", "datetime"):
            col_map[col] = "datetime"

    df = df.rename(columns=col_map)

    # Try to set datetime index
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df = df.set_index("datetime")
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try parsing index as datetime
        try:
            df.index = pd.to_datetime(df.index, utc=True)
            df.index.name = "datetime"
        except Exception:
            pass

    # Ensure UTC
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    # Ensure volume exists
    if "volume" not in df.columns:
        df["volume"] = 1.0  # Tick count fallback

    return df


def ingest_data(
    instrument: str,
    start: str,
    end: str,
    timeframe: str,
    raw_dir: str,
    processed_dir: str,
) -> pd.DataFrame:
    """
    Main ingestion entry point.
    Strategy: Check raw_dir for CSVs first → if empty, fetch from Dukascopy API.
    """
    # Check if processed parquet already exists
    os.makedirs(processed_dir, exist_ok=True)
    parquet_path = os.path.join(processed_dir, f"{instrument}_{timeframe}.parquet")
    if os.path.exists(parquet_path):
        logger.info(f"Loading processed data from {parquet_path}")
        df = pd.read_parquet(parquet_path)
        if isinstance(df.index, pd.DatetimeIndex):
            logger.info(f"  {len(df)} bars, {df.index[0]} to {df.index[-1]}")
            return df

    # Try CSV fallback first (faster, no API dependency)
    df = load_csv(raw_dir, instrument, timeframe)

    # If no CSV files, try MT5 API
    if df is None or df.empty:
        logger.info("No local CSV data found. Attempting MT5 API...")
        df = fetch_mt5(instrument, start, end, timeframe, raw_dir)

    if df is None or df.empty:
        raise RuntimeError(
            f"Failed to load data for {instrument} {timeframe}. Options:\n"
            f"  1. Place CSV files in {raw_dir}/\n"
            f"  2. Open MT5 terminal, scroll XAUUSD chart back to 2022,\n"
            f"     then run: python main.py --mode download\n"
            f"  3. Set MT5_LOGIN / MT5_PASSWORD / MT5_SERVER env vars\n"
            f"     if terminal requires login"
        )

    # Save as parquet for fast future loading
    df.to_parquet(parquet_path)
    logger.info(f"Saved processed data to {parquet_path} ({len(df)} bars)")

    return df
