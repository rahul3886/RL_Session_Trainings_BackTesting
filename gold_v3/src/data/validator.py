"""
OHLCV data validation and gap detection.
Ensures data quality before any analysis or training.
"""

import logging
from typing import Tuple, Dict

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def validate_data(df: pd.DataFrame, label: str = "data") -> Tuple[pd.DataFrame, Dict]:
    """
    Validate OHLCV data and return cleaned DataFrame with stats.

    Steps:
      1. Drop NaN OHLC rows
      2. Drop zero/negative volume
      3. Drop duplicate timestamps
      4. Sort by datetime UTC
      5. Warn on gaps > 3× median interval
      6. Log final clean bar count

    Returns:
      (cleaned_df, validation_stats)
    """
    stats = {
        "label": label,
        "original_rows": len(df),
        "nan_dropped": 0,
        "zero_vol_dropped": 0,
        "duplicate_dropped": 0,
        "gaps_detected": 0,
        "final_rows": 0,
        "date_range_start": None,
        "date_range_end": None,
    }

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError(f"DataFrame must have DatetimeIndex, got {type(df.index)}")

    required_cols = ["open", "high", "low", "close"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # 1. Drop NaN OHLC rows
    before = len(df)
    df = df.dropna(subset=["open", "high", "low", "close"])
    stats["nan_dropped"] = before - len(df)
    if stats["nan_dropped"] > 0:
        logger.info(f"  [{label}] Dropped {stats['nan_dropped']} NaN rows")

    # 2. Drop zero/negative volume
    if "volume" in df.columns:
        before = len(df)
        df = df[df["volume"] > 0]
        stats["zero_vol_dropped"] = before - len(df)
        if stats["zero_vol_dropped"] > 0:
            logger.info(f"  [{label}] Dropped {stats['zero_vol_dropped']} zero/neg volume rows")

    # 3. Drop duplicates
    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    stats["duplicate_dropped"] = before - len(df)
    if stats["duplicate_dropped"] > 0:
        logger.info(f"  [{label}] Dropped {stats['duplicate_dropped']} duplicate timestamps")

    # 4. Sort by datetime
    df = df.sort_index()

    # 5. Gap detection
    if len(df) > 1:
        intervals = df.index.to_series().diff().dropna()
        median_interval = intervals.median()
        threshold = median_interval * 3
        gaps = intervals[intervals > threshold]
        stats["gaps_detected"] = len(gaps)

        if len(gaps) > 0:
            logger.warning(
                f"  [{label}] Found {len(gaps)} gaps > 3× median interval "
                f"(median={median_interval}, threshold={threshold})"
            )
            # Log first 5 gaps
            for i, (ts, gap) in enumerate(gaps.items()):
                if i >= 5:
                    logger.warning(f"    ... and {len(gaps) - 5} more gaps")
                    break
                logger.warning(f"    Gap at {ts}: {gap}")

    # 6. Validate OHLC consistency
    invalid_hl = df[df["high"] < df["low"]]
    if len(invalid_hl) > 0:
        logger.warning(f"  [{label}] {len(invalid_hl)} bars with high < low — fixing")
        # Swap high and low
        mask = df["high"] < df["low"]
        df.loc[mask, ["high", "low"]] = df.loc[mask, ["low", "high"]].values

    # Final stats
    stats["final_rows"] = len(df)
    stats["date_range_start"] = str(df.index[0]) if len(df) > 0 else None
    stats["date_range_end"] = str(df.index[-1]) if len(df) > 0 else None

    logger.info(
        f"  [{label}] Validation complete: {stats['final_rows']} clean bars, "
        f"{stats['date_range_start']} to {stats['date_range_end']}"
    )

    return df, stats


def count_trading_days(df: pd.DataFrame) -> int:
    """Count unique trading days in the dataset."""
    return len(np.unique(df.index.date)) if len(df) > 0 else 0


def split_data(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    validate_start: str,
    validate_end: str,
    test_start: str,
    test_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/validate/test using walk-forward dates."""
    train = df[train_start:train_end].iloc[:-1] if train_end in df.index else df[train_start:train_end]
    validate = df[validate_start:validate_end].iloc[:-1] if validate_end in df.index else df[validate_start:validate_end]
    test = df[test_start:test_end].iloc[:-1] if test_end in df.index else df[test_start:test_end]

    # Safer slicing
    train = df.loc[(df.index >= pd.Timestamp(train_start, tz="UTC")) &
                   (df.index < pd.Timestamp(train_end, tz="UTC"))]
    validate = df.loc[(df.index >= pd.Timestamp(validate_start, tz="UTC")) &
                      (df.index < pd.Timestamp(validate_end, tz="UTC"))]
    test = df.loc[(df.index >= pd.Timestamp(test_start, tz="UTC")) &
                  (df.index < pd.Timestamp(test_end, tz="UTC"))]

    logger.info(f"  Train:    {len(train)} bars ({count_trading_days(train)} days)")
    logger.info(f"  Validate: {len(validate)} bars ({count_trading_days(validate)} days)")
    logger.info(f"  Test:     {len(test)} bars ({count_trading_days(test)} days)")

    return train, validate, test
