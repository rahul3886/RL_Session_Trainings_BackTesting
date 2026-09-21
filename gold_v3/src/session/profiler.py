"""
Session profiler — slices data into London/NY sessions,
computes session profiles (bias, range, key levels).
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SessionProfile:
    """Profile of a single trading session."""
    date: str
    session_type: str           # "london" or "ny"
    open_price: float
    close_price: float
    high: float
    low: float
    mid: float
    range_points: float
    bias: int                   # +1 bullish, -1 bearish, 0 neutral
    body_ratio: float           # |close-open| / range
    num_candles: int


def slice_session(
    df: pd.DataFrame,
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """
    Extract candles within a time window (UTC).

    Args:
        df: Full OHLCV DataFrame with DatetimeIndex (UTC)
        start_time: "HH:MM" format
        end_time: "HH:MM" format

    Returns:
        Filtered DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have DatetimeIndex")

    start_h, start_m = map(int, start_time.split(":"))
    end_h, end_m = map(int, end_time.split(":"))

    start_minutes = start_h * 60 + start_m
    end_minutes = end_h * 60 + end_m

    idx_minutes = df.index.hour * 60 + df.index.minute

    mask = (idx_minutes >= start_minutes) & (idx_minutes < end_minutes)
    return df[mask]


def get_trading_days(
    df: pd.DataFrame,
    london_start: str = "08:00",
    london_end: str = "17:00",
    ny_start: str = "12:00",
    ny_end: str = "21:00",
    min_london_candles: int = 10,
    min_ny_candles: int = 10,
) -> List[pd.Timestamp]:
    """
    Get list of valid trading days that have both London and NY sessions.
    """
    dates = sorted(set(df.index.date))
    valid_dates = []

    for date in dates:
        day_data = df[df.index.date == date]
        london = slice_session(day_data, london_start, london_end)
        ny = slice_session(day_data, ny_start, ny_end)

        if len(london) >= min_london_candles and len(ny) >= min_ny_candles:
            valid_dates.append(pd.Timestamp(date))

    return valid_dates


def compute_session_profile(
    candles: pd.DataFrame,
    session_type: str = "london",
) -> SessionProfile:
    """
    Compute profile for a session's candles.
    """
    if len(candles) == 0:
        return SessionProfile(
            date="", session_type=session_type,
            open_price=0, close_price=0, high=0, low=0, mid=0,
            range_points=0, bias=0, body_ratio=0, num_candles=0,
        )

    open_price = candles["open"].iloc[0]
    close_price = candles["close"].iloc[-1]
    high = candles["high"].max()
    low = candles["low"].min()
    mid = (high + low) / 2.0
    range_pts = high - low
    body = abs(close_price - open_price)
    body_ratio = body / max(range_pts, 0.01)

    if close_price > open_price:
        bias = 1
    elif close_price < open_price:
        bias = -1
    else:
        bias = 0

    return SessionProfile(
        date=str(candles.index[0].date()),
        session_type=session_type,
        open_price=open_price,
        close_price=close_price,
        high=high,
        low=low,
        mid=mid,
        range_points=range_pts,
        bias=bias,
        body_ratio=body_ratio,
        num_candles=len(candles),
    )


def get_session_pair(
    df: pd.DataFrame,
    date: pd.Timestamp,
    session_cfg=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, SessionProfile, SessionProfile]:
    """
    Get London and NY candles + profiles for a given date.

    Returns:
        (london_candles, ny_candles, london_profile, ny_profile)
    """
    london_start = "08:00"
    london_end = "17:00"
    ny_start = "12:00"
    ny_end = "21:00"

    if session_cfg is not None:
        london_start = session_cfg.london_start
        london_end = session_cfg.london_end
        ny_start = session_cfg.ny_start
        ny_end = session_cfg.ny_end

    day_data = df[df.index.date == date.date()] if hasattr(date, 'date') else df[df.index.date == date]

    london = slice_session(day_data, london_start, london_end)
    ny = slice_session(day_data, ny_start, ny_end)

    london_profile = compute_session_profile(london, "london")
    ny_profile = compute_session_profile(ny, "ny")

    return london, ny, london_profile, ny_profile


def get_london_until_anchor(
    df: pd.DataFrame,
    date,
    london_start: str = "08:00",
    anchor_time: str = "12:15",
) -> pd.DataFrame:
    """
    Get London candles from start until anchor time (12:15 UTC).
    This is the observation window for Lens 1.
    """
    day_data = df[df.index.date == (date.date() if hasattr(date, 'date') else date)]
    return slice_session(day_data, london_start, anchor_time)


def get_ny_after_anchor(
    df: pd.DataFrame,
    date,
    anchor_time: str = "12:15",
    ny_end: str = "21:00",
) -> pd.DataFrame:
    """
    Get NY candles from anchor time (12:15 UTC) to NY close.
    This is the evaluation window for Lens 1 / trading window for Lens 2.
    """
    day_data = df[df.index.date == (date.date() if hasattr(date, 'date') else date)]
    return slice_session(day_data, anchor_time, ny_end)
