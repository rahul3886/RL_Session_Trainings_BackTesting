"""
Liquidity mapping — equal highs/lows, PDH/PDL, sweep detection.
Institutional liquidity pools form where retail stop-losses cluster.
"""

import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def build_pdh_pdl_cache(df: Optional[pd.DataFrame]) -> Dict:
    """
    Pre-compute previous-day high/low values keyed by trading date.
    """
    if df is None or len(df) == 0:
        return {}

    daily = (
        df.groupby(df.index.date)
        .agg(high=("high", "max"), low=("low", "min"))
        .sort_index()
    )

    cache: Dict = {}
    prev_high = 0.0
    prev_low = 0.0

    for trading_date, row in daily.iterrows():
        cache[trading_date] = {"pdh": float(prev_high), "pdl": float(prev_low)}
        prev_high = float(row["high"])
        prev_low = float(row["low"])

    return cache


@dataclass
class LiquidityLevel:
    """A detected liquidity pool."""
    type: str          # "buy_side" (above, equal highs) or "sell_side" (below, equal lows)
    price: float
    touches: int       # number of times price touched this level
    swept: bool = False


def detect_equal_levels(
    candles: pd.DataFrame,
    tolerance_atr: float = 0.15,
    min_touches: int = 2,
    atr_val: float = 1.0,
) -> Tuple[List[LiquidityLevel], List[LiquidityLevel]]:
    """
    Detect equal highs and equal lows — engineered liquidity.

    Equal highs = buy-side liquidity (stops above)
    Equal lows  = sell-side liquidity (stops below)

    Args:
        candles: OHLCV DataFrame
        tolerance_atr: how close highs/lows must be (ATR fraction)
        min_touches: minimum touches to qualify
        atr_val: current ATR value

    Returns:
        (buy_side_levels, sell_side_levels)
    """
    highs = candles["high"].values
    lows = candles["low"].values
    tol = tolerance_atr * atr_val

    buy_side: List[LiquidityLevel] = []
    sell_side: List[LiquidityLevel] = []

    # Cluster equal highs
    high_clusters = _cluster_levels(highs, tol)
    for price, count in high_clusters:
        if count >= min_touches:
            buy_side.append(LiquidityLevel(
                type="buy_side", price=price, touches=count
            ))

    # Cluster equal lows
    low_clusters = _cluster_levels(lows, tol)
    for price, count in low_clusters:
        if count >= min_touches:
            sell_side.append(LiquidityLevel(
                type="sell_side", price=price, touches=count
            ))

    return buy_side, sell_side


def _cluster_levels(values: np.ndarray, tolerance: float) -> List[Tuple[float, int]]:
    """Cluster nearby price levels and count touches."""
    if len(values) == 0:
        return []

    sorted_vals = np.sort(values)
    clusters: List[Tuple[float, int]] = []
    cluster_start = 0

    for i in range(1, len(sorted_vals)):
        if sorted_vals[i] - sorted_vals[cluster_start] > tolerance:
            cluster_mean = np.mean(sorted_vals[cluster_start:i])
            cluster_count = i - cluster_start
            clusters.append((cluster_mean, cluster_count))
            cluster_start = i

    # Last cluster
    cluster_mean = np.mean(sorted_vals[cluster_start:])
    cluster_count = len(sorted_vals) - cluster_start
    clusters.append((cluster_mean, cluster_count))

    return clusters


def resolve_sweep_threshold(
    session_high: float,
    session_low: float,
    threshold_points: Optional[float] = None,
    threshold_lr: Optional[float] = None,
) -> float:
    """
    Resolve the sweep threshold in price points.

    The preferred form is a London-range fraction (`threshold_lr`), but
    `threshold_points` is still accepted for legacy callers and tests.
    """
    if threshold_points is not None:
        return float(threshold_points)

    if threshold_lr is None:
        threshold_lr = 0.12

    session_range = max(session_high - session_low, 0.01)
    return float(session_range * threshold_lr)


def detect_sweep(
    current_price: float,
    session_high: float,
    session_low: float,
    threshold_points: Optional[float] = None,
    candles_after: pd.DataFrame = None,
    threshold_lr: Optional[float] = None,
    require_reversal: bool = True,
) -> Dict[str, bool]:
    """
    Detect liquidity sweep — price spikes beyond level and reverses.

    A sweep occurs when price:
      1. Breaks above session high (or below session low)
      2. By more than the configured threshold
      3. Then reverses back inside the range

    Args:
        current_price: current market price
        session_high: session high to check sweep above
        session_low: session low to check sweep below
        threshold_points: legacy minimum overshoot in points
        threshold_lr: preferred overshoot as a London-range fraction
        candles_after: candles after the potential sweep for reversal check
        require_reversal: when True, only count a sweep after price closes back
            inside the reference range

    Returns:
        dict with 'sweep_high' and 'sweep_low' booleans
    """
    result = {"sweep_high": False, "sweep_low": False}
    threshold = resolve_sweep_threshold(
        session_high,
        session_low,
        threshold_points=threshold_points,
        threshold_lr=threshold_lr,
    )

    if candles_after is not None and len(candles_after) > 0:
        max_after = candles_after["high"].max()
        min_after = candles_after["low"].min()
        last_close = candles_after["close"].iloc[-1]

        # Sweep high: price spiked above session high and reversed back below
        if max_after > session_high + threshold:
            if (not require_reversal) or last_close < session_high:
                result["sweep_high"] = True

        # Sweep low: price spiked below session low and reversed back above
        if min_after < session_low - threshold:
            if (not require_reversal) or last_close > session_low:
                result["sweep_low"] = True
    elif not require_reversal:
        # Simple check without reversal confirmation
        if current_price > session_high + threshold:
            result["sweep_high"] = True
        if current_price < session_low - threshold:
            result["sweep_low"] = True

    return result


def compute_pdh_pdl(
    df: pd.DataFrame,
    current_date: pd.Timestamp,
    pdh_pdl_cache: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Compute Previous Day High and Previous Day Low.
    These are key institutional liquidity targets.
    """
    result = {"pdh": 0.0, "pdl": 0.0}

    if current_date is None:
        return result

    current_day = current_date.date()
    if pdh_pdl_cache is not None:
        cached = pdh_pdl_cache.get(current_day)
        if cached is not None:
            return {
                "pdh": float(cached.get("pdh", 0.0)),
                "pdl": float(cached.get("pdl", 0.0)),
            }

    # Get previous trading day
    prev_day_data = df[df.index.date < current_day]
    if len(prev_day_data) == 0:
        return result

    last_date = prev_day_data.index.date[-1]
    prev_day = prev_day_data[prev_day_data.index.date == last_date]

    if len(prev_day) > 0:
        result["pdh"] = prev_day["high"].max()
        result["pdl"] = prev_day["low"].min()

    return result


def map_liquidity_levels(
    candles: pd.DataFrame,
    session_high: float,
    session_low: float,
    current_price: float,
    atr_val: float,
    tolerance_atr: float = 0.15,
    min_touches: int = 2,
    sweep_threshold_lr: float = 0.12,
    pdh: float = 0.0,
    pdl: float = 0.0,
) -> Dict:
    """
    Comprehensive liquidity mapping for state vector.

    Returns:
        dict with all liquidity features for the RL state
    """
    buy_side, sell_side = detect_equal_levels(
        candles, tolerance_atr, min_touches, atr_val
    )

    sweep = detect_sweep(
        current_price, session_high, session_low,
        candles_after=candles,
        threshold_lr=sweep_threshold_lr,
        require_reversal=True,
    )

    atr_safe = max(atr_val, 0.01)

    return {
        "equal_highs_present": 1.0 if len(buy_side) > 0 else 0.0,
        "equal_lows_present": 1.0 if len(sell_side) > 0 else 0.0,
        "liq_above_density": min(len(buy_side), 5) / 5.0,
        "liq_below_density": min(len(sell_side), 5) / 5.0,
        "pdh_dist": (pdh - current_price) / atr_safe if pdh > 0 else 0.0,
        "pdl_dist": (current_price - pdl) / atr_safe if pdl > 0 else 0.0,
        "sweep_high": 1.0 if sweep["sweep_high"] else 0.0,
        "sweep_low": 1.0 if sweep["sweep_low"] else 0.0,
        "buy_side_levels": buy_side,
        "sell_side_levels": sell_side,
    }
