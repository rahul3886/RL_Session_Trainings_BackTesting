"""
Fair Value Gap (FVG) detection — 3-candle imbalance zones.
An FVG is created when candle 1's high is below candle 3's low (bullish)
or candle 1's low is above candle 3's high (bearish), leaving a gap
that price has not efficiently traded through.
"""

import logging
from typing import List, Dict
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FairValueGap:
    """Represents a detected FVG."""
    type: str          # "bull" or "bear"
    high: float        # top of the gap
    low: float         # bottom of the gap
    size: float        # gap size in points
    origin_idx: int    # candle index where FVG formed
    filled: bool = False
    fill_pct: float = 0.0
    age: int = 0


def detect_fvgs(
    candles: pd.DataFrame,
    atr: pd.Series = None,
    min_size_atr: float = 0.3,
    max_age: int = 30,
) -> List[FairValueGap]:
    """
    Detect Fair Value Gaps (3-candle imbalance).

    Bullish FVG: candle[i-2].high < candle[i].low  (gap up)
    Bearish FVG: candle[i-2].low > candle[i].high   (gap down)

    Args:
        candles: OHLCV DataFrame
        atr: ATR series for minimum size filter
        min_size_atr: minimum FVG size as ATR multiple
        max_age: maximum candle age before expiry

    Returns:
        List of FairValueGap objects
    """
    fvgs: List[FairValueGap] = []

    if len(candles) < 3:
        return fvgs

    highs = candles["high"].values
    lows = candles["low"].values
    closes = candles["close"].values

    atr_vals = atr.values if atr is not None else np.ones(len(candles))

    for i in range(2, len(candles)):
        atr_val = atr_vals[i] if i < len(atr_vals) and atr_vals[i] > 0 else 1.0
        age = len(candles) - 1 - i

        if age > max_age:
            continue

        # Bullish FVG: candle[i-2] high < candle[i] low
        if highs[i - 2] < lows[i]:
            gap_size = lows[i] - highs[i - 2]
            if gap_size / atr_val >= min_size_atr:
                fvg = FairValueGap(
                    type="bull",
                    high=lows[i],
                    low=highs[i - 2],
                    size=gap_size,
                    origin_idx=i,
                    age=age,
                )
                fvgs.append(fvg)

        # Bearish FVG: candle[i-2] low > candle[i] high
        if lows[i - 2] > highs[i]:
            gap_size = lows[i - 2] - highs[i]
            if gap_size / atr_val >= min_size_atr:
                fvg = FairValueGap(
                    type="bear",
                    high=lows[i - 2],
                    low=highs[i],
                    size=gap_size,
                    origin_idx=i,
                    age=age,
                )
                fvgs.append(fvg)

    # Check fill status against current price
    if len(candles) > 0:
        current_price = closes[-1]
        active_fvgs = []
        for fvg in fvgs:
            # Check if price has filled the gap
            if fvg.type == "bull":
                if current_price <= fvg.low:
                    fvg.filled = True
                    fvg.fill_pct = 1.0
                elif current_price < fvg.high:
                    fvg.fill_pct = (fvg.high - current_price) / max(fvg.size, 0.01)
            elif fvg.type == "bear":
                if current_price >= fvg.high:
                    fvg.filled = True
                    fvg.fill_pct = 1.0
                elif current_price > fvg.low:
                    fvg.fill_pct = (current_price - fvg.low) / max(fvg.size, 0.01)

            if not fvg.filled:
                active_fvgs.append(fvg)

        fvgs = active_fvgs

    return fvgs


def get_nearest_fvg(
    fvgs: List[FairValueGap],
    current_price: float,
    fvg_type: str,
    atr_val: float,
) -> Dict:
    """Find nearest unfilled FVG of given type."""
    result = {"active": False, "distance_atr": 5.0, "size_atr": 0.0}

    matching = [f for f in fvgs if f.type == fvg_type and not f.filled]
    if not matching:
        return result

    if fvg_type == "bull":
        # Bull FVGs below price
        below = [f for f in matching if f.high <= current_price]
        if below:
            nearest = min(below, key=lambda f: current_price - f.high)
            dist = (current_price - nearest.high) / max(atr_val, 0.01)
            result = {
                "active": True,
                "distance_atr": dist,
                "size_atr": nearest.size / max(atr_val, 0.01),
            }
    else:
        # Bear FVGs above price
        above = [f for f in matching if f.low >= current_price]
        if above:
            nearest = min(above, key=lambda f: f.low - current_price)
            dist = (nearest.low - current_price) / max(atr_val, 0.01)
            result = {
                "active": True,
                "distance_atr": dist,
                "size_atr": nearest.size / max(atr_val, 0.01),
            }

    return result


def total_fvg_imbalance(fvgs: List[FairValueGap], atr_val: float) -> float:
    """Total imbalance magnitude of all active FVGs."""
    total = sum(f.size for f in fvgs if not f.filled)
    return total / max(atr_val, 0.01)
