"""
Order Block (OB) detection — institutional supply/demand zones.
An OB forms when a strong displacement candle (> ATR threshold)
engulfs the previous candle's body, creating an imbalance zone.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OrderBlock:
    """Represents a detected order block."""
    type: str           # "bull" or "bear"
    high: float
    low: float
    origin_idx: int     # index in candle array where OB formed
    strength: float     # displacement magnitude / ATR
    age: int = 0        # candles since formation
    mitigated: bool = False


def detect_order_blocks(
    candles: pd.DataFrame,
    atr: pd.Series,
    displacement_threshold: float = 1.5,
    lookback: int = 20,
    max_age: int = 50,
) -> List[OrderBlock]:
    """
    Detect order blocks using ATR displacement.

    A Bullish OB: bearish candle followed by strong bullish displacement
    A Bearish OB: bullish candle followed by strong bearish displacement

    Args:
        candles: OHLCV DataFrame
        atr: ATR series aligned with candles
        displacement_threshold: minimum move / ATR to qualify
        lookback: how many candles back to scan
        max_age: maximum age before OB expires

    Returns:
        List of active OrderBlock objects
    """
    obs: List[OrderBlock] = []

    if len(candles) < 3:
        return obs

    opens = candles["open"].values
    highs = candles["high"].values
    lows = candles["low"].values
    closes = candles["close"].values
    atr_vals = atr.values

    start_idx = max(1, len(candles) - lookback)

    for i in range(start_idx, len(candles) - 1):
        atr_val = atr_vals[i] if i < len(atr_vals) and atr_vals[i] > 0 else 1.0

        # Current candle body direction
        curr_body = closes[i] - opens[i]
        # Next candle displacement
        next_body = closes[i + 1] - opens[i + 1]
        next_displacement = abs(next_body) / atr_val

        if next_displacement < displacement_threshold:
            continue

        # BULLISH OB: bearish candle (down) → strong bullish candle (up)
        if curr_body < 0 and next_body > 0:
            ob = OrderBlock(
                type="bull",
                high=opens[i],      # top of the bearish candle body
                low=lows[i],        # low of the bearish candle
                origin_idx=i,
                strength=next_displacement,
                age=len(candles) - 1 - i,
            )
            obs.append(ob)

        # BEARISH OB: bullish candle (up) → strong bearish candle (down)
        elif curr_body > 0 and next_body < 0:
            ob = OrderBlock(
                type="bear",
                high=highs[i],      # high of the bullish candle
                low=opens[i],       # bottom of the bullish candle body
                origin_idx=i,
                strength=next_displacement,
                age=len(candles) - 1 - i,
            )
            obs.append(ob)

    # Filter: remove mitigated OBs (price has traded through them)
    if len(candles) > 0:
        current_price = closes[-1]
        active_obs = []
        for ob in obs:
            if ob.age > max_age:
                continue
            # Check mitigation
            if ob.type == "bull" and current_price < ob.low:
                ob.mitigated = True
                continue
            if ob.type == "bear" and current_price > ob.high:
                ob.mitigated = True
                continue
            active_obs.append(ob)
        obs = active_obs

    return obs


def get_nearest_ob(
    obs: List[OrderBlock],
    current_price: float,
    ob_type: str,
    atr_val: float,
) -> Dict:
    """
    Find the nearest OB of given type and return distance/strength.

    Returns:
        dict with 'distance_atr' and 'strength'
    """
    result = {"distance_atr": 5.0, "strength": 0.0, "level": 0.0}

    matching = [ob for ob in obs if ob.type == ob_type and not ob.mitigated]
    if not matching:
        return result

    if ob_type == "bull":
        # Nearest bull OB below current price
        below = [ob for ob in matching if ob.high <= current_price]
        if below:
            nearest = min(below, key=lambda ob: current_price - ob.high)
            dist = (current_price - nearest.high) / max(atr_val, 0.01)
            result = {
                "distance_atr": dist,
                "strength": nearest.strength,
                "level": nearest.low,
            }
    else:
        # Nearest bear OB above current price
        above = [ob for ob in matching if ob.low >= current_price]
        if above:
            nearest = min(above, key=lambda ob: ob.low - current_price)
            dist = (nearest.low - current_price) / max(atr_val, 0.01)
            result = {
                "distance_atr": dist,
                "strength": nearest.strength,
                "level": nearest.high,
            }

    return result


def ob_at_session_level(
    obs: List[OrderBlock],
    session_high: float,
    session_low: float,
    atr_val: float,
    tolerance_atr: float = 0.5,
) -> Dict[str, bool]:
    """Check if OBs exist near session high/low extremes."""
    tol = tolerance_atr * atr_val

    ob_at_high = any(
        ob.type == "bear" and abs(ob.low - session_high) <= tol
        for ob in obs if not ob.mitigated
    )
    ob_at_low = any(
        ob.type == "bull" and abs(ob.high - session_low) <= tol
        for ob in obs if not ob.mitigated
    )

    return {"ob_at_high": ob_at_high, "ob_at_low": ob_at_low}
