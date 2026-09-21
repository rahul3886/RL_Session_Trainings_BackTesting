"""
Market structure detection — Swing Points, CHoCH, MSS.
Identifies the backbone of institutional price delivery.
"""

import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SwingPoint:
    """A detected swing high or swing low."""
    type: str          # "high" or "low"
    price: float
    index: int         # position in candle array
    broken: bool = False


def detect_swing_points(
    candles: pd.DataFrame,
    lookback: int = 5,
) -> List[SwingPoint]:
    """
    Detect swing highs and swing lows using lookback comparison.

    A swing high: highest point in window of 2*lookback+1 candles
    A swing low: lowest point in window of 2*lookback+1 candles

    Args:
        candles: OHLCV DataFrame
        lookback: candles on each side to compare

    Returns:
        List of SwingPoint objects ordered by index
    """
    swings: List[SwingPoint] = []

    if len(candles) < 2 * lookback + 1:
        return swings

    highs = candles["high"].values
    lows = candles["low"].values

    for i in range(lookback, len(candles) - lookback):
        # Swing High: current high > all surrounding highs
        window_highs = highs[i - lookback: i + lookback + 1]
        if highs[i] == window_highs.max() and highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
            swings.append(SwingPoint(type="high", price=highs[i], index=i))

        # Swing Low: current low < all surrounding lows
        window_lows = lows[i - lookback: i + lookback + 1]
        if lows[i] == window_lows.min() and lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
            swings.append(SwingPoint(type="low", price=lows[i], index=i))

    return swings


def detect_choch(
    candles: pd.DataFrame,
    swings: List[SwingPoint] = None,
    lookback: int = 5,
) -> Dict[str, bool]:
    """
    Detect Change of Character (CHoCH).

    Bullish CHoCH: In a downtrend, price breaks above the most recent
                   swing high → character changes from bearish to bullish.

    Bearish CHoCH: In an uptrend, price breaks below the most recent
                   swing low → character changes from bullish to bearish.

    Returns:
        dict with 'bull_choch' and 'bear_choch' recent flags
    """
    result = {"bull_choch": False, "bear_choch": False}

    if swings is None:
        swings = detect_swing_points(candles, lookback)

    if len(swings) < 3 or len(candles) < 3:
        return result

    current_price = candles["close"].iloc[-1]

    # Get recent swing highs and lows
    recent_highs = [s for s in swings if s.type == "high"]
    recent_lows = [s for s in swings if s.type == "low"]

    if len(recent_highs) >= 2 and len(recent_lows) >= 1:
        # Check for bullish CHoCH: lower lows → break above last swing high
        last_sh = recent_highs[-1]

        # Was there a downtrend? (lower highs pattern)
        if (len(recent_highs) >= 2 and
            recent_highs[-1].price < recent_highs[-2].price):
            # Break above the most recent swing high = CHoCH
            if current_price > last_sh.price:
                # Only count as recent if within last N candles
                if len(candles) - 1 - last_sh.index <= lookback * 2:
                    result["bull_choch"] = True

    if len(recent_lows) >= 2 and len(recent_highs) >= 1:
        # Check for bearish CHoCH: higher highs → break below last swing low
        last_sl = recent_lows[-1]

        # Was there an uptrend? (higher lows pattern)
        if (len(recent_lows) >= 2 and
            recent_lows[-1].price > recent_lows[-2].price):
            # Break below the most recent swing low = CHoCH
            if current_price < last_sl.price:
                if len(candles) - 1 - last_sl.index <= lookback * 2:
                    result["bear_choch"] = True

    return result


def detect_mss(
    candles: pd.DataFrame,
    swings: List[SwingPoint] = None,
    atr: pd.Series = None,
    lookback: int = 5,
) -> Dict[str, bool]:
    """
    Detect Market Structure Shift (MSS).

    MSS is similar to CHoCH but confirmed with displacement (strong move).
    It's a CHoCH + momentum confirmation.

    Bullish MSS: price breaks above swing high with strong bullish candle
    Bearish MSS: price breaks below swing low with strong bearish candle

    Returns:
        dict with 'bull_mss' and 'bear_mss' recent flags
    """
    result = {"bull_mss": False, "bear_mss": False}

    if swings is None:
        swings = detect_swing_points(candles, lookback)

    if len(swings) < 2 or len(candles) < 3:
        return result

    current_close = candles["close"].iloc[-1]
    current_open = candles["open"].iloc[-1]
    current_body = abs(current_close - current_open)

    atr_val = 1.0
    if atr is not None and len(atr) > 0:
        atr_val = atr.iloc[-1] if atr.iloc[-1] > 0 else 1.0

    # Displacement threshold: body > 0.5 ATR
    has_displacement = current_body > 0.5 * atr_val

    if not has_displacement:
        return result

    recent_highs = [s for s in swings if s.type == "high"]
    recent_lows = [s for s in swings if s.type == "low"]

    # Bullish MSS: break above swing high with bullish displacement
    if recent_highs and current_close > current_open:  # bullish candle
        last_sh = recent_highs[-1]
        if current_close > last_sh.price:
            if len(candles) - 1 - last_sh.index <= lookback * 3:
                result["bull_mss"] = True

    # Bearish MSS: break below swing low with bearish displacement
    if recent_lows and current_close < current_open:  # bearish candle
        last_sl = recent_lows[-1]
        if current_close < last_sl.price:
            if len(candles) - 1 - last_sl.index <= lookback * 3:
                result["bear_mss"] = True

    return result


def get_structure_bias(swings: List[SwingPoint]) -> int:
    """
    Determine overall structure bias from swing points.

    Returns:
        +1 (bullish: higher highs + higher lows)
        -1 (bearish: lower highs + lower lows)
         0 (ranging / unclear)
    """
    if len(swings) < 4:
        return 0

    # Get last few swing highs and lows
    recent_highs = [s for s in swings if s.type == "high"][-3:]
    recent_lows = [s for s in swings if s.type == "low"][-3:]

    if len(recent_highs) < 2 or len(recent_lows) < 2:
        return 0

    # Higher highs + higher lows = bullish
    hh = recent_highs[-1].price > recent_highs[-2].price
    hl = recent_lows[-1].price > recent_lows[-2].price

    # Lower highs + lower lows = bearish
    lh = recent_highs[-1].price < recent_highs[-2].price
    ll = recent_lows[-1].price < recent_lows[-2].price

    if hh and hl:
        return 1
    elif lh and ll:
        return -1
    else:
        return 0
