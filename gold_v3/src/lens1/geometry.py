"""
London Session Geometry Profiler.
Extracts mathematical 'Footprints' from the London session.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

from src.smc.structure import detect_swing_points

logger = logging.getLogger(__name__)

class LondonSessionProfiler:
    """ Captures mathematical geometry of the London session. """

    def __init__(self, settings):
        self.settings = settings

    def profile_geometry(self, london_candles: pd.DataFrame) -> Dict[str, float]:
        """
        Extracts structural and mathematical features from London candles.
        """
        if len(london_candles) < 5:
            return {}

        prices = london_candles["close"].values
        highs = london_candles["high"].values
        lows = london_candles["low"].values
        volumes = london_candles["volume"].values if "volume" in london_candles.columns else np.ones(len(london_candles))

        # 1. Linear Regression Slope (Trend Strength)
        x = np.arange(len(prices))
        slope, intercept, r_value, p_value, std_err = linregress(x, prices)
        
        # 2. Pivot Density
        swings = detect_swing_points(london_candles, lookback=3)
        pivot_count = len(swings)
        
        # 3. Range Position (Where did it end?)
        session_high = highs.max()
        session_low = lows.min()
        session_range = max(session_high - session_low, 0.01)
        last_close = prices[-1]
        range_pos = (last_close - session_low) / session_range
        
        # 4. Volume Bias (Accumulation/Distribution near extremes)
        # Ratio of volume in top 20% vs bottom 20% of session range
        top_threshold = session_high - 0.20 * session_range
        bot_threshold = session_low + 0.20 * session_range
        
        top_vol = volumes[highs >= top_threshold].sum()
        bot_vol = volumes[lows <= bot_threshold].sum()
        vol_bias = (top_vol - bot_vol) / max(top_vol + bot_vol, 1.0)

        # 5. Momentum Decay / Acceleration
        # Comparison of slope in first half vs second half
        mid_idx = len(prices) // 2
        slope_h1 = linregress(np.arange(mid_idx), prices[:mid_idx]).slope if mid_idx > 2 else slope
        slope_h2 = linregress(np.arange(len(prices)-mid_idx), prices[mid_idx:]).slope if (len(prices)-mid_idx) > 2 else slope
        
        momentum_ratio = slope_h2 / (slope_h1 + 1e-6)

        return {
            "london_slope": float(slope),
            "london_r2": float(r_value**2),
            "london_pivot_density": float(pivot_count / len(london_candles)),
            "london_range_pos": float(range_pos),
            "london_vol_bias": float(vol_bias),
            "london_momentum_ratio": float(np.clip(momentum_ratio, -10, 10)),
            "london_volatility_relative": float(session_range / (london_candles["atr"].mean() if "atr" in london_candles.columns else 1.0)),
        }
