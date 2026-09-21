"""
Historical Scenario Labeler (S1-S8).
Categorizes every trading day based on NY interaction with London levels.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.session.profiler import compute_session_profile, slice_session
from src.smc.liquidity import detect_sweep

logger = logging.getLogger(__name__)

# Scenario IDs
S_UNDEFINED = -1
S1_BULL_DEFEND = 0   # NY respects London Low
S2_BEAR_DEFEND = 1   # NY respects London High
S3_BULL_BREAK = 2    # NY breaks High, retests as support
S4_BEAR_BREAK = 3    # NY breaks Low, retests as resistance
S5_TIGHT_UP = 4      # Tight London, NY breaks High
S6_TIGHT_DOWN = 5    # Tight London, NY breaks Low
S7_SWEEP_HIGH = 6    # NY sweeps High, reverses Bearish
S8_SWEEP_LOW = 7     # NY sweeps Low, reverses Bullish

SCENARIO_NAMES = {
    0: "S1_BullishDefend",
    1: "S2_BearishDefend",
    2: "S3_BullishBreak",
    3: "S4_BearishBreak",
    4: "S5_TightHighBreak",
    5: "S6_TightLowBreak",
    6: "S7_SweepHigh",
    7: "S8_SweepLow",
}

class HistoricalScenarioLabeler:
    """ scans historical days to assign ground-truth scenario IDs. """

    def __init__(self, settings):
        self.settings = settings
        self.sweep_threshold_lr = settings.smc.sweep_min_lr if settings else 0.12
        self.tight_range_atr = 1.2  # Threshold for S5/S6

    def label_day(self, day_df: pd.DataFrame, london_profile, ny_candles: pd.DataFrame) -> int:
        """
        Determines which of the 8 scenarios occurred during NY.
        NY window studied: 12:15 to 21:00 UTC.
        """
        if len(ny_candles) < 5 or london_profile is None:
            return S_UNDEFINED

        lh = float(london_profile.high)
        ll = float(london_profile.low)
        lr = max(float(london_profile.range_points), 0.01)
        
        # ATR reference (London ATR)
        if "atr" in day_df.columns:
            atr_val = day_df.loc[ny_candles.index[0], "atr"] if ny_candles.index[0] in day_df.index else 1.0
        else:
            atr_val = 1.0

        ny_high = ny_candles["high"].max()
        ny_low = ny_candles["low"].min()
        ny_close = ny_candles["close"].iloc[-1]

        # 1. Check for Sweeps (Fakeouts) - Highest Priority
        # S7/S8: crossing threshold then reversing inside
        sweep = detect_sweep(
            current_price=ny_close,
            session_high=lh,
            session_low=ll,
            candles_after=ny_candles,
            threshold_lr=self.sweep_threshold_lr,
            require_reversal=True
        )
        if sweep["sweep_high"]: return S7_SWEEP_HIGH
        if sweep["sweep_low"]: return S8_SWEEP_LOW

        # 2. Check for Breaks & Retests
        # Define "Decisive Break" as close beyond level
        ny_closed_above_high = (ny_candles["close"] > lh).any()
        ny_closed_below_low = (ny_candles["close"] < ll).any()
        
        # S5/S6: Tight Range Breaks (Volatility expansion from low-vel session)
        is_tight = lr < (self.tight_range_atr * atr_val)
        if is_tight:
            if ny_closed_above_high and ny_close > lh: return S5_TIGHT_UP
            if ny_closed_below_low and ny_close < ll: return S6_TIGHT_DOWN

        # S3/S4: Standard Break & Retest
        # Logic: After breaking, the 'retest' must respect the old wall as a new floor/ceiling
        if ny_closed_above_high:
            # Find the first candle that closed above LH
            break_idx = ny_candles[ny_candles["close"] > lh].index[0]
            after_break = ny_candles.loc[break_idx:]
            # Retest check: All lows after the break must be above (LH - small buffer)
            if after_break["low"].min() >= lh - (0.05 * lr):
                return S3_BULL_BREAK
        
        if ny_closed_below_low:
            break_idx = ny_candles[ny_candles["close"] < ll].index[0]
            after_break = ny_candles.loc[break_idx:]
            if after_break["high"].max() <= ll + (0.05 * lr):
                return S4_BEAR_BREAK

        # 3. Check for Defends
        # S1: NY respects London Low (Doesn't even close below it)
        ny_ever_closed_below_low = (ny_candles["close"] < ll).any()
        if not ny_ever_closed_below_low and ny_low >= ll - (0.02 * lr):
            # Bullish rejection: Ended well above Low
            if ny_close > ll + (0.15 * lr):
                return S1_BULL_DEFEND
        
        # S2: NY respects London High
        ny_ever_closed_above_high = (ny_candles["close"] > lh).any()
        if not ny_ever_closed_above_high and ny_high <= lh + (0.02 * lr):
            if ny_close < lh - (0.15 * lr):
                return S2_BEAR_DEFEND

        return S_UNDEFINED

def generate_oracle_labels(df, trading_days, settings):
    """ scans entire dataset and returns mapping {date_str: scenario_id} """
    labeler = HistoricalScenarioLabeler(settings)
    labels = {}
    
    for date in trading_days:
        # Extract London until anchor (12:15) and NY after anchor
        from src.session.profiler import get_session_pair, get_trading_days
        
        london, ny, lp, np_prof = get_session_pair(df, date, settings.sessions if settings else None)
        
        # We need NY *after* anchor (12:15) for labeling
        from src.session.profiler import get_ny_after_anchor
        ny_after_anchor = get_ny_after_anchor(df, date, settings.sessions.lens1_anchor if settings else "12:15")
        
        s_id = labeler.label_day(df, lp, ny_after_anchor)
        labels[str(date.date())] = s_id
        
    return labels
