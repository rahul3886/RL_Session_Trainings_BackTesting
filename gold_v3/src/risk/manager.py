"""
Risk manager — hard filters that cannot be overridden by the RL agent.
These are non-negotiable institutional risk rules.
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RiskCheckResult:
    """Result of a risk check."""
    passed: bool
    reason: str = ""
    sl_price: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp3_price: float = 0.0
    position_size: float = 0.0


def check_trade_risk(
    direction: str,
    entry_price: float,
    sl_price: float,
    tp1_price: float,
    atr_val: float,
    london_range_atr: float,
    ny_minutes_remaining: int,
    trades_today: int,
    settings=None,
) -> RiskCheckResult:
    """
    Run all risk checks on a proposed trade.

    Non-negotiable filters:
      1. No trading in last 60 min of NY
      2. No trading if London > 4 ATR range
      3. SL must be within ATR gate
      4. Minimum R:R must be met
      5. Max trades per day
    """
    max_trades = 3
    min_rr = 1.5
    sl_min_atr_gate = 0.3
    sl_max_lr_gate = 1.5
    london_max_range = 4.0

    if settings is not None:
        cfg = settings.lens2
        max_trades = cfg.max_trades_per_day
        min_rr = cfg.min_rr
        sl_min_atr_gate = cfg.sl_min_atr_gate
        sl_max_lr_gate = cfg.sl_max_lr_gate
        london_max_range = cfg.london_range_atr_max

    # 1. Time gate
    if ny_minutes_remaining <= 60:
        return RiskCheckResult(False, "Time gate: < 60 min remaining in NY")

    # 2. London extension gate
    if london_range_atr > london_max_range:
        return RiskCheckResult(False, f"London over-extended: {london_range_atr:.1f} ATR > {london_max_range}")

    # 3. Max trades gate
    if trades_today >= max_trades:
        return RiskCheckResult(False, f"Max trades reached: {trades_today}/{max_trades}")

    # 4. Dual ATR / LR gate on SL
    sl_dist = abs(entry_price - sl_price)
    
    # ATR gate (volatility floor)
    if sl_dist < sl_min_atr_gate * atr_val:
        return RiskCheckResult(False, f"SL too tight for current volatility: {sl_dist:.2f} < {sl_min_atr_gate * atr_val:.2f}")
        
    # LR gate (structural ceiling sense-check)
    london_range = london_range_atr * atr_val
    if sl_dist > sl_max_lr_gate * london_range:
        return RiskCheckResult(False, f"SL wider than {sl_max_lr_gate} London Ranges — not structural")

    # 5. R:R check
    tp1_dist = abs(tp1_price - entry_price)
    rr = tp1_dist / max(sl_dist, 0.01)
    if rr < min_rr:
        return RiskCheckResult(False, f"R:R too low: {rr:.2f} < {min_rr}")

    return RiskCheckResult(True, "All risk checks passed")


def compute_structural_sl(
    direction: str,
    entry_price: float,
    geometry,
    buffer_lr: float = 0.10,
) -> float:
    """
    Compute structural stop loss based on order block levels or London boundaries.
    Buffer expressed as fraction of London Range.
    """
    buffer = buffer_lr * geometry.london_range

    if direction == "long":
        if geometry.ob_bull_level > 0:
            return geometry.ob_bull_level - buffer
        return geometry.london_low - buffer
    else:
        if geometry.ob_bear_level > 0:
            return geometry.ob_bear_level + buffer
        return geometry.london_high + buffer


def compute_structural_tp(
    direction: str,
    entry_price: float,
    london_profile,
    pdh: float = 0.0,
    pdl: float = 0.0,
) -> Dict[str, float]:
    """
    Compute waterfall TP levels.

    TP1 (50%) = LondonMid
    TP2 (30%) = opposite London level
    TP3 (20%) = PDH/PDL
    """
    if london_profile is None:
        return {"tp1": entry_price, "tp2": entry_price, "tp3": entry_price}

    if direction == "long":
        tp1 = london_profile.mid if london_profile.mid > entry_price else london_profile.high
        tp2 = london_profile.high if tp1 == london_profile.mid else pdh if pdh > 0 else london_profile.high * 1.001
        tp3 = pdh if pdh > tp2 else tp2 * 1.001
    else:
        tp1 = london_profile.mid if london_profile.mid < entry_price else london_profile.low
        tp2 = london_profile.low if tp1 == london_profile.mid else pdl if pdl > 0 else london_profile.low * 0.999
        tp3 = pdl if pdl < tp2 else tp2 * 0.999

    return {"tp1": tp1, "tp2": tp2, "tp3": tp3}
