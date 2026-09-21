"""
SMC Geometric Reference Frame.
Translates all raw price distances into London-Range-normalized fractions.
"""

from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.smc.order_blocks import detect_order_blocks, get_nearest_ob
from src.smc.fair_value_gaps import detect_fvgs, get_nearest_fvg
from src.smc.liquidity import map_liquidity_levels, compute_pdh_pdl
from src.smc.structure import detect_swing_points, detect_choch, detect_mss
from src.session.profiler import compute_session_profile

@dataclass
class SmcGeometry:
    """
    The SMC geometric reference frame for one trading day.
    All distances are expressed as fractions of London Range.
    """
    london_high: float
    london_low: float
    london_open: float
    london_close: float
    london_mid: float
    london_range: float
    london_bias: int

    ob_bull_level: float
    ob_bear_level: float
    ob_bull_dist_lr: float
    ob_bear_dist_lr: float
    ob_bull_strength: float
    ob_bear_strength: float

    fvg_bull_top: float
    fvg_bull_bot: float
    fvg_bear_top: float
    fvg_bear_bot: float
    fvg_bull_dist_lr: float
    fvg_bear_dist_lr: float
    fvg_bull_active: bool
    fvg_bear_active: bool

    equal_highs_level: float
    equal_lows_level: float
    equal_highs_present: bool
    equal_lows_present: bool
    equal_highs_dist_lr: float
    equal_lows_dist_lr: float

    pdh: float
    pdl: float
    pdh_dist_lr: float
    pdl_dist_lr: float

    buy_side_liq_above_lh: bool
    sell_side_liq_below_ll: bool
    liq_imbalance: float

    last_swing_direction: int
    choch_bull_present: bool
    choch_bear_present: bool
    mss_bull_present: bool
    mss_bear_present: bool

    ny_open_direction: int
    ny_open_body_lr: float
    ny_open_broke_lh: bool
    ny_open_broke_ll: bool
    ny_open_upper_wick_ratio: float
    ny_open_lower_wick_ratio: float

    atr: float
    london_range_atr_ratio: float

    def normalise(self, price: float) -> float:
        if self.london_range <= 0: return 0.0
        return (price - self.london_mid) / self.london_range

    def dist_lr(self, price: float, level: float) -> float:
        if self.london_range <= 0: return 0.0
        return abs(price - level) / self.london_range


def build_smc_geometry(
    london_candles: pd.DataFrame,
    ny_open_candle: pd.Series,
    full_df: pd.DataFrame,
    smc_config,
    atr_series: pd.Series,
) -> SmcGeometry:
    """Compute the full SMC geometry for a trading day (derived at 12:15 UTC)."""
    profile = compute_session_profile(london_candles)
    london_range = max(profile.range_points, 0.01)
    atr_val = atr_series.iloc[-1] if len(atr_series) > 0 else 1.0

    obs = detect_order_blocks(
        london_candles, atr_series,
        displacement_threshold=smc_config.ob_atr_displacement,
        lookback=min(smc_config.ob_lookback, len(london_candles) - 1),
        max_age=smc_config.ob_max_age,
    )
    bull_ob = get_nearest_ob(obs, profile.close_price, "bull", atr_val)
    bear_ob = get_nearest_ob(obs, profile.close_price, "bear", atr_val)

    # Note: detectors use atr_val only for volatility sizing constraints, not distances
    fvgs = detect_fvgs(london_candles, atr_series, smc_config.fvg_min_size_lr, smc_config.fvg_max_age)
    bull_fvg = get_nearest_fvg(fvgs, profile.close_price, "bull", atr_val)
    bear_fvg = get_nearest_fvg(fvgs, profile.close_price, "bear", atr_val)

    swings = detect_swing_points(london_candles, smc_config.swing_lookback)
    choch = detect_choch(london_candles, swings, smc_config.choch_lookback)
    mss = detect_mss(london_candles, swings, atr_series, smc_config.mss_lookback)

    pdh_pdl = compute_pdh_pdl(full_df, london_candles.index[-1])
    pdh, pdl = pdh_pdl["pdh"], pdh_pdl["pdl"]

    liq = map_liquidity_levels(
        london_candles, profile.high, profile.low, profile.close_price, atr_val,
        smc_config.equal_tolerance_lr,
        smc_config.equal_min_touches,
        sweep_threshold_lr=smc_config.sweep_min_lr,
        pdh=pdh,
        pdl=pdl,
    )

    eq_highs = liq.get("buy_side_levels", [])
    eq_lows  = liq.get("sell_side_levels", [])
    eq_high_level = eq_highs[0].price if eq_highs else 0.0
    eq_low_level  = eq_lows[0].price  if eq_lows  else 0.0

    ny_body = abs(ny_open_candle["close"] - ny_open_candle["open"])
    ny_upper_wick = ny_open_candle["high"] - max(ny_open_candle["open"], ny_open_candle["close"])
    ny_lower_wick = min(ny_open_candle["open"], ny_open_candle["close"]) - ny_open_candle["low"]

    return SmcGeometry(
        london_high=profile.high,
        london_low=profile.low,
        london_open=profile.open_price,
        london_close=profile.close_price,
        london_mid=profile.mid,
        london_range=london_range,
        london_bias=profile.bias,

        ob_bull_level=bull_ob.get("level", 0.0),
        ob_bear_level=bear_ob.get("level", 0.0),
        ob_bull_dist_lr=abs(profile.close_price - bull_ob.get("level", profile.close_price)) / london_range,
        ob_bear_dist_lr=abs(profile.close_price - bear_ob.get("level", profile.close_price)) / london_range,
        ob_bull_strength=bull_ob.get("strength", 0.0) / london_range,
        ob_bear_strength=bear_ob.get("strength", 0.0) / london_range,

        fvg_bull_top=bull_fvg.get("top", 0.0),
        fvg_bull_bot=bull_fvg.get("bot", 0.0),
        fvg_bear_top=bear_fvg.get("top", 0.0),
        fvg_bear_bot=bear_fvg.get("bot", 0.0),
        fvg_bull_dist_lr=bull_fvg.get("distance", london_range * 5) / london_range,
        fvg_bear_dist_lr=bear_fvg.get("distance", london_range * 5) / london_range,
        fvg_bull_active=bull_fvg.get("active", False),
        fvg_bear_active=bear_fvg.get("active", False),

        equal_highs_level=eq_high_level,
        equal_lows_level=eq_low_level,
        equal_highs_present=len(eq_highs) > 0,
        equal_lows_present=len(eq_lows) > 0,
        equal_highs_dist_lr=abs(profile.high - eq_high_level) / london_range if eq_highs else 5.0,
        equal_lows_dist_lr=abs(profile.low - eq_low_level) / london_range if eq_lows else 5.0,

        pdh=pdh,
        pdl=pdl,
        pdh_dist_lr=abs(profile.close_price - pdh) / london_range if pdh > 0 else 5.0,
        pdl_dist_lr=abs(profile.close_price - pdl) / london_range if pdl > 0 else 5.0,

        buy_side_liq_above_lh=liq.get("buy_side_above_lh", False),
        sell_side_liq_below_ll=liq.get("sell_side_below_ll", False),
        liq_imbalance=liq.get("liq_imbalance", 0.0),

        last_swing_direction=swings[-1].type_int if swings else 0,
        choch_bull_present=choch.get("bull_choch", False),
        choch_bear_present=choch.get("bear_choch", False),
        mss_bull_present=mss.get("bull_mss", False),
        mss_bear_present=mss.get("bear_mss", False),

        ny_open_direction=1 if ny_open_candle["close"] > ny_open_candle["open"] else -1,
        ny_open_body_lr=ny_body / london_range,
        ny_open_broke_lh=ny_open_candle["high"] > profile.high,
        ny_open_broke_ll=ny_open_candle["low"] < profile.low,
        ny_open_upper_wick_ratio=ny_upper_wick / max(ny_body, 0.01),
        ny_open_lower_wick_ratio=ny_lower_wick / max(ny_body, 0.01),

        atr=atr_val,
        london_range_atr_ratio=london_range / max(atr_val, 0.01),
    )
