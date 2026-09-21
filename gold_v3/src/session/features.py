"""
Session feature extractor — shared foundation for both lenses.
Runs all SMC detectors on a candle window and returns
a standardised raw feature dictionary.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
import numpy as np

from src.smc.order_blocks import detect_order_blocks, get_nearest_ob, ob_at_session_level
from src.smc.fair_value_gaps import detect_fvgs, get_nearest_fvg, total_fvg_imbalance
from src.smc.liquidity import map_liquidity_levels, compute_pdh_pdl
from src.smc.structure import detect_swing_points, detect_choch, detect_mss, get_structure_bias
from src.session.profiler import compute_session_profile

logger = logging.getLogger(__name__)


def extract_session_features(
    candles: pd.DataFrame,
    atr_series: pd.Series,
    full_df: pd.DataFrame = None,
    session_high: float = None,
    session_low: float = None,
    smc_config=None,
    pdh_pdl_cache: Optional[Dict] = None,
) -> Dict:
    """
    Run all SMC detectors on a candle window.
    Returns raw (unnormalised) feature dict.
    Used by Lens 1 for London candles and
    by Lens 2 for NY candle-by-candle state.

    Args:
        candles: OHLCV candles for the session window
        atr_series: ATR values aligned to candles
        full_df: full dataset for PDH/PDL computation
        session_high: session high (can be pre-computed)
        session_low: session low (can be pre-computed)
        smc_config: SMC configuration parameters

    Returns:
        dict with all SMC features
    """
    if len(candles) == 0:
        return _empty_features()

    # Configuration defaults
    ob_displacement = 1.5
    ob_lookback = 20
    ob_max_age = 50
    fvg_min_size = 0.05
    fvg_max_age = 30
    eq_tolerance = 0.05
    eq_min_touches = 2
    sweep_threshold_lr = 0.12
    swing_lookback = 5

    if smc_config is not None:
        ob_displacement = smc_config.ob_atr_displacement
        ob_lookback = smc_config.ob_lookback
        ob_max_age = smc_config.ob_max_age
        fvg_min_size = smc_config.fvg_min_size_lr
        fvg_max_age = smc_config.fvg_max_age
        eq_tolerance = smc_config.equal_tolerance_lr
        eq_min_touches = smc_config.equal_min_touches
        sweep_threshold_lr = smc_config.sweep_min_lr
        swing_lookback = smc_config.swing_lookback

    # Current state
    current_price = candles["close"].iloc[-1]
    atr_val = atr_series.iloc[-1] if len(atr_series) > 0 and atr_series.iloc[-1] > 0 else 1.0

    if session_high is None:
        session_high = candles["high"].max()
    if session_low is None:
        session_low = candles["low"].min()

    session_mid = (session_high + session_low) / 2.0
    session_open = candles["open"].iloc[0]
    session_close = candles["close"].iloc[-1]
    session_range = max(session_high - session_low, 0.01)

    # ── Session Profile ──
    profile = compute_session_profile(candles)

    # ── Order Blocks ──
    obs = detect_order_blocks(
        candles, atr_series,
        displacement_threshold=ob_displacement,
        lookback=min(ob_lookback, len(candles) - 1),
        max_age=ob_max_age,
    )
    bull_ob = get_nearest_ob(obs, current_price, "bull", atr_val)
    bear_ob = get_nearest_ob(obs, current_price, "bear", atr_val)
    ob_levels = ob_at_session_level(obs, session_high, session_low, atr_val)

    # ── Fair Value Gaps ──
    fvgs = detect_fvgs(candles, atr_series, fvg_min_size, fvg_max_age)
    bull_fvg = get_nearest_fvg(fvgs, current_price, "bull", atr_val)
    bear_fvg = get_nearest_fvg(fvgs, current_price, "bear", atr_val)
    fvg_imbalance = total_fvg_imbalance(fvgs, session_range)  # Switch to LR base

    # ── Swing Points & Structure ──
    swings = detect_swing_points(candles, swing_lookback)
    choch = detect_choch(candles, swings, swing_lookback)
    mss = detect_mss(candles, swings, atr_series, swing_lookback)

    # ── Liquidity ──
    pdh, pdl = 0.0, 0.0
    if full_df is not None and len(candles) > 0:
        pdh_pdl = compute_pdh_pdl(full_df, candles.index[0], pdh_pdl_cache=pdh_pdl_cache)
        pdh = pdh_pdl["pdh"]
        pdl = pdh_pdl["pdl"]

    liq = map_liquidity_levels(
        candles, session_high, session_low, current_price, atr_val,
        eq_tolerance, eq_min_touches, sweep_threshold_lr, pdh, pdl,
    )

    atr_safe = max(atr_val, 0.01)

    return {
        "profile": profile,
        "order_blocks": obs,
        "fvgs": fvgs,
        "swings": swings,
        "choch": choch,
        "mss": mss,
        "liquidity": liq,
        # Pre-computed state vector components
        "state": {
            # Session state (7)
            "session_bias": float(profile.bias),
            "session_range_atr": session_range / atr_safe,  # This tells the network how big LR is vs ATR
            "session_body_ratio": profile.body_ratio,
            "dist_to_session_high": min(abs(session_high - current_price) / session_range, 3.0),
            "dist_to_session_low": min(abs(current_price - session_low) / session_range, 3.0),
            "dist_to_session_mid": min(abs(current_price - session_mid) / session_range, 3.0),
            "time_elapsed_norm": 0.0,  # set by caller

            # Order blocks (6)
            "bull_ob_dist": min(abs(current_price - bull_ob.get("level", current_price)) / session_range, 3.0),
            "bull_ob_strength": bull_ob.get("strength", 0.0) / session_range,
            "bear_ob_dist": min(abs(current_price - bear_ob.get("level", current_price)) / session_range, 3.0),
            "bear_ob_strength": bear_ob.get("strength", 0.0) / session_range,
            "ob_at_high_flag": 1.0 if ob_levels["ob_at_high"] else 0.0,
            "ob_at_low_flag": 1.0 if ob_levels["ob_at_low"] else 0.0,

            # Fair value gaps (5)
            "bull_fvg_unfilled": 1.0 if bull_fvg.get("active", False) else 0.0,
            "bull_fvg_dist": min(bull_fvg.get("distance", session_range * 3) / session_range, 3.0),
            "bear_fvg_unfilled": 1.0 if bear_fvg.get("active", False) else 0.0,
            "bear_fvg_dist": min(bear_fvg.get("distance", session_range * 3) / session_range, 3.0),
            "fvg_total_size_atr": min(fvg_imbalance / session_range, 3.0),

            # Liquidity (6)
            "equal_highs_present": liq.get("equal_highs_present", 1.0 if liq.get("equal_highs") else 0.0),
            "equal_lows_present": liq.get("equal_lows_present", 1.0 if liq.get("equal_lows") else 0.0),
            "liq_above_density": liq.get("liq_above_density", 0.0),
            "liq_below_density": liq.get("liq_below_density", 0.0),
            "pdh_dist": min(abs(current_price - pdh) / session_range if pdh > 0 else 3.0, 3.0),
            "pdl_dist": min(abs(current_price - pdl) / session_range if pdl > 0 else 3.0, 3.0),

            # Structure (4)
            "choch_bull_recent": 1.0 if choch["bull_choch"] else 0.0,
            "choch_bear_recent": 1.0 if choch["bear_choch"] else 0.0,
            "mss_bull_recent": 1.0 if mss["bull_mss"] else 0.0,
            "mss_bear_recent": 1.0 if mss["bear_mss"] else 0.0,

            # Candle anatomy (3)
            "current_body_atr": candles["candle_body_atr"].iloc[-1] if "candle_body_atr" in candles.columns else 0.0,
            "current_upper_wick_atr": candles["upper_wick_atr"].iloc[-1] if "upper_wick_atr" in candles.columns else 0.0,
            "current_lower_wick_atr": candles["lower_wick_atr"].iloc[-1] if "lower_wick_atr" in candles.columns else 0.0,
            
            # --- Phase 8: Institutional Quant Confirmations (4) ---
            "vwap_zscore": candles["vwap_zscore"].iloc[-1] if "vwap_zscore" in candles.columns else 0.0,
            "momentum_r2": candles["momentum_r2"].iloc[-1] if "momentum_r2" in candles.columns else 0.0,
            "volatility_compression": candles["volatility_compression"].iloc[-1] if "volatility_compression" in candles.columns else 0.0,
            "displacement": candles["displacement"].iloc[-1] if "displacement" in candles.columns else 0.0,
        },
        # Raw values for SL/TP computation
        "raw": {
            "current_price": current_price,
            "atr": atr_val,
            "session_high": session_high,
            "session_low": session_low,
            "session_mid": session_mid,
            "session_open": session_open,
            "pdh": pdh,
            "pdl": pdl,
            "bull_ob_level": bull_ob["level"],
            "bear_ob_level": bear_ob["level"],
            "sweep_high": liq["sweep_high"],
            "sweep_low": liq["sweep_low"],
        }
    }


def build_lens1_state_vector(features: Dict, time_norm: float = 0.0) -> np.ndarray:
    """
    Build the 31-dim observation vector for Lens 1.
    """
    s = features["state"]
    s["time_elapsed_norm"] = time_norm

    return np.array([
        # Session state (7)
        s["session_bias"],
        s["session_range_atr"],
        s["session_body_ratio"],
        s["dist_to_session_high"],
        s["dist_to_session_low"],
        s["dist_to_session_mid"],
        s["time_elapsed_norm"],
        # Order blocks (6)
        s["bull_ob_dist"],
        s["bull_ob_strength"],
        s["bear_ob_dist"],
        s["bear_ob_strength"],
        s["ob_at_high_flag"],
        s["ob_at_low_flag"],
        # Fair value gaps (5)
        s["bull_fvg_unfilled"],
        s["bull_fvg_dist"],
        s["bear_fvg_unfilled"],
        s["bear_fvg_dist"],
        s["fvg_total_size_atr"],
        # Liquidity (6)
        s["equal_highs_present"],
        s["equal_lows_present"],
        s["liq_above_density"],
        s["liq_below_density"],
        s["pdh_dist"],
        s["pdl_dist"],
        # Structure (4)
        s["choch_bull_recent"],
        s["choch_bear_recent"],
        s["mss_bull_recent"],
        s["mss_bear_recent"],
        # Candle anatomy (3)
        s["current_body_atr"],
        s["current_upper_wick_atr"],
        s["current_lower_wick_atr"],
    ], dtype=np.float32)


def build_lens3_state_vector(
    lens1_encoding: np.ndarray,
    ny_features: Dict,
    london_profile=None,
    ny_time_norm: float = 0.0,
    in_ny_open: float = 0.0,
    momentum: float = 0.0,
    volume_ratio: float = 1.0,
    ny_broke_lh: float = 0.0,
    ny_broke_ll: float = 0.0,
    distance_to_nearest_ob_atr: float = 5.0,
    # Phase 8 Quant Overrides
    vwap_zscore: float = 0.0,
    momentum_r2: float = 0.0,
    volatility_compression: float = 1.0,
    displacement: float = 0.0,
    # Phase 9 Regime Overrides
    regime_state: float = 0.0,
    cpd_age: float = -1.0,
    # Phase 10 Fractal Sync
    hmm_velocity: float = 0.0,
    fractal_sync: float = 0.0,
) -> np.ndarray:
    """
    Build the 40-dim observation vector for Lens 2.
    = 8 (Lens 1 encoding) + 32 (live NY features)
    """
    s = ny_features["state"]

    # London context from Lens 1
    london_bias = 0.0
    london_range_atr = 0.0
    dist_to_lh = 0.0
    dist_to_ll = 0.0
    dist_to_lm = 0.0
    if london_profile is not None:
        atr_safe = max(ny_features["raw"]["atr"], 0.01)
        london_bias = float(london_profile.bias)
        london_range = max(london_profile.range_points, 0.01)
        london_range_atr = london_range / atr_safe
        cp = ny_features["raw"]["current_price"]
        dist_to_lh = min(abs(london_profile.high - cp) / london_range, 3.0)
        dist_to_ll = min(abs(cp - london_profile.low) / london_range, 3.0)
        dist_to_lm = min(abs(cp - london_profile.mid) / london_range, 3.0)

    ny_state = np.array([
        # London context via raw features (5)
        london_bias,
        london_range_atr,
        dist_to_lh,
        dist_to_ll,
        dist_to_lm,
        # OB features (4)
        s["bull_ob_dist"],   # ob_bull_active implied by dist < threshold
        s["bear_ob_dist"],
        s["bull_ob_strength"],
        s["bear_ob_strength"],
        # FVG features (4)
        s["bull_fvg_unfilled"],
        s["bear_fvg_unfilled"],
        s["bull_fvg_dist"],
        s["bear_fvg_dist"],
        # Structure (4)
        s["choch_bull_recent"],
        s["choch_bear_recent"],
        s["mss_bull_recent"],
        s["mss_bear_recent"],
        # NY broke London levels (2)
        ny_broke_lh,
        ny_broke_ll,
        # Sweep (2)
        ny_features["raw"].get("sweep_high", 0.0),
        ny_features["raw"].get("sweep_low", 0.0),
        # PDH/PDL (2)
        s["pdh_dist"],
        s["pdl_dist"],
        # Time features (2)
        ny_time_norm,
        in_ny_open,
        # Candle features (3)
        s["current_body_atr"],
        s["current_upper_wick_atr"],
        s["current_lower_wick_atr"],
        # Momentum & volume (2)
        momentum,
        volume_ratio,
        # Distance to nearest OB (1)
        distance_to_nearest_ob_atr,
        # Liquidity density (1)
        s["liq_above_density"],
        # Phase 8: Institutional Quant Filters (4)
        vwap_zscore,
        momentum_r2,
        volatility_compression,
        displacement,
        # Phase 9: Regime State & CPD Breaks (2)
        regime_state,
        cpd_age,
        # Phase 10: Fractal Edge (2)
        hmm_velocity,
        fractal_sync,
    ], dtype=np.float32)

    return np.concatenate([lens1_encoding.astype(np.float32), ny_state])


def _empty_features() -> Dict:
    """Return empty feature dict when no candles available."""
    return {
        "profile": None,
        "order_blocks": [],
        "fvgs": [],
        "swings": [],
        "choch": {"bull_choch": False, "bear_choch": False},
        "mss": {"bull_mss": False, "bear_mss": False},
        "liquidity": {
            "equal_highs_present": 0.0, "equal_lows_present": 0.0,
            "liq_above_density": 0.0, "liq_below_density": 0.0,
            "pdh_dist": 0.0, "pdl_dist": 0.0,
            "sweep_high": 0.0, "sweep_low": 0.0,
            "buy_side_levels": [], "sell_side_levels": [],
        },
        "state": {k: 0.0 for k in [
            "session_bias", "session_range_atr", "session_body_ratio",
            "dist_to_session_high", "dist_to_session_low", "dist_to_session_mid",
            "time_elapsed_norm",
            "bull_ob_dist", "bull_ob_strength", "bear_ob_dist", "bear_ob_strength",
            "ob_at_high_flag", "ob_at_low_flag",
            "bull_fvg_unfilled", "bull_fvg_dist",
            "bear_fvg_unfilled", "bear_fvg_dist", "fvg_total_size_atr",
            "equal_highs_present", "equal_lows_present",
            "liq_above_density", "liq_below_density",
            "pdh_dist", "pdl_dist",
            "choch_bull_recent", "choch_bear_recent",
            "mss_bull_recent", "mss_bear_recent",
            "current_body_atr", "current_upper_wick_atr", "current_lower_wick_atr",
        ]},
        "raw": {
            "current_price": 0.0, "atr": 1.0,
            "session_high": 0.0, "session_low": 0.0,
            "session_mid": 0.0, "session_open": 0.0,
            "pdh": 0.0, "pdl": 0.0,
            "bull_ob_level": 0.0, "bear_ob_level": 0.0,
            "sweep_high": 0.0, "sweep_low": 0.0,
        },
    }
