"""
Test suite for SMC detectors.
Tests order blocks, FVGs, liquidity, structure.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


def _make_candles(n=20, base_price=2000.0, trend="up"):
    """Create synthetic OHLCV data for testing."""
    dates = pd.date_range("2024-01-15 08:00", periods=n, freq="15min", tz="UTC")
    data = []
    price = base_price
    for i in range(n):
        if trend == "up":
            o = price
            c = price + np.random.uniform(0.5, 3.0)
            h = max(o, c) + np.random.uniform(0.1, 1.5)
            l = min(o, c) - np.random.uniform(0.1, 1.0)
            price = c
        elif trend == "down":
            o = price
            c = price - np.random.uniform(0.5, 3.0)
            h = max(o, c) + np.random.uniform(0.1, 1.0)
            l = min(o, c) - np.random.uniform(0.1, 1.5)
            price = c
        else:
            o = price
            c = price + np.random.uniform(-1.0, 1.0)
            h = max(o, c) + np.random.uniform(0.1, 0.5)
            l = min(o, c) - np.random.uniform(0.1, 0.5)
            price = c

        data.append({"open": o, "high": h, "low": l, "close": c, "volume": 100 + i * 10})

    df = pd.DataFrame(data, index=dates)
    return df


def _make_atr(df, value=5.0):
    """Create constant ATR series for testing."""
    return pd.Series(value, index=df.index)


def _make_sweep_window(direction="high", reversed_back=True):
    """Create a deterministic NY window with an optional reversal-confirmed sweep."""
    dates = pd.date_range("2024-01-15 12:15", periods=4, freq="15min", tz="UTC")

    if direction == "high":
        closes = [99.2, 99.8, 102.6, 99.7 if reversed_back else 102.4]
        data = [
            {"open": 99.0, "high": 99.6, "low": 98.7, "close": closes[0], "volume": 100},
            {"open": 99.2, "high": 100.0, "low": 99.0, "close": closes[1], "volume": 110},
            {"open": 99.8, "high": 103.2, "low": 99.6, "close": closes[2], "volume": 120},
            {"open": 102.6, "high": 102.8, "low": 99.1, "close": closes[3], "volume": 130},
        ]
    else:
        closes = [90.8, 90.2, 87.8, 90.3 if reversed_back else 88.4]
        data = [
            {"open": 91.0, "high": 91.3, "low": 90.4, "close": closes[0], "volume": 100},
            {"open": 90.8, "high": 91.0, "low": 90.0, "close": closes[1], "volume": 110},
            {"open": 90.2, "high": 90.4, "low": 87.6, "close": closes[2], "volume": 120},
            {"open": 87.8, "high": 90.5, "low": 87.7, "close": closes[3], "volume": 130},
        ]

    return pd.DataFrame(data, index=dates)


class TestOrderBlocks:
    def test_detection_returns_list(self):
        from src.smc.order_blocks import detect_order_blocks
        candles = _make_candles(20)
        atr = _make_atr(candles)
        obs = detect_order_blocks(candles, atr)
        assert isinstance(obs, list)

    def test_ob_has_required_fields(self):
        from src.smc.order_blocks import detect_order_blocks
        candles = _make_candles(20)
        atr = _make_atr(candles, 1.0)  # small ATR to trigger more OBs
        obs = detect_order_blocks(candles, atr, displacement_threshold=0.5)
        if len(obs) > 0:
            ob = obs[0]
            assert hasattr(ob, 'type')
            assert hasattr(ob, 'high')
            assert hasattr(ob, 'low')
            assert hasattr(ob, 'strength')
            assert ob.type in ("bull", "bear")

    def test_nearest_ob(self):
        from src.smc.order_blocks import detect_order_blocks, get_nearest_ob
        candles = _make_candles(20)
        atr = _make_atr(candles)
        obs = detect_order_blocks(candles, atr)
        result = get_nearest_ob(obs, 2010.0, "bull", 5.0)
        assert "distance_atr" in result
        assert "strength" in result


class TestFairValueGaps:
    def test_fvg_detection_returns_list(self):
        from src.smc.fair_value_gaps import detect_fvgs
        candles = _make_candles(20)
        atr = _make_atr(candles)
        fvgs = detect_fvgs(candles, atr)
        assert isinstance(fvgs, list)

    def test_fvg_has_type(self):
        from src.smc.fair_value_gaps import detect_fvgs
        candles = _make_candles(20)
        atr = _make_atr(candles, 0.5)  # small ATR
        fvgs = detect_fvgs(candles, atr, min_size_atr=0.01)
        if len(fvgs) > 0:
            assert fvgs[0].type in ("bull", "bear")
            assert fvgs[0].high >= fvgs[0].low

    def test_total_imbalance(self):
        from src.smc.fair_value_gaps import detect_fvgs, total_fvg_imbalance
        candles = _make_candles(20)
        atr = _make_atr(candles)
        fvgs = detect_fvgs(candles, atr)
        imb = total_fvg_imbalance(fvgs, 5.0)
        assert isinstance(imb, float)
        assert imb >= 0


class TestLiquidity:
    def test_equal_levels(self):
        from src.smc.liquidity import detect_equal_levels
        candles = _make_candles(20, trend="range")
        buy_side, sell_side = detect_equal_levels(candles, atr_val=5.0)
        assert isinstance(buy_side, list)
        assert isinstance(sell_side, list)

    def test_sweep_detection(self):
        from src.smc.liquidity import detect_sweep
        candles = _make_sweep_window(direction="high", reversed_back=True)
        result = detect_sweep(
            current_price=float(candles["close"].iloc[-1]),
            session_high=100.0,
            session_low=90.0,
            candles_after=candles,
            threshold_lr=0.2,
        )
        assert result["sweep_high"] is True
        assert result["sweep_low"] is False

    def test_sweep_detection_requires_reversal(self):
        from src.smc.liquidity import detect_sweep
        candles = _make_sweep_window(direction="high", reversed_back=False)
        result = detect_sweep(
            current_price=float(candles["close"].iloc[-1]),
            session_high=100.0,
            session_low=90.0,
            candles_after=candles,
            threshold_lr=0.2,
        )
        assert result["sweep_high"] is False
        assert result["sweep_low"] is False

    def test_map_liquidity_levels_uses_range_based_sweep_threshold(self):
        from src.smc.liquidity import map_liquidity_levels
        candles = _make_sweep_window(direction="low", reversed_back=True)
        result = map_liquidity_levels(
            candles=candles,
            session_high=100.0,
            session_low=90.0,
            current_price=float(candles["close"].iloc[-1]),
            atr_val=1.0,
            sweep_threshold_lr=0.2,
        )
        assert result["sweep_high"] == 0.0
        assert result["sweep_low"] == 1.0

    def test_pdh_pdl(self):
        from src.smc.liquidity import compute_pdh_pdl
        dates = pd.date_range("2024-01-14 08:00", periods=40, freq="15min", tz="UTC")
        df = pd.DataFrame({
            "open": np.random.uniform(1990, 2010, 40),
            "high": np.random.uniform(2005, 2015, 40),
            "low": np.random.uniform(1985, 1995, 40),
            "close": np.random.uniform(1990, 2010, 40),
        }, index=dates)
        result = compute_pdh_pdl(df, pd.Timestamp("2024-01-15", tz="UTC"))
        # May not have prev day data in this small sample
        assert "pdh" in result
        assert "pdl" in result

    def test_pdh_pdl_cache_matches_direct_lookup(self):
        from src.smc.liquidity import build_pdh_pdl_cache, compute_pdh_pdl
        dates = pd.date_range("2024-01-14 08:00", periods=200, freq="15min", tz="UTC")
        df = pd.DataFrame({
            "open": np.linspace(1990, 2010, len(dates)),
            "high": np.linspace(2000, 2020, len(dates)),
            "low": np.linspace(1980, 2000, len(dates)),
            "close": np.linspace(1995, 2015, len(dates)),
        }, index=dates)
        cache = build_pdh_pdl_cache(df)
        current_date = pd.Timestamp("2024-01-15 12:00", tz="UTC")
        direct = compute_pdh_pdl(df, current_date)
        cached = compute_pdh_pdl(df, current_date, pdh_pdl_cache=cache)
        assert cached == direct


class TestStructure:
    def test_swing_detection(self):
        from src.smc.structure import detect_swing_points
        candles = _make_candles(30)
        swings = detect_swing_points(candles, lookback=3)
        assert isinstance(swings, list)

    def test_choch(self):
        from src.smc.structure import detect_choch
        candles = _make_candles(30)
        result = detect_choch(candles)
        assert "bull_choch" in result
        assert "bear_choch" in result

    def test_mss(self):
        from src.smc.structure import detect_mss
        candles = _make_candles(30)
        atr = _make_atr(candles)
        result = detect_mss(candles, atr=atr)
        assert "bull_mss" in result
        assert "bear_mss" in result

    def test_structure_bias(self):
        from src.smc.structure import detect_swing_points, get_structure_bias
        candles = _make_candles(30, trend="up")
        swings = detect_swing_points(candles, lookback=3)
        bias = get_structure_bias(swings)
        assert bias in (-1, 0, 1)


class TestConfig:
    def test_load_settings(self):
        from src.config import load_settings
        settings = load_settings()
        assert settings.lens1.obs_dim == 31
        assert settings.lens1.encoding_dim == 8
        assert settings.lens2.obs_dim == 40
        assert settings.lens2.max_trades_per_day == 3

    def test_session_times(self):
        from src.config import load_settings
        settings = load_settings()
        assert settings.sessions.london_start == "08:00"
        assert settings.sessions.ny_end == "21:00"
        assert settings.sessions.lens1_anchor == "12:15"
