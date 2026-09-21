"""
Tests for Lens 1 environment.
"""

import pytest
import numpy as np
import pandas as pd


def _make_test_data():
    """Create minimal test data for Lens 1 env."""
    # 2 days of 15M data covering London + NY (08:00-21:00 UTC)
    dates = []
    for day in ["2024-01-15", "2024-01-16"]:
        for hour in range(8, 22):
            for minute in [0, 15, 30, 45]:
                dates.append(f"{day} {hour:02d}:{minute:02d}")

    dates = pd.to_datetime(dates, utc=True)
    n = len(dates)
    base = 2000.0
    prices = base + np.cumsum(np.random.randn(n) * 0.5)

    df = pd.DataFrame({
        "open": prices,
        "high": prices + np.abs(np.random.randn(n)) * 1.5,
        "low": prices - np.abs(np.random.randn(n)) * 1.5,
        "close": prices + np.random.randn(n) * 0.3,
        "volume": np.random.randint(50, 200, n).astype(float),
    }, index=dates)

    # Add indicators
    from src.data.indicators import add_indicators
    df = add_indicators(df)

    return df


def _make_ny_high_sweep(reversed_back=True):
    """Create a deterministic NY window with a high-side liquidity sweep."""
    dates = pd.date_range("2024-01-15 12:15", periods=4, freq="15min", tz="UTC")
    closes = [99.2, 99.8, 102.6, 99.7 if reversed_back else 102.4]

    return pd.DataFrame({
        "open": [99.0, 99.2, 99.8, 102.6],
        "high": [99.6, 100.0, 103.2, 102.8],
        "low": [98.7, 99.0, 99.6, 99.1],
        "close": closes,
        "volume": [100.0, 110.0, 120.0, 130.0],
    }, index=dates)


def _make_ny_low_sweep(reversed_back=True):
    dates = pd.date_range("2024-01-15 12:15", periods=4, freq="15min", tz="UTC")
    closes = [100.8, 100.4, 88.7, 91.5 if reversed_back else 88.9]

    return pd.DataFrame({
        "open": [101.0, 100.8, 100.4, 88.7],
        "high": [101.2, 101.0, 100.6, 92.1],
        "low": [100.4, 100.0, 87.9, 88.1],
        "close": closes,
        "volume": [100.0, 110.0, 120.0, 130.0],
    }, index=dates)


class TestLens1Env:
    def test_creation(self):
        from src.lens1.environment import Lens1LondonEnv
        from src.config import load_settings
        from src.session.profiler import get_trading_days

        settings = load_settings()
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens1LondonEnv(df, days, settings)
            assert env.observation_space.shape == (31,)
            assert env.action_space.shape == (8,)

    def test_seed_compatibility(self):
        from src.lens1.environment import Lens1LondonEnv
        from src.config import load_settings
        from src.session.profiler import get_trading_days

        settings = load_settings()
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens1LondonEnv(df, days, settings)
            seeds = env.seed(123)
            assert isinstance(seeds, list)
            assert len(seeds) == 1
            assert isinstance(seeds[0], int)

    def test_reset_returns_obs(self):
        from src.lens1.environment import Lens1LondonEnv
        from src.config import load_settings
        from src.session.profiler import get_trading_days

        settings = load_settings()
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens1LondonEnv(df, days, settings)
            obs, info = env.reset()
            assert obs.shape == (31,)
            assert isinstance(info, dict)

    def test_step_works(self):
        from src.lens1.environment import Lens1LondonEnv
        from src.config import load_settings
        from src.session.profiler import get_trading_days

        settings = load_settings()
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens1LondonEnv(df, days, settings)
            obs, info = env.reset()
            action = env.action_space.sample()
            obs2, reward, term, trunc, info = env.step(action)
            assert obs2.shape == (31,)
            assert isinstance(reward, (int, float))

    def test_full_episode(self):
        from src.lens1.environment import Lens1LondonEnv
        from src.config import load_settings
        from src.session.profiler import get_trading_days

        settings = load_settings()
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens1LondonEnv(df, days, settings)
            obs, info = env.reset()
            done = False
            steps = 0
            while not done and steps < 200:
                action = env.action_space.sample()
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                steps += 1
            assert steps > 0

    def test_reward_uses_reversal_confirmed_sweep_logic(self):
        from src.config import load_settings
        from src.lens1.reward import Lens1RewardCalculator
        from src.session.profiler import SessionProfile

        settings = load_settings()
        settings.smc.sweep_min_lr = 0.2

        london_profile = SessionProfile(
            date="2024-01-15",
            session_type="london",
            open_price=96.0,
            close_price=99.0,
            high=100.0,
            low=90.0,
            mid=95.0,
            range_points=10.0,
            bias=1,
            body_ratio=0.3,
            num_candles=17,
        )
        encoding = np.zeros(8, dtype=np.float32)
        encoding[0] = 1.0
        encoding[3] = 0.8

        reversed_ny = _make_ny_high_sweep(reversed_back=True)
        calc = Lens1RewardCalculator(settings)
        calc.reset(london_profile, reversed_ny, pd.Series(1.0, index=reversed_ny.index))
        reward_with_reversal = calc.compute_step_reward(
            encoding,
            eval_step=len(reversed_ny),
            total_ny_candles=len(reversed_ny),
        )

        no_reversal_ny = _make_ny_high_sweep(reversed_back=False)
        calc.reset(london_profile, no_reversal_ny, pd.Series(1.0, index=no_reversal_ny.index))
        reward_without_reversal = calc.compute_step_reward(
            encoding,
            eval_step=len(no_reversal_ny),
            total_ny_candles=len(no_reversal_ny),
        )

        assert reward_with_reversal > reward_without_reversal + 1.0

    def test_opening_targets_reward_dims_5_to_7(self):
        from src.config import load_settings
        from src.lens1.reward import Lens1RewardCalculator
        from src.session.profiler import SessionProfile

        settings = load_settings()
        london_profile = SessionProfile(
            date="2024-01-15",
            session_type="london",
            open_price=96.0,
            close_price=99.0,
            high=100.0,
            low=90.0,
            mid=95.0,
            range_points=10.0,
            bias=1,
            body_ratio=0.3,
            num_candles=17,
        )
        ny_window = _make_ny_low_sweep(reversed_back=True)
        calc = Lens1RewardCalculator(settings)
        calc.reset(london_profile, ny_window, pd.Series(1.0, index=ny_window.index))

        aligned = np.zeros(8, dtype=np.float32)
        aligned[5] = 1.0
        aligned[6] = 1.0
        aligned[7] = 0.9

        misaligned = np.zeros(8, dtype=np.float32)
        misaligned[5] = -1.0
        misaligned[6] = -1.0
        misaligned[7] = 0.0

        aligned_reward = calc.compute_step_reward(
            aligned,
            eval_step=settings.lens1.opening_target_window_candles,
            total_ny_candles=len(ny_window),
        )
        calc.reset(london_profile, ny_window, pd.Series(1.0, index=ny_window.index))
        misaligned_reward = calc.compute_step_reward(
            misaligned,
            eval_step=settings.lens1.opening_target_window_candles,
            total_ny_candles=len(ny_window),
        )

        assert aligned_reward > misaligned_reward
