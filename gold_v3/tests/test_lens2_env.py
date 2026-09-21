"""
Tests for Lens 2 environment.
"""

import pytest
import numpy as np
import pandas as pd


def _make_test_data():
    """Create minimal test data for Lens 2 env."""
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

    from src.data.indicators import add_indicators
    df = add_indicators(df)
    return df


class TestLens2Env:
    def test_creation(self):
        from src.lens2.environment import Lens2NYEnv
        from src.config import load_settings
        from src.session.profiler import get_trading_days

        settings = load_settings()
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens2NYEnv(df, days, settings=settings)
            assert env.observation_space.shape == (40,)
            assert env.action_space.n == 3

    def test_seed_compatibility(self):
        from src.lens2.environment import Lens2NYEnv
        from src.config import load_settings
        from src.session.profiler import get_trading_days

        settings = load_settings()
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens2NYEnv(df, days, settings=settings)
            seeds = env.seed(123)
            assert isinstance(seeds, list)
            assert len(seeds) == 1
            assert isinstance(seeds[0], int)

    def test_action_masks(self):
        from src.lens2.environment import Lens2NYEnv
        from src.config import load_settings
        from src.session.profiler import get_trading_days

        settings = load_settings()
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens2NYEnv(df, days, settings=settings)
            obs, info = env.reset()
            masks = env.action_masks()
            assert masks.shape == (3,)
            assert masks[0] == True  # HOLD always valid
            assert masks.dtype == bool

    def test_step_hold(self):
        from src.lens2.environment import Lens2NYEnv
        from src.config import load_settings
        from src.session.profiler import get_trading_days

        settings = load_settings()
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens2NYEnv(df, days, settings=settings)
            obs, info = env.reset()
            obs2, reward, term, trunc, info = env.step(0)  # HOLD
            assert obs2.shape == (40,)

    def test_full_episode(self):
        from src.lens2.environment import Lens2NYEnv
        from src.config import load_settings
        from src.session.profiler import get_trading_days

        settings = load_settings()
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens2NYEnv(df, days, settings=settings)
            obs, info = env.reset()
            done = False
            steps = 0
            while not done and steps < 200:
                masks = env.action_masks()
                valid = np.where(masks)[0]
                action = np.random.choice(valid)
                obs, reward, term, trunc, info = env.step(action)
                done = term or trunc
                steps += 1
            assert steps > 0

    def test_news_gate(self):
        from src.lens2.environment import Lens2NYEnv
        from src.config import load_settings
        from src.session.profiler import get_trading_days

        settings = load_settings()
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            # Add all dates as news days
            news = {str(d.date()) if hasattr(d, 'date') else str(d) for d in days}
            env = Lens2NYEnv(df, days, settings=settings, news_dates=news)
            obs, info = env.reset()
            masks = env.action_masks()
            # On news day, BUY and SELL should be masked
            assert masks[1] == False  # BUY masked
            assert masks[2] == False  # SELL masked

    def test_countertrend_buy_uses_live_sweep_flags(self):
        from src.config import load_settings
        from src.lens2.reward import Lens2RewardCalculator

        settings = load_settings()
        calc = Lens2RewardCalculator(settings)
        lens1_encoding = np.zeros(8, dtype=np.float32)
        lens1_encoding[0] = -0.7

        no_sweep_reward = calc.compute_entry_reward(
            1,
            {"state": {}, "raw": {"sweep_low": 0.0, "sweep_high": 0.0}},
            lens1_encoding,
        )
        live_sweep_reward = calc.compute_entry_reward(
            1,
            {"state": {}, "raw": {"sweep_low": 1.0, "sweep_high": 0.0}},
            lens1_encoding,
        )

        assert no_sweep_reward < 0.0
        assert live_sweep_reward > 0.0

    def test_unconfirmed_high_sweep_is_not_missed_setup(self):
        from src.config import load_settings
        from src.lens2.reward import Lens2RewardCalculator

        settings = load_settings()
        calc = Lens2RewardCalculator(settings)
        hold_reward = calc.compute_hold_reward(
            {
                "state": {"choch_bear_recent": 0.0},
                "raw": {"sweep_high": 1.0, "sweep_low": 0.0},
            },
            np.zeros(8, dtype=np.float32),
            has_position=False,
        )

        assert hold_reward == pytest.approx(settings.reward.correct_hold)

    def test_confirmed_high_sweep_uses_stricter_min_rr(self):
        from src.config import load_settings
        from src.lens2.environment import Lens2NYEnv, SELL
        from src.session.profiler import SessionProfile, get_trading_days

        settings = load_settings()
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens2NYEnv(df, days, settings=settings)
            env.london_profile = SessionProfile(
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
            env.lens1_encoding = np.zeros(8, dtype=np.float32)

            features = {
                "state": {"choch_bear_recent": 1.0},
                "raw": {
                    "bull_ob_level": 0.0,
                    "bear_ob_level": 100.0,
                    "pdh": 0.0,
                    "pdl": 0.0,
                    "sweep_high": 1.0,
                    "sweep_low": 0.0,
                },
            }

            stricter_rr = env._get_required_min_rr(SELL, features["raw"], features["state"])
            assert stricter_rr == pytest.approx(env.sweep_high_min_rr)

            features["state"]["choch_bear_recent"] = 0.0
            relaxed_rr = env._get_required_min_rr(SELL, features["raw"], features["state"])
            assert relaxed_rr == pytest.approx(env.min_rr)

    def test_step_info_includes_sweep_day_fields(self):
        from src.lens2.environment import Lens2NYEnv
        from src.config import load_settings
        from src.session.profiler import get_trading_days

        settings = load_settings()
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens2NYEnv(df, days, settings=settings)
            env.reset()
            env._get_current_features = lambda: {
                "state": {"choch_bear_recent": 1.0},
                "raw": {"sweep_high": 1.0, "sweep_low": 0.0},
            }
            _, _, _, _, info = env.step(0)

            assert info["sweep_day"] == 1
            assert info["sweep_high_day"] == 1
            assert info["sweep_low_day"] == 0
            assert info["sweep_direction"] == "high"
            assert "ny_open_trade_taken" in info
            assert "first_trade_pnl" in info

    def test_opening_intent_mapper_fallback_uses_directional_snapshot(self):
        from src.config import load_settings
        from src.lens2.intent import Lens1OpeningIntentMapper

        settings = load_settings()
        mapper = Lens1OpeningIntentMapper(settings)
        mapper.cluster_df = pd.DataFrame()

        encoding = np.zeros(8, dtype=np.float32)
        encoding[5] = 0.8
        encoding[6] = -0.8
        encoding[7] = 0.6
        intent = mapper.classify(encoding, london_bias=1)

        assert intent.scenario_label == "S1_BullContinuation"
        assert intent.opening_side_bias == 1
        assert intent.opening_trade_expected == 1

    def test_action_masks_can_force_high_confidence_opening_entry(self):
        from src.config import load_settings
        from src.lens2.environment import Lens2NYEnv
        from src.lens2.intent import OpeningIntent
        from src.session.profiler import SessionProfile, get_trading_days

        settings = load_settings()
        settings.lens2.opening_intent_force_entry_enabled = True
        settings.lens2.opening_intent_force_hold_mask = True
        settings.lens2.opening_intent_force_entry_confidence = 0.4
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens2NYEnv(df, days, settings=settings)
            env.london_profile = SessionProfile(
                date="2024-01-15",
                session_type="london",
                open_price=96.0,
                close_price=99.0,
                high=97.0,
                low=94.0,
                mid=95.5,
                range_points=3.0,
                bias=1,
                body_ratio=0.3,
                num_candles=17,
            )
            env.opening_intent = OpeningIntent(
                scenario_label="S8_LowLiqGrab",
                scenario_confidence=0.9,
                opening_side_bias=1,
                opening_style_bias=1.0,
                opening_trade_expected=1,
                opening_setup_family="sweep_low",
                opening_aggression=0.9,
                opening_urgency=0.9,
            )
            env.current_step = 0
            env.position = None
            env.trades_this_episode = 0
            env.is_news_day = False
            env.ny_candle_count = 10
            env.ny_candles = df.iloc[:10].copy()
            env.ny_atr = pd.Series(1.0, index=env.ny_candles.index)
            env._get_current_features = lambda: {
                "state": {
                    "bull_ob_dist": 0.5,
                    "bull_fvg_unfilled": 1.0,
                    "bull_fvg_dist": 1.0,
                    "choch_bull_recent": 1.0,
                    "mss_bull_recent": 1.0,
                    "bear_ob_dist": 5.0,
                    "bear_fvg_unfilled": 0.0,
                    "bear_fvg_dist": 5.0,
                    "choch_bear_recent": 0.0,
                    "mss_bear_recent": 0.0,
                },
                "raw": {
                    "current_price": 94.2,
                    "atr": 1.0,
                    "bull_ob_level": 94.0,
                    "bear_ob_level": 0.0,
                    "pdh": 102.0,
                    "pdl": 88.0,
                    "sweep_low": 1.0,
                    "sweep_high": 0.0,
                },
            }

            masks = env.action_masks()
            assert masks[0] == False
            assert masks[1] == True
            assert masks[2] == False

    def test_action_masks_can_force_entry_after_no_trade_delay(self):
        from src.config import load_settings
        from src.lens2.environment import Lens2NYEnv
        from src.lens2.intent import OpeningIntent
        from src.session.profiler import SessionProfile, get_trading_days

        settings = load_settings()
        settings.lens2.no_trade_force_entry_enabled = True
        settings.lens2.no_trade_force_hold_mask = True
        settings.lens2.no_trade_force_min_step = 4
        settings.lens2.no_trade_force_max_step = 12
        settings.lens2.no_trade_force_entry_confidence = 0.35
        settings.lens2.no_trade_force_min_confluence = 3.0
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens2NYEnv(df, days, settings=settings)
            env.london_profile = SessionProfile(
                date="2024-01-15",
                session_type="london",
                open_price=96.0,
                close_price=99.0,
                high=97.0,
                low=94.0,
                mid=95.5,
                range_points=3.0,
                bias=1,
                body_ratio=0.3,
                num_candles=17,
            )
            env.opening_intent = OpeningIntent(
                scenario_label="S8_LowLiqGrab",
                scenario_confidence=0.9,
                opening_side_bias=1,
                opening_style_bias=1.0,
                opening_trade_expected=1,
                opening_setup_family="sweep_low",
                opening_aggression=0.9,
                opening_urgency=0.9,
            )
            env.current_step = 5
            env.position = None
            env.trades_this_episode = 0
            env.is_news_day = False
            env.ny_open_reward_window_steps = 4
            env.ny_candle_count = 16
            env.ny_candles = df.iloc[:16].copy()
            env.ny_atr = pd.Series(1.0, index=env.ny_candles.index)
            env._get_current_features = lambda: {
                "state": {
                    "bull_ob_dist": 0.5,
                    "bull_fvg_unfilled": 1.0,
                    "bull_fvg_dist": 1.0,
                    "choch_bull_recent": 1.0,
                    "mss_bull_recent": 1.0,
                    "bear_ob_dist": 5.0,
                    "bear_fvg_unfilled": 0.0,
                    "bear_fvg_dist": 5.0,
                    "choch_bear_recent": 0.0,
                    "mss_bear_recent": 0.0,
                },
                "raw": {
                    "current_price": 94.2,
                    "atr": 1.0,
                    "bull_ob_level": 94.0,
                    "bear_ob_level": 0.0,
                    "pdh": 102.0,
                    "pdl": 88.0,
                    "sweep_low": 1.0,
                    "sweep_high": 0.0,
                },
            }

            masks = env.action_masks()
            assert masks[0] == False
            assert masks[1] == True
            assert masks[2] == False

    def test_opening_intent_bonus_rewards_matching_ny_open_entry(self):
        from src.config import load_settings
        from src.lens2.reward import Lens2RewardCalculator

        settings = load_settings()
        calc = Lens2RewardCalculator(settings)
        lens1_encoding = np.zeros(8, dtype=np.float32)
        opening_intent = {
            "scenario_label": "S8_LowLiqGrab",
            "scenario_confidence": 0.9,
            "opening_trade_expected": 1,
            "opening_side_bias": 1,
            "opening_setup_family": "sweep_low",
            "opening_aggression": 0.9,
        }
        features = {
            "state": {
                "bull_ob_dist": 0.5,
                "bull_fvg_unfilled": 1.0,
                "bull_fvg_dist": 1.0,
                "choch_bull_recent": 1.0,
                "mss_bull_recent": 1.0,
                "bear_ob_dist": 5.0,
                "bear_fvg_unfilled": 0.0,
                "bear_fvg_dist": 5.0,
                "choch_bear_recent": 0.0,
                "mss_bear_recent": 0.0,
            },
            "raw": {"sweep_low": 1.0, "sweep_high": 0.0},
        }

        matching = calc.compute_entry_reward(
            1,
            features,
            lens1_encoding,
            is_first_trade=True,
            in_ny_open=True,
            opening_intent=opening_intent,
        )
        opposite = calc.compute_entry_reward(
            2,
            features,
            lens1_encoding,
            is_first_trade=True,
            in_ny_open=True,
            opening_intent=opening_intent,
        )

        assert matching > opposite

    def test_opening_intent_hold_penalizes_missing_opening_trade(self):
        from src.config import load_settings
        from src.lens2.reward import Lens2RewardCalculator

        settings = load_settings()
        calc = Lens2RewardCalculator(settings)
        features = {
            "state": {
                "bull_ob_dist": 0.5,
                "bull_fvg_unfilled": 1.0,
                "bull_fvg_dist": 1.0,
                "choch_bull_recent": 1.0,
                "mss_bull_recent": 0.0,
            },
            "raw": {"sweep_low": 1.0, "sweep_high": 0.0},
        }
        opening_intent = {
            "scenario_label": "S8_LowLiqGrab",
            "scenario_confidence": 0.9,
            "opening_trade_expected": 1,
            "opening_side_bias": 1,
            "opening_setup_family": "sweep_low",
            "opening_aggression": 0.9,
        }

        reward = calc.compute_hold_reward(
            features,
            np.zeros(8, dtype=np.float32),
            has_position=False,
            opening_intent=opening_intent,
            is_first_trade_pending=True,
            in_ny_open=True,
        )

        assert reward == pytest.approx(settings.lens2.opening_intent_missed_penalty)

    def test_hold_reward_penalizes_feasible_no_trade_trap(self):
        from src.config import load_settings
        from src.lens2.reward import Lens2RewardCalculator

        settings = load_settings()
        calc = Lens2RewardCalculator(settings)
        features = {"state": {}, "raw": {}}

        baseline = calc.compute_hold_reward(
            features,
            np.zeros(8, dtype=np.float32),
            has_position=False,
            trades_this_episode=0,
            current_step=6,
            opening_window_steps=4,
            trade_plan_feasible=False,
        )
        penalized = calc.compute_hold_reward(
            features,
            np.zeros(8, dtype=np.float32),
            has_position=False,
            trades_this_episode=0,
            current_step=6,
            opening_window_steps=4,
            trade_plan_feasible=True,
        )

        assert penalized < baseline

    def test_trade_log_captures_first_trade_timing_and_intent(self):
        from src.config import load_settings
        from src.lens2.environment import Lens2NYEnv, BUY
        from src.lens2.intent import OpeningIntent
        from src.session.profiler import SessionProfile, get_trading_days

        settings = load_settings()
        df = _make_test_data()
        days = get_trading_days(df)

        if len(days) > 0:
            env = Lens2NYEnv(df, days, settings=settings)
            env.ny_candles = df.iloc[:8].copy()
            env.ny_candle_count = len(env.ny_candles)
            env.candle_minutes = 15
            env.current_date_str = "2024-01-15"
            env.ny_session_start_ts = pd.Timestamp("2024-01-15 12:00:00", tz="UTC")
            env.opening_intent = OpeningIntent(
                scenario_label="S8_LowLiqGrab",
                scenario_confidence=0.9,
                opening_side_bias=1,
                opening_trade_expected=1,
                opening_setup_family="sweep_low",
                opening_aggression=0.9,
            )
            env.london_profile = SessionProfile(
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
            env.current_step = 2
            features = {
                "state": {
                    "bull_ob_dist": 0.5,
                    "bull_fvg_unfilled": 1.0,
                    "bull_fvg_dist": 1.0,
                    "choch_bull_recent": 1.0,
                    "mss_bull_recent": 1.0,
                },
                "raw": {
                    "bull_ob_level": 94.0,
                    "bear_ob_level": 0.0,
                    "pdh": 102.0,
                    "pdl": 88.0,
                    "sweep_low": 1.0,
                    "sweep_high": 0.0,
                },
            }

            env._open_position(BUY, price=95.0, atr_val=1.0, features=features)
            env.current_step = 4
            env._close_position(99.0, "tp3_hit")
            info = env._build_info()
            trade = env.trade_log[-1]

            assert trade["is_first_trade"] == 1
            assert trade["matches_opening_intent"] == 1
            assert trade["entry_step"] == 2
            assert trade["holding_minutes"] == 45.0
            assert trade["opening_scenario_label"] == "S8_LowLiqGrab"
            assert info["first_trade_duration_min"] == pytest.approx(45.0)
            assert info["first_trade_matches_intent"] == 1
