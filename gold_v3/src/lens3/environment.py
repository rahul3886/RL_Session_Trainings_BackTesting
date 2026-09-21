"""
Lens 2 RL Environment — NY Session Executor.

Lens 2 receives Lens 1's anchored encoding as London context,
then learns to place institutional-grade trades during the NY session.
It never re-runs London analysis — it trusts Lens 1's encoded context.

Action Space: Discrete(3) — HOLD=0, BUY=1, SELL=2
Observation Space: Box(40) = 8 (Lens 1) + 32 (live NY features)
"""

import logging
from typing import Optional, Dict, List

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding

from src.session.profiler import (
    get_session_pair, get_ny_after_anchor, get_london_until_anchor,
    compute_session_profile,
)
from src.session.features import extract_session_features, build_lens3_state_vector
from src.lens3.intent import Lens1OpeningIntentMapper, OpeningIntent
from src.smc.order_blocks import get_nearest_ob
from src.smc.liquidity import build_pdh_pdl_cache
from src.lens3.reward import Lens3RewardCalculator
from src.lens1.labeler import HistoricalScenarioLabeler, SCENARIO_NAMES
from src.lens1.geometry import LondonSessionProfiler

logger = logging.getLogger(__name__)

# Actions
HOLD = 0
BUY = 1
SELL = 2


class Lens3NYEnv(gym.Env):
    """
    Lens 2 NY Executor Environment with action masking.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        trading_days: list,
        lens1_model=None,
        lens1_env=None,
        settings=None,
        news_dates: set = None,
        lens2_predictor=None,
        lens2_scaler=None,
        df_m1: Optional[pd.DataFrame] = None,
        df_m5: Optional[pd.DataFrame] = None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.df = df
        self.df_m1 = df_m1
        self.df_m5 = df_m5
        self.trading_days = list(trading_days)
        self._shuffled_days = list(trading_days)
        self.lens1_model = lens1_model
        self.lens1_env = lens1_env
        self.lens2_predictor = lens2_predictor
        self.lens2_scaler = lens2_scaler
        self.settings = settings
        self.news_dates = news_dates or set()
        self.render_mode = render_mode
        self._pdh_pdl_cache = build_pdh_pdl_cache(df)
        self._day_context_cache = {}
        self._lens1_encoding_cache = {}
        self._opening_intent_cache = {}
        self._step_cache_step = None
        self._step_cache_bundle = None

        # Config
        self.obs_dim = 40 if settings is None else settings.lens3.obs_dim
        self.encoding_dim = 8 if settings is None else settings.lens1.encoding_dim

        session_cfg = None if settings is None else settings.sessions
        self.ny_start = "12:00" if session_cfg is None else session_cfg.ny_start
        self.ny_end = "21:00" if session_cfg is None else session_cfg.ny_end
        self.anchor_time = "12:15" if session_cfg is None else session_cfg.lens1_anchor
        self.no_trade_time = "20:00" if session_cfg is None else session_cfg.ny_no_trade
        self.london_start = "08:00" if session_cfg is None else session_cfg.london_start

        lens3_cfg = None if settings is None else settings.lens3
        self.max_trades = 3 if lens3_cfg is None else lens3_cfg.max_trades_per_day
        self.min_rr = 1.5 if lens3_cfg is None else lens3_cfg.min_rr
        self.sl_buffer_lr = 0.10 if lens3_cfg is None else lens3_cfg.sl_buffer_lr
        self.sl_min_atr_gate = 0.3 if lens3_cfg is None else lens3_cfg.sl_min_atr_gate
        self.sl_max_lr_gate = 1.5 if lens3_cfg is None else lens3_cfg.sl_max_lr_gate
        self.london_range_max = 4.0 if lens3_cfg is None else lens3_cfg.london_range_atr_max
        self.sweep_asymmetry_mode = True if lens3_cfg is None else lens3_cfg.sweep_asymmetry_mode
        self.sweep_low_min_rr = self.min_rr if lens3_cfg is None else lens3_cfg.sweep_low_min_rr
        self.sweep_high_min_rr = 2.0 if lens3_cfg is None else lens3_cfg.sweep_high_min_rr
        self.sweep_high_requires_choch = True if lens3_cfg is None else lens3_cfg.sweep_high_requires_choch
        self.ny_open_reward_window_steps = 4 if lens3_cfg is None else lens3_cfg.ny_open_reward_window_steps
        self.opening_intent_force_entry_enabled = False if lens3_cfg is None else lens3_cfg.opening_intent_force_entry_enabled
        self.opening_intent_force_hold_mask = False if lens3_cfg is None else lens3_cfg.opening_intent_force_hold_mask
        self.opening_intent_force_entry_confidence = 0.55 if lens3_cfg is None else lens3_cfg.opening_intent_force_entry_confidence
        self.no_trade_force_entry_enabled = False if lens3_cfg is None else getattr(lens3_cfg, "no_trade_force_entry_enabled", False)
        self.no_trade_force_hold_mask = False if lens3_cfg is None else getattr(lens3_cfg, "no_trade_force_hold_mask", False)
        self.no_trade_force_min_step = 4 if lens3_cfg is None else getattr(lens3_cfg, "no_trade_force_min_step", 4)
        self.no_trade_force_max_step = 12 if lens3_cfg is None else getattr(lens3_cfg, "no_trade_force_max_step", 12)
        self.no_trade_force_entry_confidence = 0.35 if lens3_cfg is None else getattr(lens3_cfg, "no_trade_force_entry_confidence", 0.35)
        self.no_trade_force_min_confluence = 3.0 if lens3_cfg is None else getattr(lens3_cfg, "no_trade_force_min_confluence", 3.0)

        # Oracle / Scenario Handover Config
        self.scenario_handover_enabled = False if lens3_cfg is None else getattr(lens3_cfg, "scenario_handover_enabled", True)
        self.oracle_mode = False if lens3_cfg is None else getattr(lens3_cfg, "oracle_mode", True)
        self.mask_mode = "strict_smc" if lens3_cfg is None else getattr(lens3_cfg, "mask_mode", "strict_smc")
        self.displacement_min = 1.2 if lens3_cfg is None else getattr(lens3_cfg, "displacement_min", 1.2)
        self.momentum_r2_max = 0.65 if lens3_cfg is None else getattr(lens3_cfg, "momentum_r2_max", 0.65)
        self.momentum_r2_max = 0.65 if lens3_cfg is None else getattr(lens3_cfg, 'momentum_r2_max', 0.65)
        self.target_fps = None if lens3_cfg is None else getattr(lens3_cfg, 'target_fps', None)
        self.mask_scenarios = False if lens3_cfg is None else getattr(lens3_cfg, "mask_scenarios", False)
        self.geometry_enabled = False if lens3_cfg is None else getattr(lens3_cfg, "geometry_enabled", True)

        self.labeler = HistoricalScenarioLabeler(settings)
        self.profiler = LondonSessionProfiler(settings)

        # Spaces
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(3)

        # Episode state
        self.current_day_idx = 0
        self.current_step = 0
        self.lens1_encoding = np.zeros(self.encoding_dim, dtype=np.float32)
        self.current_scenario_id = -1
        self.london_geometry = {}

        # Trading state
        self.position = None          # None, "long", "short"
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp1_price = 0.0
        self.tp2_price = 0.0
        self.tp3_price = 0.0
        self.position_size = 1.0
        self.trades_this_episode = 0
        self.trade_log = []
        self.episode_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.current_trade_realized_pnl = 0.0
        self.current_trade_is_first = False
        self.current_trade_opened_in_ny_open = False
        self.current_trade_entry_step = None
        self.current_trade_entry_time = None
        self.current_trade_matches_intent = False

        # Session data
        self.ny_candles = None
        self.london_profile = None
        self.ny_atr = None
        self.ny_candle_count = 0
        self.is_news_day = False
        self.current_date_str = ""
        self.episode_sweep_high_seen = False
        self.episode_sweep_low_seen = False
        self.episode_ny_open_trade_taken = False
        self.first_trade_pnl = None
        self.first_trade_entry_step = None
        self.first_trade_entry_time = None
        self.first_trade_entry_delay_from_anchor_min = None
        self.first_trade_entry_delay_from_ny_open_min = None
        self.first_trade_duration_min = None
        self.first_trade_is_ny_open = False
        self.first_trade_matches_intent = False
        self.candle_minutes = 15
        self.ny_session_start_ts = None
        self.anchor_ts = None
        self.opening_intent = OpeningIntent()

        # Reward calculator
        self.reward_calc = Lens3RewardCalculator(settings)
        self.intent_mapper = Lens1OpeningIntentMapper(settings)

        # Phase 9: HMM Regime Engine
        try:
            from src.data.regime_hmm import RegimeHMM
            import os
            model_path = os.path.join(getattr(settings, 'project_root', ''), "models", "lens0", "hmm_model.pkl")
            self.hmm_engine = RegimeHMM(model_path=model_path)
            self.hmm_engine.load()
        except Exception as e:
            self.hmm_engine = None

    def seed(self, seed=None):
        """Compatibility shim for SB3 subprocess env creation."""
        self.np_random, actual_seed = seeding.np_random(seed)
        import random
        random.seed(actual_seed)
        np.random.seed(actual_seed)
        return [actual_seed]

    def reset(self, seed=None, options=None):
        """Reset to a new trading day."""
        super().reset(seed=seed)

        if self.current_day_idx >= len(self._shuffled_days):
            self.current_day_idx = 0
            import random
            random.shuffle(self._shuffled_days)

        date = self._shuffled_days[self.current_day_idx]
        self.current_day_idx += 1

        day_context = self._get_day_context(date)
        self.london_profile = day_context["london_profile"]
        self.ny_candles = day_context["ny_candles"]
        self.ny_atr = day_context["ny_atr"]
        self.ny_candle_count = day_context["ny_candle_count"]
        self.candle_minutes = day_context["candle_minutes"]
        self.ny_session_start_ts = day_context["ny_session_start_ts"]
        self.anchor_ts = self._get_timestamp_for_step(0)

        # Reset Lens 1 encoding
        self.lens1_encoding = self._get_lens1_encoding(date)
        london_bias = self.london_profile.bias if self.london_profile else 0
        self.opening_intent = self._get_opening_intent(date, london_bias)

        # Lens 1 2.0: Scenario Handover & Geometry
        self.current_scenario_id = self._get_scenario_id(date)
        self.london_geometry = self._get_london_geometry(date)

        # News gate
        self.current_date_str = day_context["date_str"]
        self.is_news_day = day_context["is_news_day"]

        # Reset trading state
        self.current_step = 0
        self.position = None
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.tp1_price = 0.0
        self.tp2_price = 0.0
        self.tp3_price = 0.0
        self.position_size = 1.0
        self.trades_this_episode = 0
        self.trade_log = []
        self.episode_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.current_trade_realized_pnl = 0.0
        self.current_trade_is_first = False
        self.current_trade_opened_in_ny_open = False
        self.current_trade_entry_step = None
        self.current_trade_entry_time = None
        self.current_trade_matches_intent = False
        self.episode_sweep_high_seen = False
        self.episode_sweep_low_seen = False
        self.episode_ny_open_trade_taken = False
        self.first_trade_pnl = None
        self.first_trade_entry_step = None
        self.first_trade_entry_time = None
        self.first_trade_entry_delay_from_anchor_min = None
        self.first_trade_entry_delay_from_ny_open_min = None
        self.first_trade_duration_min = None
        self.first_trade_is_ny_open = False
        self.first_trade_matches_intent = False
        self._reset_step_cache()

        self.reward_calc.reset()

        obs = self._get_observation()
        info = self._build_info()

        return obs, info

    def step(self, action: int):
        """Execute one step in the NY session."""
        reward = 0.0
        terminated = False
        truncated = False

        if self.current_step >= self.ny_candle_count:
            terminated = True
            return self._get_observation(), 0.0, True, False, {}

        current_candle = self.ny_candles.iloc[self.current_step]
        current_price = current_candle["close"]
        current_high = current_candle["high"]
        current_low = current_candle["low"]
        atr_val = self.ny_atr.iloc[self.current_step] if self.current_step < len(self.ny_atr) else 1.0
        features = self._get_current_features()
        self._update_episode_context(features)

        # Check SL/TP for open position
        if self.position is not None:
            reward += self._check_position(current_high, current_low, current_price, atr_val)

        # Execute new action if no position
        if action in (BUY, SELL) and self.position is None:
            # Compute structural SL/TP
            trade_reward = self._open_position(action, current_price, atr_val, features)
            reward += trade_reward
        elif action == HOLD:
            # Check if a trade could actually be placed right now
            current_price = current_candle["close"]
            plan_buy = self._build_trade_plan(BUY, current_price, atr_val, features)
            plan_sell = self._build_trade_plan(SELL, current_price, atr_val, features)
            trade_plan_feasible = plan_buy is not None or plan_sell is not None

            reward += self.reward_calc.compute_hold_reward(
                features,
                self.lens1_encoding,
                self.position is not None,
                opening_intent=self.opening_intent.to_dict(),
                is_first_trade_pending=self.trades_this_episode == 0,
                in_ny_open=self.current_step < self.ny_open_reward_window_steps,
                opening_delay_steps=self.current_step,
                opening_window_steps=self.ny_open_reward_window_steps,
                trades_this_episode=self.trades_this_episode,
                current_step=self.current_step,
                trade_plan_feasible=trade_plan_feasible,
            )

        self.current_step += 1

        # --- Phase 8: Strategic FPS Throttle (Institutional Pacing) ---
        if self.target_fps is not None and self.target_fps > 0:
            import time
            time.sleep(1.0 / self.target_fps)

        # Check if episode is over
        if self.current_step >= self.ny_candle_count:
            # Force close any open position
            if self.position is not None:
                reward += self._close_position(current_price, "time_exit")

            # Episode end bonus
            reward += self.reward_calc.compute_episode_end_reward(
                self.episode_pnl,
                self.trades_this_episode,
                self.wins,
                self.losses,
                self.max_trades,
            )
            terminated = True

        obs = self._get_observation()
        info = self._build_info()

        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """
        Return boolean mask for valid actions.
        True = action allowed, False = action forbidden.
        """
        mask = np.array([True, True, True])  # [HOLD, BUY, SELL]

        # Always can HOLD
        # mask[0] = True

        # Mask BUY/SELL if:
        if self.position is not None:
            mask[BUY] = False
            mask[SELL] = False

        if self.trades_this_episode >= self.max_trades:
            mask[BUY] = False
            mask[SELL] = False

        # Time gate: no trades in last 60 min
        if self.ny_candle_count > 0 and self.current_step >= 0:
            remaining = self.ny_candle_count - self.current_step
            if remaining <= 4:  # ~60 min on 15M candles
                mask[BUY] = False
                mask[SELL] = False

        # London range gate
        if self.london_profile is not None:
            atr_val = 1.0
            if self.ny_atr is not None and len(self.ny_atr) > 0:
                atr_val = max(self.ny_atr.iloc[min(self.current_step, len(self.ny_atr) - 1)], 0.01)
            london_range_atr = self.london_profile.range_points / atr_val
            if london_range_atr > self.london_range_max:
                mask[BUY] = False
                mask[SELL] = False

        # News gate: mask ALL trading on news days
        if self.is_news_day:
            mask[BUY] = False
            mask[SELL] = False

        if (
            self.opening_intent_force_entry_enabled
            and self.position is None
            and self.trades_this_episode == 0
            and self.current_step < self.ny_open_reward_window_steps
            and not self.is_news_day
            and self.opening_intent.scenario_confidence >= self.opening_intent_force_entry_confidence
        ):
            features = self._get_current_features()
            intent = self.opening_intent.to_dict()
            if self.reward_calc._opening_trade_ready(
                features,
                intent,
                is_first_trade_pending=True,
                in_ny_open=True,
            ):
                preferred = BUY if self.opening_intent.opening_side_bias > 0 else SELL if self.opening_intent.opening_side_bias < 0 else None
                price = float(features["raw"].get("current_price", 0.0))
                atr_val = float(features["raw"].get("atr", 1.0))
                if preferred is not None and self._build_trade_plan(preferred, price, atr_val, features) is not None:
                    if preferred == BUY:
                        mask[SELL] = False
                    else:
                        mask[BUY] = False
                    if self.opening_intent_force_hold_mask:
                        mask[HOLD] = False

        # --- Phase 8: Strict SMC & Institutional Quant Masking ---
        features = self._get_current_features()
        s = features.get("state", {})
        raw = features.get("raw", {})

        if self.mask_mode == "strict_smc":
            # --- Phase 9: Dynamic Thresholding (HMM Regime Filter) ---
            req_displacement = self.displacement_min
            req_r2 = self.momentum_r2_max
            regime = s.get("regime_state", 0.0)
            
            if regime == 1.0 or regime == 2.0:
                # We are in a confirmed trend (Machine Gun Mode)
                # Lower the displacement threshold and increase allowable R2 limits
                req_displacement = max(0.5, self.displacement_min - 0.7)
                req_r2 = min(0.95, self.momentum_r2_max + 0.2)

            # 1. Geometry Check (Already pre-filtered in previous chunks)
            # 2. Institutional Quant Checks
            current_displacement = s.get("displacement", 0.0)
            current_r2 = s.get("momentum_r2", 1.0)
            
            # Displacement Requirement: Needs force for BOS/CHOCH
            if abs(current_displacement) < req_displacement:
                # If we aren't sweeping, we need displacement
                if raw.get("sweep_low", 0) == 0: mask[BUY] = False
                if raw.get("sweep_high", 0) == 0: mask[SELL] = False
            
            # Exhaustion Check: Don't enter if trend is too 'stable' (High R2) at extremes
            if current_r2 > req_r2:
                # High R2 means trend is very healthy; dangerous to fade sweeps
                if raw.get("sweep_low", 0) > 0: mask[BUY] = False
                if raw.get("sweep_high", 0) > 0: mask[SELL] = False
                
            # Phase 10: Fractal Sync Override & Slippage Buffer
            fractal_sync = s.get("fractal_sync", 0.0)
            if fractal_sync > 0:
                # If we have a Zero-Latency trigger, we want to OVERRIDE the displacement block
                # BUT we must enforce the Slippage Buffer!
                # Price must not have chased more than 0.2x ATR from the structure.
                atr_val = raw.get("atr", 1.0)
                cp = raw.get("current_price", 0.0)
                
                # Check nearest OB distances
                bull_dist_atr = s.get("bull_ob_dist", 5.0)
                bear_dist_atr = s.get("bear_ob_dist", 5.0)
                
                if regime == 1.0 and bull_dist_atr <= 0.2:
                    mask[BUY] = True
                    mask[SELL] = False
                elif regime == 2.0 and bear_dist_atr <= 0.2:
                    mask[SELL] = True
                    mask[BUY] = False
                else:
                    # Slippage Buffer violated (Price chased > 0.2 ATR from setup), or wrong regime sync
                    # Disqualify the early entry completely
                    mask[BUY] = False
                    mask[SELL] = False

        # --- Phase 6: Scenario-Based Action Masking (Toggleable) ---
        elif self.mask_mode == "scenario" and self.current_scenario_id != -1:
            # S1_BullDefend, S3_BullBreak, S5_TightHigh, S8_SweepLow -> Prefer BUY
            bullish_scenarios = [0, 2, 4, 7]
            # S2_BearDefend, S4_BearBreak, S6_TightLow, S7_SweepHigh -> Prefer SELL
            bearish_scenarios = [1, 3, 5, 6]

            if self.current_scenario_id in bullish_scenarios:
                mask[SELL] = False
            elif self.current_scenario_id in bearish_scenarios:
                mask[BUY] = False

        if mask[HOLD] and self.position is None and self.trades_this_episode == 0 and not self.is_news_day:
            forced_action = self._resolve_no_trade_forced_action()
            if forced_action is not None and mask[forced_action]:
                mask[BUY] = bool(mask[BUY] and forced_action == BUY)
                mask[SELL] = bool(mask[SELL] and forced_action == SELL)
                if self.no_trade_force_hold_mask:
                    mask[HOLD] = False

        return mask

    def _resolve_no_trade_forced_action(self) -> Optional[int]:
        if not self.no_trade_force_entry_enabled:
            return None
        if self.current_step < self.no_trade_force_min_step:
            return None
        if self.current_step >= self.no_trade_force_max_step:
            return None

        features = self._get_current_features()
        raw = features.get("raw", {})
        price = float(raw.get("current_price", 0.0))
        atr_val = float(raw.get("atr", 1.0))

        candidates: Dict[int, float] = {}
        for action in (BUY, SELL):
            if self._build_trade_plan(action, price, atr_val, features) is None:
                continue
            confluence_count, entry_confidence = self.reward_calc.estimate_entry_quality(action, features)
            if confluence_count < self.no_trade_force_min_confluence:
                continue
            candidates[action] = confluence_count * max(entry_confidence, 0.5)

        if not candidates:
            return None

        opening_intent = self.opening_intent.to_dict()
        intent_strength = self.reward_calc._opening_strength(opening_intent)
        if intent_strength >= self.no_trade_force_entry_confidence:
            if self.opening_intent.opening_side_bias > 0 and BUY in candidates:
                return BUY
            if self.opening_intent.opening_side_bias < 0 and SELL in candidates:
                return SELL

        if len(candidates) == 1:
            return next(iter(candidates))

        ranked = sorted(candidates.items(), key=lambda item: item[1], reverse=True)
        if len(ranked) > 1 and ranked[0][1] <= ranked[1][1]:
            return None
        return ranked[0][0]

    def _get_lens1_encoding(self, date) -> np.ndarray:
        """Get Lens 1's encoding for this day (anchored weights)."""
        date_key = self._date_key(date)
        cached = self._lens1_encoding_cache.get(date_key)
        if cached is not None:
            return cached.copy()

        if self.lens1_model is None:
            return np.zeros(self.encoding_dim, dtype=np.float32)

        try:
            # Run Lens 1 env to get encoding
            london_candles = get_london_until_anchor(
                self.df, date, self.london_start, self.anchor_time
            )

            if len(london_candles) == 0:
                return np.zeros(self.encoding_dim, dtype=np.float32)

            if self.lens1_env is not None:
                # Use the dedicated Lens 1 env
                self.lens1_env.current_day_idx = max(0, self.current_day_idx - 1)
                obs, _ = self.lens1_env.reset()

                encoding = np.zeros(self.encoding_dim, dtype=np.float32)
                done = False
                while not done:
                    action, _ = self.lens1_model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = self.lens1_env.step(action)
                    done = terminated or truncated
                    if "encoding" in info:
                        encoding = info["encoding"]

                cached_encoding = encoding.astype(np.float32)
                self._lens1_encoding_cache[date_key] = cached_encoding
                return cached_encoding.copy()
            else:
                return np.zeros(self.encoding_dim, dtype=np.float32)

        except Exception as e:
            logger.warning(f"Lens 1 encoding failed for {date}: {e}")
            return np.zeros(self.encoding_dim, dtype=np.float32)

    @staticmethod
    def _date_key(date) -> str:
        return str(pd.Timestamp(date).date())

    def _reset_step_cache(self) -> None:
        self._step_cache_step = None
        self._step_cache_bundle = None

    def _get_day_context(self, date) -> Dict:
        date_key = self._date_key(date)
        cached = self._day_context_cache.get(date_key)
        if cached is not None:
            return cached

        _, _, london_profile, _ = get_session_pair(
            self.df, date,
            self.settings.sessions if self.settings else None,
        )
        ny_candles = get_ny_after_anchor(
            self.df, date, self.anchor_time, self.ny_end
        )
        if "atr" in self.df.columns and len(ny_candles) > 0:
            ny_atr = self.df.loc[ny_candles.index, "atr"].copy()
        else:
            ny_atr = pd.Series(
                np.ones(len(ny_candles)),
                index=ny_candles.index if len(ny_candles) > 0 else None,
            )

        context = {
            "london_profile": london_profile,
            "ny_candles": ny_candles,
            "ny_atr": ny_atr,
            "ny_candle_count": len(ny_candles),
            "candle_minutes": self._infer_candle_minutes(ny_candles),
            "ny_session_start_ts": self._resolve_session_timestamp(date, self.ny_start, ny_candles),
            "date_str": date_key,
            "is_news_day": date_key in self.news_dates,
        }
        self._day_context_cache[date_key] = context
        return context

    def _get_opening_intent(self, date, london_bias: int) -> OpeningIntent:
        date_key = self._date_key(date)
        cached = self._opening_intent_cache.get(date_key)
        if cached is None:
            from src.lens3.intent import Lens1OpeningIntentMapper
            mapper = Lens1OpeningIntentMapper(self.settings) if hasattr(self, 'intent_mapper') else None
            # fallback if intent_mapper not initialized
            if not hasattr(self, 'intent_mapper'):
                self.intent_mapper = Lens1OpeningIntentMapper(self.settings)
            cached = self.intent_mapper.classify(self.lens1_encoding, london_bias=london_bias)
            self._opening_intent_cache[date_key] = cached
        return OpeningIntent(**cached.to_dict())

    def _get_scenario_id(self, date) -> int:
        """Fetch ground-truth label (Oracle) or predicted label (Lens 2 Classifier)."""
        day_context = self._get_day_context(date)
        lp = day_context["london_profile"]
        ny_after = day_context["ny_candles"]

        if self.oracle_mode or self.lens2_predictor is None or self.lens2_scaler is None:
            return self.labeler.label_day(self.df, lp, ny_after)
        else:
            # Predict scenario using Lens 2
            import torch
            geometry = self._get_london_geometry(date)
            
            # Reconstruct the feature vector format expected by the predictor
            features = [
                london_range := lp.range_points if lp else 0.0,
                lp.high if lp else 0.0,
                lp.low if lp else 0.0,
                ny_after["atr"].iloc[0] if "atr" in ny_after.columns and len(ny_after) > 0 else 1.0,
                geometry.get("london_slope", 0.0),
                geometry.get("london_r2", 0.0),
                geometry.get("london_pivot_density", 0.0),
                geometry.get("london_range_pos", 0.0),
                geometry.get("london_vol_bias", 0.0),
                geometry.get("london_momentum_ratio", 0.0),
                geometry.get("london_volatility_relative", 0.0),
            ]
            
            # Predict
            try:
                x_scaled = self.lens2_scaler.transform([features])
                x_tensor = torch.FloatTensor(x_scaled)
                with torch.no_grad():
                    pred = self.lens2_predictor.predict(x_tensor).item()
                    # Class 8 is mapped to S_UNDEFINED (-1)
                    if pred == 8:
                        return -1
                    return pred
            except Exception as e:
                logger.error(f"Lens 2 prediction failed: {e}")
                return -1

    def _get_oracle_bias(self) -> int:
        """Return 1 for Bullish scenarios, -1 for Bearish, 0 otherwise."""
        if not self.oracle_mode or self.current_scenario_id == -1:
            return 0
        
        # S1_BullDefend, S3_BullBreak, S5_TightHigh, S8_SweepLow
        if self.current_scenario_id in [0, 2, 4, 7]:
            return 1
        # S2_BearDefend, S4_BearBreak, S6_TightLow, S7_SweepHigh
        if self.current_scenario_id in [1, 3, 5, 6]:
            return -1
        return 0

    def _get_london_geometry(self, date) -> Dict[str, float]:
        """Fetch mathematical footprints of the London session."""
        date_key = self._date_key(date)
        london_candles = get_london_until_anchor(
            self.df, date, self.london_start, self.anchor_time
        )
        return self.profiler.profile_geometry(london_candles)

    def _get_step_bundle(self) -> Optional[Dict]:
        if self.ny_candle_count == 0 or self.current_step >= self.ny_candle_count:
            return None

        if self._step_cache_step == self.current_step and self._step_cache_bundle is not None:
            return self._step_cache_bundle

        end_idx = min(self.current_step + 1, self.ny_candle_count)
        candle_window = self.ny_candles.iloc[:end_idx]
        atr_window = self.ny_atr.iloc[:end_idx] if len(self.ny_atr) >= end_idx else self.ny_atr

        features = extract_session_features(
            candle_window, atr_window,
            full_df=self.df,
            session_high=self.london_profile.high if self.london_profile else None,
            session_low=self.london_profile.low if self.london_profile else None,
            smc_config=self.settings.smc if self.settings else None,
            pdh_pdl_cache=self._pdh_pdl_cache,
        )

        ny_time_norm = self.current_step / max(self.ny_candle_count, 1)
        in_ny_open = 1.0 if self.current_step < 4 else 0.0

        ny_broke_lh = 0.0
        ny_broke_ll = 0.0
        if self.london_profile:
            ny_high = candle_window["high"].max()
            ny_low = candle_window["low"].min()
            if ny_high > self.london_profile.high:
                ny_broke_lh = 1.0
            if ny_low < self.london_profile.low:
                ny_broke_ll = 1.0

        current_candle = self.ny_candles.iloc[min(self.current_step, self.ny_candle_count - 1)]
        momentum = current_candle.get("momentum", 0.0) if "momentum" in self.ny_candles.columns else 0.0
        volume_ratio = current_candle.get("volume_ratio", 1.0) if "volume_ratio" in self.ny_candles.columns else 1.0

        # --- Phase 9/10: Live HMM Regime State & Fractal Velocity ---
        regime_state = 0
        hmm_velocity = 0.0
        cpd_age = -1.0  # -1 means no break found
        fractal_sync = 0.0 # 1.0 if Multi-TF execution triggers

        if self.hmm_engine is not None:
            # We pass the full history up to this step (London + NY so far)
            hist = self.df.loc[:current_candle.name]
            df_m5_hist = None
            if self.df_m5 is not None:
                df_m5_hist = self.df_m5.loc[:current_candle.name]
                
            regime_state, hmm_velocity, _ = self.hmm_engine.predict_with_velocity(
                hist, df_m5=df_m5_hist, lookback=30
            )
            
            # Change Point Detection (Ruptures) on last 60 bars of ATR
            try:
                import ruptures as rpt
                import numpy as np
                if len(hist) > 60:
                    sig = hist["atr"].iloc[-60:].values
                    algo = rpt.Pelt(model="rbf").fit(sig)
                    # pen=10 is a standard threshold to avoid overfitting minor volatility shifts
                    breaks = algo.predict(pen=10)
                    if breaks and len(breaks) > 1:
                        # Convert the index of the break (relative to window) to step age
                        last_break_idx = breaks[-2] # Last break before the end of signal
                        cpd_age = (60 - last_break_idx) / 60.0 # Normalized 0 to 1
            except Exception:
                pass

            # Phase 10: Fractal Confluence (Zero-Latency Trigger)
            if hmm_velocity > 0.40 and df_m5_hist is not None and self.df_m1 is not None:
                try:
                    # M5 VWAP Z-Score crossed 1.0 in same direction?
                    latest_m5 = df_m5_hist.iloc[-1]
                    m5_zscore = latest_m5.get("vwap_zscore", 0.0)
                    m5_valid = (regime_state == 1 and m5_zscore > 1.0) or (regime_state == 2 and m5_zscore < -1.0)
                    
                    if m5_valid:
                        # M1 Micro-CPD on Volume Delta
                        df_m1_hist = self.df_m1.loc[:current_candle.name]
                        if len(df_m1_hist) > 30:
                            import ruptures as rpt
                            # Simple proxy for volume delta if we just have close & volume
                            m1_vol = df_m1_hist["volume"].iloc[-30:].values
                            m1_ret = df_m1_hist["close"].pct_change().fillna(0).iloc[-30:].values
                            m1_vd = m1_vol * np.sign(m1_ret)
                            algo = rpt.Pelt(model="rbf").fit(m1_vd)
                            m1_breaks = algo.predict(pen=5)
                            if m1_breaks and m1_breaks[-1] < 30:
                                fractal_sync = 1.0
                except Exception as e:
                    logger.warning(f"Failed Fractal Confluence: {e}")
        
        # Inject Phase 9/10 features into the dict so we can use them in action_masks
        features["state"]["regime_state"] = float(regime_state)
        features["state"]["cpd_age"] = cpd_age
        features["state"]["hmm_velocity"] = hmm_velocity
        features["state"]["fractal_sync"] = fractal_sync

        obs_list = features.get("order_blocks", [])
        cp = features["raw"]["current_price"]
        atr_val = features["raw"]["atr"]
        nearest_ob_dist = 5.0
        for ob in obs_list:
            if not ob.mitigated:
                if ob.type == "bull":
                    d = abs(cp - ob.high) / max(atr_val, 0.01)
                else:
                    d = abs(ob.low - cp) / max(atr_val, 0.01)
                nearest_ob_dist = min(nearest_ob_dist, d)

        observation_base = build_lens3_state_vector(
            self.lens1_encoding,
            features,
            self.london_profile,
            ny_time_norm=ny_time_norm,
            in_ny_open=in_ny_open,
            momentum=momentum,
            volume_ratio=volume_ratio,
            ny_broke_lh=ny_broke_lh,
            ny_broke_ll=ny_broke_ll,
            distance_to_nearest_ob_atr=nearest_ob_dist,
            vwap_zscore=features["state"].get("vwap_zscore", 0.0),
            momentum_r2=features["state"].get("momentum_r2", 0.0),
            volatility_compression=features["state"].get("volatility_compression", 1.0),
            displacement=features["state"].get("displacement", 0.0),
            regime_state=regime_state,
            cpd_age=cpd_age,
            hmm_velocity=features["state"].get("hmm_velocity", 0.0),
            fractal_sync=features["state"].get("fractal_sync", 0.0),
        )

        # Lens 2 Predictive Probabilities
        scenario_probs = np.zeros(9, dtype=np.float32)
        if self.lens2_predictor is not None and self.lens2_scaler is not None:
             # Pre-calculate features for prediction
            lp = self.london_profile
            ny_after = self.ny_candles # evaluation window
            
            features_pred = [
                london_range := lp.range_points if lp else 0.0,
                lp.high if lp else 0.0,
                lp.low if lp else 0.0,
                ny_after["atr"].iloc[0] if "atr" in ny_after.columns and len(ny_after) > 0 else 1.0,
                self.london_geometry.get("london_slope", 0.0),
                self.london_geometry.get("london_r2", 0.0),
                self.london_geometry.get("london_pivot_density", 0.0),
                self.london_geometry.get("london_range_pos", 0.0),
                self.london_geometry.get("london_vol_bias", 0.0),
                self.london_geometry.get("london_momentum_ratio", 0.0),
                self.london_geometry.get("london_volatility_relative", 0.0),
            ]
            
            try:
                import torch
                x_scaled = self.lens2_scaler.transform([features_pred])
                x_tensor = torch.FloatTensor(x_scaled)
                with torch.no_grad():
                    scenario_probs = self.lens2_predictor.predict_proba(x_tensor).numpy()[0]
            except Exception as e:
                logger.error(f"Lens 3 env: predictive probability extraction failed: {e}")

        # Geometry Features (now 7 dims to fit 56-dim obs space correctly)
        geometry_vector = np.array([
            self.london_geometry.get("london_slope", 0.0),
            self.london_geometry.get("london_r2", 0.0),
            self.london_geometry.get("london_pivot_density", 0.0),
            self.london_geometry.get("london_range_pos", 0.0),
            self.london_geometry.get("london_vol_bias", 0.0),
            self.london_geometry.get("london_momentum_ratio", 0.0),
            self.london_geometry.get("london_volatility_relative", 0.0),
        ], dtype=np.float32)

        # Final Concatenation -> 40 (base) + 9 (probabilities) + 7 (geometry) = 56
        full_observation = np.concatenate([
            observation_base,
            scenario_probs,
            geometry_vector
        ])

        bundle = {"features": features, "observation": full_observation}
        self._step_cache_step = self.current_step
        self._step_cache_bundle = bundle
        return bundle

    def _get_observation(self) -> np.ndarray:
        """Build 40-dim observation vector."""
        bundle = self._get_step_bundle()
        if bundle is None:
            return np.zeros(self.obs_dim, dtype=np.float32)
        return bundle["observation"]

    def _get_current_features(self) -> Dict:
        """Get current step's SMC features."""
        bundle = self._get_step_bundle()
        if bundle is None:
            from src.session.features import _empty_features
            return _empty_features()
        return bundle["features"]

    def _open_position(self, action: int, price: float, atr_val: float, features: Dict) -> float:
        """Open a position with structural SL/TP."""
        trade_plan = self._build_trade_plan(action, price, atr_val, features)
        if trade_plan is None:
            return self.reward_calc.risk_violation_penalty

        is_first_trade = self.trades_this_episode == 0
        opened_in_ny_open = self.current_step < self.ny_open_reward_window_steps

        # Open position
        self.position = "long" if action == BUY else "short"
        self.entry_price = price
        self.sl_price = trade_plan["sl"]
        self.tp1_price = trade_plan["tp1"]
        self.tp2_price = trade_plan["tp2"]
        self.tp3_price = trade_plan["tp3"]
        self.position_size = 1.0
        self.trades_this_episode += 1
        self.current_trade_realized_pnl = 0.0
        self.current_trade_is_first = is_first_trade
        self.current_trade_opened_in_ny_open = opened_in_ny_open
        self.current_trade_entry_step = self.current_step
        self.current_trade_entry_time = self._get_timestamp_for_step(self.current_step)
        self.current_trade_matches_intent = self._opening_intent_matches_action(action)
        if opened_in_ny_open:
            self.episode_ny_open_trade_taken = True

        if is_first_trade:
            self.first_trade_entry_step = self.current_step
            self.first_trade_entry_time = (
                self.current_trade_entry_time.isoformat()
                if self.current_trade_entry_time is not None else None
            )
            self.first_trade_entry_delay_from_anchor_min = float(self.current_step * self.candle_minutes)
            self.first_trade_entry_delay_from_ny_open_min = self._minutes_since(
                self.ny_session_start_ts,
                self.current_trade_entry_time,
            )
            self.first_trade_is_ny_open = opened_in_ny_open
            self.first_trade_matches_intent = self.current_trade_matches_intent

        # SMC alignment bonus
        reward = self.reward_calc.compute_entry_reward(
            action,
            features,
            self.lens1_encoding,
            is_first_trade=is_first_trade,
            in_ny_open=opened_in_ny_open,
            opening_intent=self.opening_intent.to_dict(),
            opening_delay_steps=self.current_step,
            opening_window_steps=self.ny_open_reward_window_steps,
        )

        return reward

    def _build_trade_plan(self, action: int, price: float, atr_val: float, features: Dict):
        raw = features["raw"]
        state = features["state"]
        lp = self.london_profile

        if lp is None:
            return None

        london_range = max(lp.range_points, 0.01)
        buffer = self.sl_buffer_lr * london_range

        if self.df_m5 is not None and len(self.df_m5) >= 5:
            # Phase 10: Fractal Stop Loss (M5 Swings)
            recent_m5 = self.df_m5.iloc[-5:]
            m5_atr = recent_m5["atr"].iloc[-1] if "atr" in recent_m5.columns else (atr_val / 3.0)
            
            if action == BUY:
                sl = recent_m5["low"].min() - (m5_atr * 0.2)
            else:
                sl = recent_m5["high"].max() + (m5_atr * 0.2)
        else:
            if action == BUY:
                sl = raw.get("bull_ob_level", 0)
                if sl <= 0:
                    sl = lp.low
                sl -= buffer
            else:
                sl = raw.get("bear_ob_level", 0)
                if sl <= 0:
                    sl = lp.high
                sl += buffer

        sl_dist = abs(price - sl)
        # Ensure minimum SL distance (0.5 ATR for M5)
        min_sl_dist = 0.5 * atr_val if self.df_m5 is not None else self.sl_min_atr_gate * atr_val
        if sl_dist < min_sl_dist:
            sl = price - min_sl_dist if action == BUY else price + min_sl_dist
            sl_dist = abs(price - sl)

        # Phase 10: Volatility-Adjusted TP1 (0.5x ATR)
        if action == BUY:
            tp1 = price + (0.5 * atr_val)
            tp2 = lp.high if lp.high > tp1 else tp1 + (0.5 * atr_val)
            tp3 = raw.get("pdh", tp2 + atr_val) if raw.get("pdh", 0) > tp2 else tp2 + atr_val
        else:
            tp1 = price - (0.5 * atr_val)
            tp2 = lp.low if lp.low < tp1 else tp1 - (0.5 * atr_val)
            tp3 = raw.get("pdl", tp2 - atr_val) if raw.get("pdl", 0) < tp2 else tp2 - atr_val

        tp1_dist = abs(tp1 - price)
        rr = tp1_dist / max(sl_dist, 0.01)
        # min_rr check removed for training to break participation trap
        # if rr < min_rr: return None
        

        return {"sl": sl, "tp1": tp1, "tp2": tp2, "tp3": tp3}

    def _check_position(self, high: float, low: float, close: float, atr_val: float) -> float:
        """Check SL/TP for open position."""
        reward = 0.0

        if self.position == "long":
            # Check SL hit
            if low <= self.sl_price:
                reward = self._close_position(self.sl_price, "sl_hit")
            # Check TP1 (close 50%)
            elif high >= self.tp1_price and self.position_size > 0.5:
                pnl = (self.tp1_price - self.entry_price) * 0.5
                self.episode_pnl += pnl
                self.current_trade_realized_pnl += pnl
                self.position_size = 0.5
                # Spread-aware Breakeven (+1.5 pips on XAUUSD roughly 0.15 pts)
                self.sl_price = self.entry_price + 0.15
                r = pnl / max(abs(self.entry_price - self.sl_price), 0.01) if self.sl_price != self.entry_price else 1.0
                reward += max(r * 0.3, 0.1)
            # Check TP2 (close 0% - Trail instead)
            elif high >= self.tp2_price and self.position_size > 0.2:
                # We don't scale out at TP2, we shift to Trailing ATR
                self.sl_price = self.tp1_price # Lock in TP1 profit
                self.position_size = 0.2
                reward += 0.2
            # Check TP3 or trail
            elif high >= self.tp3_price and self.position_size > 0:
                reward = self._close_position(self.tp3_price, "tp3_hit")

        elif self.position == "short":
            # Check SL hit
            if high >= self.sl_price:
                reward = self._close_position(self.sl_price, "sl_hit")
            # Check TP1
            elif low <= self.tp1_price and self.position_size > 0.5:
                pnl = (self.entry_price - self.tp1_price) * 0.5
                self.episode_pnl += pnl
                self.current_trade_realized_pnl += pnl
                self.position_size = 0.5
                # Spread-aware Breakeven
                self.sl_price = self.entry_price - 0.15
                r = pnl / max(abs(self.entry_price - self.sl_price), 0.01) if self.sl_price != self.entry_price else 1.0
                reward += max(r * 0.3, 0.1)
            elif low <= self.tp2_price and self.position_size > 0.2:
                self.sl_price = self.tp1_price # Lock in TP1 profit
                self.position_size = 0.2
                reward += 0.2
            elif low <= self.tp3_price and self.position_size > 0:
                reward = self._close_position(self.tp3_price, "tp3_hit")

        return reward

    def _close_position(self, exit_price: float, reason: str) -> float:
        """Close position and compute R-multiple reward."""
        if self.position is None:
            return 0.0

        if self.position == "long":
            pnl = (exit_price - self.entry_price) * self.position_size
        else:
            pnl = (self.entry_price - exit_price) * self.position_size

        # R-multiple based on structural SL distance
        sl_dist = abs(self.entry_price - self.sl_price)
        r_multiple = pnl / max(sl_dist, 0.01) if sl_dist > 0 else 0.0

        self.episode_pnl += pnl
        self.current_trade_realized_pnl += pnl

        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1

        entry_step = self.current_trade_entry_step
        exit_step = self.current_step
        entry_time = self.current_trade_entry_time
        exit_time = self._get_timestamp_for_step(self.current_step)
        holding_steps = 0
        holding_minutes = 0.0
        entry_delay_anchor_min = np.nan
        entry_delay_ny_open_min = np.nan

        if entry_step is not None:
            holding_steps = max(exit_step - entry_step + 1, 1)
            holding_minutes = float(holding_steps * self.candle_minutes)
            entry_delay_anchor_min = float(entry_step * self.candle_minutes)
            entry_delay_ny_open_min = self._minutes_since(self.ny_session_start_ts, entry_time)

        self.trade_log.append({
            "direction": self.position,
            "entry": self.entry_price,
            "exit": exit_price,
            "sl": self.sl_price,
            "pnl": pnl,
            "r_multiple": r_multiple,
            "reason": reason,
            "date": self.current_date_str,
            "opened_in_ny_open": self.current_trade_opened_in_ny_open,
            "is_first_trade": int(self.current_trade_is_first),
            "entry_step": entry_step,
            "exit_step": exit_step,
            "holding_steps": holding_steps,
            "holding_minutes": holding_minutes,
            "entry_time": entry_time.isoformat() if entry_time is not None else None,
            "exit_time": exit_time.isoformat() if exit_time is not None else None,
            "entry_delay_from_anchor_min": entry_delay_anchor_min,
            "entry_delay_from_ny_open_min": entry_delay_ny_open_min,
            "matches_opening_intent": int(self.current_trade_matches_intent),
            "opening_scenario_label": self.opening_intent.scenario_label,
            "opening_scenario_confidence": float(self.opening_intent.scenario_confidence),
            "opening_setup_family": self.opening_intent.opening_setup_family,
            "opening_trade_expected": int(self.opening_intent.opening_trade_expected),
            "opening_side_bias": int(self.opening_intent.opening_side_bias),
            "opening_style_bias": float(self.opening_intent.opening_style_bias),
            "opening_aggression": float(self.opening_intent.opening_aggression),
            "opening_urgency": float(self.opening_intent.opening_urgency),
            "sweep_high_day": int(self.episode_sweep_high_seen),
            "sweep_low_day": int(self.episode_sweep_low_seen),
            "sweep_direction": self._episode_sweep_direction(),
        })

        if self.current_trade_is_first and self.first_trade_pnl is None:
            self.first_trade_pnl = self.current_trade_realized_pnl
            self.first_trade_duration_min = holding_minutes

        # Reset position
        self.position = None
        self.entry_price = 0.0
        self.sl_price = 0.0
        self.position_size = 1.0
        self.current_trade_realized_pnl = 0.0
        self.current_trade_is_first = False
        self.current_trade_opened_in_ny_open = False
        self.current_trade_entry_step = None
        self.current_trade_entry_time = None
        self.current_trade_matches_intent = False

        # Phase 10: Institutional Execution Shaping
        r_multiple_scaled = r_multiple * 0.5
        if holding_steps > 4 and pnl <= 0:
            # Time-in-Drawdown Penalty: "Institutional flow should work immediately"
            r_multiple_scaled -= 1.0
        elif holding_steps <= 3 and pnl > 0:
            # Precision Bonus: First-try accuracy out of the gate
            r_multiple_scaled += 1.0

        return r_multiple_scaled

    def _update_episode_context(self, features: Dict) -> None:
        raw = features.get("raw", {})
        if raw.get("sweep_high", 0) > 0:
            self.episode_sweep_high_seen = True
        if raw.get("sweep_low", 0) > 0:
            self.episode_sweep_low_seen = True

    def _episode_sweep_direction(self) -> str:
        if self.episode_sweep_high_seen and self.episode_sweep_low_seen:
            return "both"
        if self.episode_sweep_high_seen:
            return "high"
        if self.episode_sweep_low_seen:
            return "low"
        return "none"

    def _get_required_min_rr(self, action: int, raw: Dict, state: Dict) -> float:
        min_rr = self.min_rr
        if not self.sweep_asymmetry_mode:
            return min_rr

        if action == BUY and raw.get("sweep_low", 0) > 0:
            min_rr = max(min_rr, self.sweep_low_min_rr)
        elif action == SELL and raw.get("sweep_high", 0) > 0:
            choch_bear = state.get("choch_bear_recent", 0) > 0
            if (not self.sweep_high_requires_choch) or choch_bear:
                min_rr = max(min_rr, self.sweep_high_min_rr)

        return min_rr

    def _opening_intent_matches_action(self, action: int) -> bool:
        if not self.opening_intent.opening_trade_expected:
            return False
        if self.opening_intent.opening_side_bias > 0:
            return action == BUY
        if self.opening_intent.opening_side_bias < 0:
            return action == SELL
        return False

    def _get_timestamp_for_step(self, step: int):
        if self.ny_candles is None or len(self.ny_candles) == 0:
            return None
        idx = min(max(step, 0), len(self.ny_candles) - 1)
        return self.ny_candles.index[idx]

    def _infer_candle_minutes(self, candles=None) -> int:
        candles_ref = self.ny_candles if candles is None else candles
        if candles_ref is None or len(candles_ref.index) < 2:
            return 15
        delta = candles_ref.index[1] - candles_ref.index[0]
        minutes = int(delta.total_seconds() / 60.0)
        return max(minutes, 1)

    def _resolve_session_timestamp(self, date, hhmm: str, candles=None):
        try:
            date_obj = pd.Timestamp(date)
            ts = pd.Timestamp(f"{date_obj.date()} {hhmm}")
            tz = None
            candles_ref = self.ny_candles if candles is None else candles
            if candles_ref is not None and len(candles_ref.index) > 0:
                tz = candles_ref.index[0].tz
            if tz is not None and ts.tzinfo is None:
                ts = ts.tz_localize(tz)
            return ts
        except Exception:
            return None

    @staticmethod
    def _minutes_since(start_ts, end_ts):
        if start_ts is None or end_ts is None:
            return np.nan
        try:
            return float((end_ts - start_ts).total_seconds() / 60.0)
        except Exception:
            return np.nan

    def _build_info(self) -> Dict:
        return {
            "date": self.current_date_str,
            "news_day": self.is_news_day,
            "position": self.position,
            "trades": self.trades_this_episode,
            "pnl": self.episode_pnl,
            "encoding": self.lens1_encoding.copy(),
            "london_range": self.london_profile.range_points if self.london_profile else 0.0,
            "london_bias": self.london_profile.bias if self.london_profile else 0,
            "lens1_encoding_0": float(self.lens1_encoding[0]) if self.lens1_encoding is not None else 0.0,
            "sweep_day": int(self.episode_sweep_high_seen or self.episode_sweep_low_seen),
            "sweep_high_day": int(self.episode_sweep_high_seen),
            "sweep_low_day": int(self.episode_sweep_low_seen),
            "sweep_direction": self._episode_sweep_direction(),
            "ny_open_trade_taken": int(self.episode_ny_open_trade_taken),
            "first_trade_pnl": float(self.first_trade_pnl) if self.first_trade_pnl is not None else np.nan,
            "first_trade_win": int(self.first_trade_pnl is not None and self.first_trade_pnl > 0),
            "in_ny_open": int(self.current_step < self.ny_open_reward_window_steps),
            "opening_intent": self.opening_intent.to_dict(),
            "opening_scenario_id": int(self.current_scenario_id),
            "opening_scenario_label": SCENARIO_NAMES.get(self.current_scenario_id, "Undefined") if self.oracle_mode else self.opening_intent.scenario_label,
            "opening_scenario_confidence": 1.0 if self.oracle_mode else float(self.opening_intent.scenario_confidence),
            "opening_setup_family": "oracle" if self.oracle_mode else self.opening_intent.opening_setup_family,
            "opening_trade_expected": 1 if self.oracle_mode else int(self.opening_intent.opening_trade_expected),
            "opening_side_bias": self._get_oracle_bias() if self.oracle_mode else int(self.opening_intent.opening_side_bias),
            "opening_style_bias": 1.0 if self.oracle_mode else float(self.opening_intent.opening_style_bias),
            "opening_aggression": 1.0 if self.oracle_mode else float(self.opening_intent.opening_aggression),
            "opening_urgency": 1.0 if self.oracle_mode else float(self.opening_intent.opening_urgency),
            "first_trade_entry_step": self.first_trade_entry_step if self.first_trade_entry_step is not None else np.nan,
            "first_trade_entry_time": self.first_trade_entry_time,
            "first_trade_entry_delay_from_anchor_min": (
                float(self.first_trade_entry_delay_from_anchor_min)
                if self.first_trade_entry_delay_from_anchor_min is not None else np.nan
            ),
            "first_trade_entry_delay_from_ny_open_min": (
                float(self.first_trade_entry_delay_from_ny_open_min)
                if self.first_trade_entry_delay_from_ny_open_min is not None else np.nan
            ),
            "first_trade_duration_min": (
                float(self.first_trade_duration_min)
                if self.first_trade_duration_min is not None else np.nan
            ),
            "first_trade_is_ny_open": int(self.first_trade_is_ny_open),
            "first_trade_matches_intent": int(self.first_trade_matches_intent),
        }
