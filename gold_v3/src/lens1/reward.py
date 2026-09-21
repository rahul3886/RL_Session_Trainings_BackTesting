"""
Lens 1 Reward Function.

Lens 1 is rewarded based on how accurately its encoding
predicted NY session behaviour. Rewards are given RETROSPECTIVELY
as NY unfolds. The key insight: the reward signal comes from NY,
forcing the model to learn information predictive of NY.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.lens1.opening_targets import infer_opening_outcome
from src.smc.liquidity import detect_sweep

logger = logging.getLogger(__name__)


class Lens1RewardCalculator:
    """
    Computes rewards for Lens 1 based on NY session outcomes.

        5 reward components:
          1. Directional accuracy (weight 1.0)
          2. Level respect (weight 0.5)
          3. Sweep detection (config-driven weight)
          4. Continuation accuracy (weight 0.6)
          5. Ranging session expansion (weight 0.4)
          6. Opening side bias (dim 5)
          7. Opening style bias (dim 6)
          8. Opening urgency (dim 7)
    """

    def __init__(self, settings=None):
        # Reward weights
        if settings is not None:
            cfg = settings.lens1
            self.w_direction = cfg.reward_direction_weight
            self.w_level = cfg.reward_level_weight
            self.w_sweep = cfg.reward_sweep_weight
            self.w_continuation = cfg.reward_continuation_weight
            self.w_ranging = cfg.reward_ranging_weight
            self.w_opening_bias = cfg.reward_opening_bias_weight
            self.w_opening_style = cfg.reward_opening_style_weight
            self.w_opening_urgency = cfg.reward_opening_urgency_weight
            self.opening_window_candles = cfg.opening_target_window_candles
            self.sweep_threshold_lr = settings.smc.sweep_min_lr
        else:
            self.w_direction = 1.0
            self.w_level = 0.5
            self.w_sweep = 5.0
            self.w_continuation = 0.6
            self.w_ranging = 0.4
            self.w_opening_bias = 0.8
            self.w_opening_style = 0.7
            self.w_opening_urgency = 0.6
            self.opening_window_candles = 4
            self.sweep_threshold_lr = 0.12

        # Episode state
        self.london_profile = None
        self.ny_candles = None
        self.ny_atr = None
        self.direction_rewarded = False
        self.sweep_high_rewarded = False
        self.sweep_low_rewarded = False
        self.opening_targets_rewarded = False
        self.opening_outcome = None

    def reset(self, london_profile, ny_candles, ny_atr):
        """Reset reward state for new episode."""
        self.london_profile = london_profile
        self.ny_candles = ny_candles
        self.ny_atr = ny_atr
        self.direction_rewarded = False
        self.sweep_high_rewarded = False
        self.sweep_low_rewarded = False
        self.opening_targets_rewarded = False
        self.opening_outcome = infer_opening_outcome(
            london_profile,
            ny_candles,
            sweep_threshold_lr=self.sweep_threshold_lr,
            window_candles=self.opening_window_candles,
        )

    def compute_step_reward(
        self,
        encoding: np.ndarray,
        eval_step: int,
        total_ny_candles: int,
    ) -> float:
        """
        Compute reward for current NY evaluation step.

        Args:
            encoding: Lens 1's cached 8-dim encoding
            eval_step: current step in evaluation phase
            total_ny_candles: total NY candles in episode
        """
        if self.london_profile is None or self.ny_candles is None:
            return 0.0
        if eval_step <= 0 or eval_step > len(self.ny_candles):
            return 0.0

        reward = 0.0
        lp = self.london_profile

        # Current NY state
        ny_so_far = self.ny_candles.iloc[:eval_step]
        current_price = ny_so_far["close"].iloc[-1]
        ny_high_so_far = ny_so_far["high"].max()
        ny_low_so_far = ny_so_far["low"].min()

        london_mid = lp.mid
        london_high = lp.high
        london_low = lp.low

        # ── 1. DIRECTIONAL ACCURACY (after first 2 NY candles) ──
        if eval_step >= 2 and not self.direction_rewarded:
            ny_direction = 1.0 if current_price > london_mid else -1.0
            encoding_direction = encoding[0]  # primary bias dimension

            if np.sign(encoding_direction) == np.sign(ny_direction):
                reward += 0.5 * self.w_direction
            else:
                reward -= 0.3 * self.w_direction

            self.direction_rewarded = True

        # ── 2. LEVEL RESPECT ──
        london_range = max(lp.range_points, 0.01)
        level_touch_threshold = 0.10 * london_range
        # Price near London High
        if abs(current_price - london_high) < level_touch_threshold:
            if encoding[1] > 0.3:  # encoded resistance
                reward += 0.3 * self.w_level
            elif encoding[1] < -0.3:  # encoded as non-significant
                reward -= 0.2 * self.w_level

        # Price near London Low
        if abs(current_price - london_low) < level_touch_threshold:
            if encoding[2] > 0.3:  # encoded support
                reward += 0.3 * self.w_level
            elif encoding[2] < -0.3:
                reward -= 0.2 * self.w_level

        # ── 3. SWEEP DETECTION ──
        sweep_state = detect_sweep(
            current_price=current_price,
            session_high=london_high,
            session_low=london_low,
            candles_after=ny_so_far,
            threshold_lr=self.sweep_threshold_lr,
            require_reversal=True,
        )

        # Sweep high: NY spikes above London High and reverses
        if not self.sweep_high_rewarded and sweep_state["sweep_high"]:
            if encoding[3] > 0.5:
                reward += 1.0 * self.w_sweep
            else:
                reward -= 0.5 * self.w_sweep
            self.sweep_high_rewarded = True

        # Sweep low: NY spikes below London Low and reverses
        if not self.sweep_low_rewarded and sweep_state["sweep_low"]:
            if encoding[4] > 0.5:
                reward += 1.0 * self.w_sweep
            else:
                reward -= 0.5 * self.w_sweep
            self.sweep_low_rewarded = True

        opening_window = min(self.opening_window_candles, len(self.ny_candles))
        if opening_window > 0 and eval_step >= opening_window and not self.opening_targets_rewarded:
            reward += self._compute_opening_target_reward(encoding)
            self.opening_targets_rewarded = True

        return reward

    def compute_episode_end_reward(self, encoding: np.ndarray) -> float:
        """
        Compute end-of-episode reward components.

        Called at NY session close (21:00 UTC).
        """
        if self.london_profile is None or self.ny_candles is None or len(self.ny_candles) == 0:
            return 0.0

        reward = 0.0
        lp = self.london_profile
        ny_close = self.ny_candles["close"].iloc[-1]
        london_open = lp.open_price

        # ── 4. CONTINUATION ACCURACY ──
        if lp.bias == 1 and ny_close > london_open:
            # Bullish continuation occurred
            if encoding[0] > 0.3:
                reward += 0.5 * self.w_continuation
        elif lp.bias == -1 and ny_close < london_open:
            # Bearish continuation occurred
            if encoding[0] < -0.3:
                reward += 0.5 * self.w_continuation

        # ── 5. RANGING SESSION ACCURACY ──
        london_range = lp.range_points
        if london_range < 30:  # Ranging London
            ny_range = self.ny_candles["high"].max() - self.ny_candles["low"].min()
            if ny_range > london_range * 1.5:  # NY expanded
                ny_direction = 1.0 if ny_close > london_open else -1.0
                # Lens 1 should have encoded expansion direction
                if np.sign(encoding[0]) == np.sign(ny_direction):
                    reward += 0.4 * self.w_ranging
                else:
                    reward -= 0.2 * self.w_ranging

        return reward

    def _compute_opening_target_reward(self, encoding: np.ndarray) -> float:
        if self.opening_outcome is None:
            return 0.0

        target_bias, target_style, target_urgency = self.opening_outcome.to_targets()
        reward = 0.0
        reward += self.w_opening_bias * self._score_signed_dimension(encoding[5], target_bias)
        reward += self.w_opening_style * self._score_signed_dimension(encoding[6], target_style)
        reward += self.w_opening_urgency * self._score_positive_dimension(encoding[7], target_urgency)
        return reward

    @staticmethod
    def _score_signed_dimension(prediction: float, target: float) -> float:
        if abs(target) < 0.1:
            if abs(prediction) < 0.25:
                return 0.35
            return -0.20 * min(abs(prediction), 1.0)

        sign_reward = 0.45 if np.sign(prediction) == np.sign(target) else -0.35
        closeness = 1.0 - min(abs(prediction - target), 2.0) / 2.0
        return sign_reward + (0.55 * closeness)

    @staticmethod
    def _score_positive_dimension(prediction: float, target: float) -> float:
        target = float(np.clip(target, 0.0, 1.0))
        prediction = float(np.clip(prediction, 0.0, 1.0))
        if target <= 0.05:
            if prediction <= 0.20:
                return 0.30
            return -0.25 * min(prediction, 1.0)
        return 1.0 - min(abs(prediction - target), 1.0)
