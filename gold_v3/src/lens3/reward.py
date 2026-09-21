"""
Lens 2 Reward Function.

Carries forward all v2 fixes + Lens 1 alignment bonus.
Penalises passive holding, rewards SMC confluence entries.
"""

import logging
import numpy as np
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class Lens3RewardCalculator:
    """
    Computes rewards for Lens 2 trading actions.
    """

    def __init__(self, settings=None):
        lens3_cfg = None if settings is None else settings.lens3
        if settings is not None:
            cfg = settings.reward
            self.correct_hold = cfg.correct_hold
            self.missed_setup_penalty = cfg.missed_setup_penalty
            self.smc_alignment_bonus = cfg.smc_alignment_bonus
            self.confluence_bonus = cfg.confluence_bonus
            self.confluence_max = cfg.confluence_max
            self.risk_violation_penalty = cfg.risk_violation_penalty
            self.overtrading_penalty = cfg.overtrading_penalty
            self.lens1_align_bonus = cfg.lens1_align_bonus
            self.lens1_align_penalty = cfg.lens1_align_penalty
            self.episode_pnl_bonus = cfg.episode_pnl_bonus
            self.episode_winrate_bonus = cfg.episode_winrate_bonus
            self.episode_no_trade_penalty = cfg.episode_no_trade_penalty
            self.episode_overtrade_penalty = getattr(cfg, 'episode_overtrade_penalty', -0.20)
            self.participation_bonus = getattr(cfg, 'participation_bonus', 0.0)
            self.progressive_no_trade_step_penalty = getattr(cfg, 'progressive_no_trade_step_penalty', -0.015)
        else:
            self.correct_hold = 0.0
            self.missed_setup_penalty = -0.15
            self.smc_alignment_bonus = 0.30
            self.confluence_bonus = 0.10
            self.confluence_max = 0.60
            self.risk_violation_penalty = -0.10
            self.overtrading_penalty = -0.30
            self.lens1_align_bonus = 0.20
            self.lens1_align_penalty = -0.20
            self.episode_pnl_bonus = 0.50
            self.episode_winrate_bonus = 0.30
            self.episode_no_trade_penalty = -0.80
            self.episode_overtrade_penalty = -0.20
            self.participation_bonus = 0.0
            self.progressive_no_trade_step_penalty = -0.015

        self.sweep_asymmetry_mode = True if lens3_cfg is None else lens3_cfg.sweep_asymmetry_mode
        self.sweep_low_entry_confidence = 1.0 if lens3_cfg is None else lens3_cfg.sweep_low_entry_confidence
        self.sweep_high_entry_confidence = 0.6 if lens3_cfg is None else lens3_cfg.sweep_high_entry_confidence
        self.sweep_high_requires_choch = True if lens3_cfg is None else lens3_cfg.sweep_high_requires_choch
        self.ny_open_entry_bonus = 0.05 if lens3_cfg is None else lens3_cfg.ny_open_entry_bonus
        self.opening_intent_enabled = True if lens3_cfg is None else lens3_cfg.opening_intent_enabled
        self.opening_intent_min_confidence = 0.20 if lens3_cfg is None else lens3_cfg.opening_intent_min_confidence
        self.opening_intent_bonus = 0.20 if lens3_cfg is None else lens3_cfg.opening_intent_bonus
        self.opening_intent_missed_penalty = -0.12 if lens3_cfg is None else lens3_cfg.opening_intent_missed_penalty
        self.opening_intent_opposite_penalty = -0.20 if lens3_cfg is None else lens3_cfg.opening_intent_opposite_penalty
        self.opening_intent_first_trade_only = True if lens3_cfg is None else lens3_cfg.opening_intent_first_trade_only
        self.opening_intent_urgency_bonus = 0.18 if lens3_cfg is None else lens3_cfg.opening_intent_urgency_bonus
        self.opening_intent_delay_penalty = -0.04 if lens3_cfg is None else lens3_cfg.opening_intent_delay_penalty
        self.no_trade_feasible_hold_penalty = -0.08 if lens3_cfg is None else getattr(lens3_cfg, "no_trade_feasible_hold_penalty", -0.08)
        self.no_trade_feasible_hold_penalty_growth = 1.0 if lens3_cfg is None else getattr(lens3_cfg, "no_trade_feasible_hold_penalty_growth", 1.0)

    def reset(self):
        """Reset per-episode state."""
        pass

    def compute_hold_reward(
        self,
        features: Dict,
        lens1_encoding: np.ndarray,
        has_position: bool,
        opening_intent: Optional[Dict] = None,
        is_first_trade_pending: bool = False,
        in_ny_open: bool = False,
        opening_delay_steps: int = 0,
        opening_window_steps: int = 4,
        trades_this_episode: int = 0,
        current_step: int = 0,
        trade_plan_feasible: bool = False,
    ) -> float:
        """
        Compute reward for HOLD action.

        Zero baseline for hold. Progressive penalty when no trade has been taken.
        Missed-setup penalty only fires when the trade plan is actually feasible.
        """
        if has_position:
            return 0.0  # holding while in position is neutral

        reward = self.correct_hold

        # Progressive no-trade step penalty: grows with time if agent hasn't traded
        if trades_this_episode == 0 and current_step >= opening_window_steps:
            reward += self.progressive_no_trade_step_penalty
            if trade_plan_feasible:
                post_open_steps = max(current_step - opening_window_steps, 0)
                growth = 1.0 + min(post_open_steps / max(opening_window_steps, 1), 2.0) * self.no_trade_feasible_hold_penalty_growth
                reward += self.no_trade_feasible_hold_penalty * growth

        # Check for missed setups — only penalise if the trade could actually be placed
        if trade_plan_feasible and self._is_setup_present(
            features,
            lens1_encoding,
            opening_intent=opening_intent,
            is_first_trade_pending=is_first_trade_pending,
            in_ny_open=in_ny_open,
        ):
            reward += self.missed_setup_penalty

        if self._opening_trade_ready(
            features,
            opening_intent,
            is_first_trade_pending=is_first_trade_pending,
            in_ny_open=in_ny_open,
        ):
            reward += self.opening_intent_missed_penalty * max(1.0, self._opening_strength(opening_intent))
        elif self._intent_active(opening_intent, is_first_trade_pending, in_ny_open):
            delay_scale = min(opening_delay_steps / max(opening_window_steps, 1), 1.0)
            reward += self.opening_intent_delay_penalty * self._opening_strength(opening_intent) * delay_scale

        return reward

    def estimate_entry_quality(self, action: int, features: Dict) -> tuple[float, float]:
        """Shared confluence score used by reward shaping and anti-hold action forcing."""
        s = features.get("state", {})
        raw = features.get("raw", {})
        sweep_low_live = raw.get("sweep_low", 0.0)
        sweep_high_live = raw.get("sweep_high", 0.0)
        confirmed_high_sweep = self._is_confirmed_high_sweep(sweep_high_live, s)

        confluence_count = 0.0
        entry_confidence = 1.0

        if action == 1:  # BUY
            if s.get("bull_ob_dist", 5.0) < 1.0:
                confluence_count += 1
            if s.get("bull_fvg_unfilled", 0) > 0 and s.get("bull_fvg_dist", 5.0) < 1.5:
                confluence_count += 1
            if s.get("choch_bull_recent", 0) > 0:
                confluence_count += 1
            if s.get("mss_bull_recent", 0) > 0:
                confluence_count += 1
            if sweep_low_live > 0:
                entry_confidence = self.sweep_low_entry_confidence
                confluence_count += 2
        elif action == 2:  # SELL
            if s.get("bear_ob_dist", 5.0) < 1.0:
                confluence_count += 1
            if s.get("bear_fvg_unfilled", 0) > 0 and s.get("bear_fvg_dist", 5.0) < 1.5:
                confluence_count += 1
            if s.get("choch_bear_recent", 0) > 0:
                confluence_count += 1
            if s.get("mss_bear_recent", 0) > 0:
                confluence_count += 1
            if confirmed_high_sweep:
                entry_confidence = self.sweep_high_entry_confidence
                confluence_count += 2

        return confluence_count, entry_confidence

    def compute_entry_reward(
        self,
        action: int,
        features: Dict,
        lens1_encoding: np.ndarray,
        is_first_trade: bool = False,
        in_ny_open: bool = False,
        opening_intent: Optional[Dict] = None,
        opening_delay_steps: int = 0,
        opening_window_steps: int = 4,
    ) -> float:
        """
        Compute reward for BUY/SELL entry.
        """
        reward = self.participation_bonus  # Reward the act of trading
        s = features.get("state", {})
        raw = features.get("raw", {})
        sweep_low_live = raw.get("sweep_low", 0.0)
        sweep_high_live = raw.get("sweep_high", 0.0)
        confirmed_high_sweep = self._is_confirmed_high_sweep(sweep_high_live, s)

        # ── SMC Alignment Bonus ──
        confluence_count, entry_confidence = self.estimate_entry_quality(action, features)

        if action == 1:  # BUY
            # Lens 1 alignment
            if lens1_encoding[0] > 0.2:
                reward += self.lens1_align_bonus
            elif lens1_encoding[0] < -0.2:
                # Counter-trend buys are only acceptable after a live sweep low.
                if sweep_low_live > 0:
                    reward += self.lens1_align_bonus * 0.5 * entry_confidence
                else:
                    reward += self.lens1_align_penalty

        elif action == 2:  # SELL
            # Lens 1 alignment
            if lens1_encoding[0] < -0.2:
                reward += self.lens1_align_bonus
            elif lens1_encoding[0] > 0.2:
                if confirmed_high_sweep:
                    reward += self.lens1_align_bonus * 0.5 * entry_confidence
                else:
                    reward += self.lens1_align_penalty

        # Confluence bonus (capped)
        conf_reward = min(confluence_count * self.confluence_bonus, self.confluence_max) * entry_confidence
        reward += conf_reward

        # Base SMC alignment if any confluence
        if confluence_count >= 2:
            reward += self.smc_alignment_bonus * entry_confidence

        # Favor the first qualified NY-open trade without forcing bad entries.
        if is_first_trade and in_ny_open and confluence_count >= 2:
            reward += self.ny_open_entry_bonus * entry_confidence

        reward += self._opening_intent_entry_adjustment(
            action,
            features,
            opening_intent,
            is_first_trade=is_first_trade,
            in_ny_open=in_ny_open,
        )

        if self._intent_active(opening_intent, is_first_trade, in_ny_open):
            delay_scale = 1.0 - min(opening_delay_steps / max(opening_window_steps - 1, 1), 1.0)
            if self._action_matches_intent(action, int(opening_intent.get("opening_side_bias", 0))):
                reward += self.opening_intent_urgency_bonus * self._opening_strength(opening_intent) * max(delay_scale, 0.25)

        return reward

    def compute_episode_end_reward(
        self,
        total_pnl: float,
        num_trades: int,
        wins: int,
        losses: int,
        max_trades: int,
    ) -> float:
        """Compute end-of-episode reward."""
        reward = 0.0

        # PnL bonus
        if total_pnl > 0:
            reward += self.episode_pnl_bonus

        # Win rate bonus
        total = wins + losses
        if total > 0 and (wins / total) > 0.6:
            reward += self.episode_winrate_bonus

        # No trade penalty
        if num_trades == 0:
            reward += self.episode_no_trade_penalty

        # Overtrading penalty
        if num_trades > max_trades:
            reward += self.episode_overtrade_penalty * (num_trades - max_trades)

        return reward

    def _is_setup_present(
        self,
        features: Dict,
        lens1_encoding: np.ndarray,
        opening_intent: Optional[Dict] = None,
        is_first_trade_pending: bool = False,
        in_ny_open: bool = False,
    ) -> bool:
        """Check if a valid SMC setup exists (for missed setup penalty)."""
        s = features.get("state", {})
        raw = features.get("raw", {})

        # Bull OB + bullish Lens 1
        if s.get("bull_ob_dist", 5.0) < 1.0 and lens1_encoding[0] > 0.2:
            return True

        # Bear OB + bearish Lens 1
        if s.get("bear_ob_dist", 5.0) < 1.0 and lens1_encoding[0] < -0.2:
            return True

        # Bull FVG near London Low
        if s.get("bull_fvg_unfilled", 0) > 0 and s.get("dist_to_session_low", 5.0) < 1.0:
            return True

        # Bear FVG near London High
        if s.get("bear_fvg_unfilled", 0) > 0 and s.get("dist_to_session_high", 5.0) < 1.0:
            return True

        # Sweep detected (highest priority)
        if raw.get("sweep_low", 0) > 0:
            return True

        if self._is_confirmed_high_sweep(raw.get("sweep_high", 0), s):
            return True

        if self._opening_trade_ready(
            features,
            opening_intent,
            is_first_trade_pending=is_first_trade_pending,
            in_ny_open=in_ny_open,
        ):
            return True

        return False

    def _is_confirmed_high_sweep(self, sweep_high: float, state: Dict) -> bool:
        """High-side sweeps need bearish confirmation before we score them as premium setups."""
        if sweep_high <= 0:
            return False
        if not self.sweep_asymmetry_mode:
            return True
        if not self.sweep_high_requires_choch:
            return True
        return state.get("choch_bear_recent", 0) > 0

    def _opening_trade_ready(
        self,
        features: Dict,
        opening_intent: Optional[Dict],
        is_first_trade_pending: bool,
        in_ny_open: bool,
    ) -> bool:
        if not self._intent_active(opening_intent, is_first_trade_pending, in_ny_open):
            return False

        s = features.get("state", {})
        raw = features.get("raw", {})
        side_bias = int(opening_intent.get("opening_side_bias", 0))
        setup_family = opening_intent.get("opening_setup_family", "mixed")

        if side_bias > 0:
            bull_structure = (
                s.get("bull_ob_dist", 5.0) < 1.0
                or (s.get("bull_fvg_unfilled", 0) > 0 and s.get("bull_fvg_dist", 5.0) < 1.5)
            )
            bull_trigger = s.get("choch_bull_recent", 0) > 0 or s.get("mss_bull_recent", 0) > 0
            if setup_family == "sweep_low":
                return raw.get("sweep_low", 0) > 0 or (bull_structure and bull_trigger)
            return bull_structure and bull_trigger

        if side_bias < 0:
            bear_structure = (
                s.get("bear_ob_dist", 5.0) < 1.0
                or (s.get("bear_fvg_unfilled", 0) > 0 and s.get("bear_fvg_dist", 5.0) < 1.5)
            )
            bear_trigger = s.get("choch_bear_recent", 0) > 0 or s.get("mss_bear_recent", 0) > 0
            if setup_family == "sweep_high":
                return self._is_confirmed_high_sweep(raw.get("sweep_high", 0), s)
            return bear_structure and bear_trigger

        return False

    def _opening_intent_entry_adjustment(
        self,
        action: int,
        features: Dict,
        opening_intent: Optional[Dict],
        is_first_trade: bool,
        in_ny_open: bool,
    ) -> float:
        if not self._intent_active(opening_intent, is_first_trade, in_ny_open):
            return 0.0

        side_bias = int(opening_intent.get("opening_side_bias", 0))
        confidence = float(opening_intent.get("scenario_confidence", 0.0))
        aggression = float(opening_intent.get("opening_aggression", 0.0))
        strength = self._opening_strength(opening_intent)
        if side_bias == 0 or strength <= 0:
            return 0.0

        if self._action_matches_intent(action, side_bias):
            if self._opening_trade_ready(features, opening_intent, is_first_trade, in_ny_open):
                return self.opening_intent_bonus * strength
            return 0.0

        return self.opening_intent_opposite_penalty * strength

    def _intent_active(
        self,
        opening_intent: Optional[Dict],
        is_first_trade_pending: bool,
        in_ny_open: bool,
    ) -> bool:
        if not self.opening_intent_enabled:
            return False
        if opening_intent is None:
            return False
        if not in_ny_open:
            return False
        if self.opening_intent_first_trade_only and not is_first_trade_pending:
            return False
        if int(opening_intent.get("opening_trade_expected", 0)) <= 0:
            return False
        return self._opening_strength(opening_intent) >= self.opening_intent_min_confidence

    @staticmethod
    def _action_matches_intent(action: int, side_bias: int) -> bool:
        if side_bias > 0:
            return action == 1
        if side_bias < 0:
            return action == 2
        return False

    @staticmethod
    def _opening_strength(opening_intent: Optional[Dict]) -> float:
        if opening_intent is None:
            return 0.0
        confidence = float(opening_intent.get("scenario_confidence", 0.0))
        aggression = float(opening_intent.get("opening_aggression", 0.0))
        urgency = float(opening_intent.get("opening_urgency", 0.0))
        return max(confidence, aggression, urgency)
