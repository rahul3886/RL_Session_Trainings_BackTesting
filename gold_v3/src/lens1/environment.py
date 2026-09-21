"""
Lens 1 RL Environment — London Session Observer.

Lens 1 observes the London session candle-by-candle and learns
to produce an 8-dim continuous encoding that predicts NY behaviour.
It does NOT trade. Its only job is to observe and encode.

Episode structure:
  OBSERVATION PHASE (08:00–12:15 UTC): receive London candles, output attention
  EVALUATION PHASE (12:15–21:00 UTC): encoding evaluated against NY reality
"""

import logging
from typing import Optional, Dict, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding

from src.session.profiler import (
    get_london_until_anchor, get_ny_after_anchor,
    compute_session_profile, get_session_pair,
)
from src.session.features import extract_session_features, build_lens1_state_vector
from src.data.indicators import add_indicators
from src.lens1.reward import Lens1RewardCalculator
from src.smc.liquidity import build_pdh_pdl_cache

logger = logging.getLogger(__name__)


class Lens1LondonEnv(gym.Env):
    """
    Lens 1 London Observer Environment.

    The agent observes London candles and produces a continuous
    8-dim encoding vector. Rewards come from NY session outcomes.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        trading_days: list,
        settings=None,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.df = df
        self.trading_days = list(trading_days)
        self._shuffled_days = list(trading_days)
        self.settings = settings
        self.render_mode = render_mode
        self._pdh_pdl_cache = build_pdh_pdl_cache(df)

        # Config
        self.obs_dim = 31 if settings is None else settings.lens1.obs_dim
        self.encoding_dim = 8 if settings is None else settings.lens1.encoding_dim

        session_cfg = None if settings is None else settings.sessions
        self.london_start = "08:00" if session_cfg is None else session_cfg.london_start
        self.london_end = "17:00" if session_cfg is None else session_cfg.london_end
        self.ny_start = "12:00" if session_cfg is None else session_cfg.ny_start
        self.ny_end = "21:00" if session_cfg is None else session_cfg.ny_end
        self.anchor_time = "12:15" if session_cfg is None else session_cfg.lens1_anchor

        smc_cfg = None if settings is None else settings.smc

        # Spaces
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.encoding_dim,),
            dtype=np.float32,
        )

        # Episode state
        self.current_day_idx = 0
        self.current_step = 0
        self.phase = "observation"  # "observation" or "evaluation"

        # Session data for current episode
        self.london_candles = None
        self.ny_candles = None
        self.london_profile = None
        self.ny_profile = None
        self.london_atr = None

        # Lens 1 cached encoding (output at anchor time)
        self.cached_encoding = np.zeros(self.encoding_dim, dtype=np.float32)

        # Reward calculator
        reward_cfg = None if settings is None else settings
        self.reward_calc = Lens1RewardCalculator(reward_cfg)

        # Observation candle index (steps through London candles)
        self.obs_candle_idx = 0
        self.eval_candle_idx = 0

        # Total london and NY candle counts
        self.london_candle_count = 0
        self.ny_candle_count = 0

    def seed(self, seed=None):
        """Compatibility shim for vectorized env wrappers."""
        self.np_random, actual_seed = seeding.np_random(seed)
        import random
        random.seed(actual_seed)
        np.random.seed(actual_seed)
        return [actual_seed]

    def reset(self, seed=None, options=None):
        """Reset to a new trading day."""
        super().reset(seed=seed)

        # Select trading day
        if self.current_day_idx >= len(self._shuffled_days):
            self.current_day_idx = 0
            import random
            random.shuffle(self._shuffled_days)

        date = self._shuffled_days[self.current_day_idx]
        self.current_day_idx += 1

        # Get session data
        london, ny, self.london_profile, self.ny_profile = get_session_pair(
            self.df, date,
            self.settings.sessions if self.settings else None,
        )

        # Get London candles until anchor
        self.london_candles = get_london_until_anchor(
            self.df, date, self.london_start, self.anchor_time
        )
        self.ny_candles = get_ny_after_anchor(
            self.df, date, self.anchor_time, self.ny_end
        )

        # ATR series
        if "atr" in self.df.columns:
            self.london_atr = self.df.loc[self.london_candles.index, "atr"] if len(self.london_candles) > 0 else pd.Series([1.0])
            self.ny_atr = self.df.loc[self.ny_candles.index, "atr"] if len(self.ny_candles) > 0 else pd.Series([1.0])
        else:
            self.london_atr = pd.Series(np.ones(len(self.london_candles)), index=self.london_candles.index)
            self.ny_atr = pd.Series(np.ones(len(self.ny_candles)), index=self.ny_candles.index)

        self.london_candle_count = len(self.london_candles)
        self.ny_candle_count = len(self.ny_candles)

        # Reset episode state
        self.current_step = 0
        self.obs_candle_idx = 0
        self.eval_candle_idx = 0
        self.phase = "observation"
        self.cached_encoding = np.zeros(self.encoding_dim, dtype=np.float32)

        # Reset reward calculator
        self.reward_calc.reset(self.london_profile, self.ny_candles, self.ny_atr)

        obs = self._get_observation()
        info = {"phase": self.phase, "date": str(date)}

        return obs, info

    def step(self, action: np.ndarray):
        """
        Step through the episode.

        During observation phase: action is the encoding attempt (no reward).
        During evaluation phase: cached encoding is evaluated against NY.
        """
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        reward = 0.0
        terminated = False
        truncated = False

        if self.phase == "observation":
            # Agent is observing London candles
            self.cached_encoding = action.copy()
            self.obs_candle_idx += 1

            # Check if observation phase is over
            if self.obs_candle_idx >= self.london_candle_count:
                self.phase = "evaluation"
                self.eval_candle_idx = 0

        elif self.phase == "evaluation":
            # Agent's encoding is being evaluated against NY
            self.eval_candle_idx += 1

            # Calculate reward based on NY candle
            reward = self.reward_calc.compute_step_reward(
                self.cached_encoding,
                self.eval_candle_idx,
                self.ny_candle_count,
            )

            # Check if episode is over
            if self.eval_candle_idx >= self.ny_candle_count:
                # End-of-episode reward
                bonus = self.reward_calc.compute_episode_end_reward(
                    self.cached_encoding,
                )
                reward += bonus
                terminated = True

        self.current_step += 1
        obs = self._get_observation()
        info = {
            "phase": self.phase,
            "step": self.current_step,
            "encoding": self.cached_encoding.copy(),
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Build the 31-dim observation vector."""
        if self.phase == "observation":
            if self.obs_candle_idx < self.london_candle_count and self.london_candle_count > 0:
                # Progressive London candle window
                end_idx = min(self.obs_candle_idx + 1, self.london_candle_count)
                candle_window = self.london_candles.iloc[:end_idx]
                atr_window = self.london_atr.iloc[:end_idx]

                features = extract_session_features(
                    candle_window, atr_window,
                    full_df=self.df,
                    smc_config=self.settings.smc if self.settings else None,
                    pdh_pdl_cache=self._pdh_pdl_cache,
                )

                # Time normalisation: minutes into London / total London minutes
                time_norm = self.obs_candle_idx / max(self.london_candle_count, 1)
                return build_lens1_state_vector(features, time_norm)
            else:
                return np.zeros(self.obs_dim, dtype=np.float32)

        elif self.phase == "evaluation":
            # During evaluation, show NY candles progressively
            if self.eval_candle_idx < self.ny_candle_count and self.ny_candle_count > 0:
                end_idx = min(self.eval_candle_idx + 1, self.ny_candle_count)
                candle_window = self.ny_candles.iloc[:end_idx]
                atr_window = self.ny_atr.iloc[:end_idx]

                features = extract_session_features(
                    candle_window, atr_window,
                    full_df=self.df,
                    session_high=self.london_profile.high if self.london_profile else None,
                    session_low=self.london_profile.low if self.london_profile else None,
                    smc_config=self.settings.smc if self.settings else None,
                    pdh_pdl_cache=self._pdh_pdl_cache,
                )

                time_norm = self.eval_candle_idx / max(self.ny_candle_count, 1)
                return build_lens1_state_vector(features, time_norm)
            else:
                return np.zeros(self.obs_dim, dtype=np.float32)

        return np.zeros(self.obs_dim, dtype=np.float32)
