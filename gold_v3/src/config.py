"""
Typed configuration loader for settings.yaml.
All settings flow through dataclasses — no magic strings in code.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path


@dataclass
class DataConfig:
    instrument: str = "xauusd"
    timeframes: List[str] = field(default_factory=lambda: ["M5", "M15"])
    train_start: str = "2022-01-01"
    train_end: str = "2024-01-01"
    validate_start: str = "2024-01-01"
    validate_end: str = "2024-07-01"
    test_start: str = "2024-07-01"
    test_end: str = "2025-01-01"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    news_calendar_path: str = "data/news_calendar.csv"
    mt5_symbol: str = "XAUUSD"
    mt5_login: Optional[int] = None
    mt5_server: Optional[str] = None


@dataclass
class SessionConfig:
    london_start: str = "08:00"
    london_end: str = "17:00"
    ny_start: str = "12:00"
    ny_end: str = "21:00"
    overlap_start: str = "12:00"
    overlap_end: str = "17:00"
    lens1_anchor: str = "12:15"
    ny_no_trade: str = "20:00"


@dataclass
class SMCConfig:
    ob_atr_displacement: float = 1.5
    ob_lookback: int = 20
    ob_max_age: int = 50
    fvg_min_size_lr: float = 0.05
    fvg_max_age: int = 30
    equal_tolerance_lr: float = 0.05
    equal_min_touches: int = 2
    sweep_min_lr: float = 0.12
    swing_lookback: int = 5
    choch_lookback: int = 5
    mss_lookback: int = 5


@dataclass
class Lens1Config:
    obs_dim: int = 31
    encoding_dim: int = 8
    total_timesteps: int = 500000
    load_checkpoint: str = "final"
    resume_checkpoint: str = ""
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef_start: float = 0.05
    ent_coef_end: float = 0.01
    ent_coef_anneal_steps: int = 300000
    net_arch: List[int] = field(default_factory=lambda: [256, 128, 64])
    reward_direction_weight: float = 1.0
    reward_level_weight: float = 0.5
    reward_sweep_weight: float = 5.0
    reward_continuation_weight: float = 0.6
    reward_ranging_weight: float = 0.4
    reward_opening_bias_weight: float = 0.8
    reward_opening_style_weight: float = 0.7
    reward_opening_urgency_weight: float = 0.6
    opening_target_window_candles: int = 4
    model_save_path: str = "models/lens1"
    checkpoint_freq: int = 10000
    eval_freq: int = 5000
    eval_episodes: int = 5
    seed: int = 42


@dataclass
class Lens2Config:
    batch_size: int = 64
    learning_rate: float = 0.001
    n_epochs: int = 50
    net_arch: List[int] = field(default_factory=lambda: [128, 64, 32])
    model_save_path: str = "models/lens2"
    dataset_save_path: str = "data/processed/lens2_scenarios.csv"


@dataclass
class Lens3Config:
    obs_dim: int = 64
    total_timesteps: int = 1000000
    load_checkpoint: str = "final"
    resume_checkpoint: str = ""
    scenario_handover_enabled: bool = True
    oracle_mode: bool = False
    mask_scenarios: bool = False
    geometry_enabled: bool = True
    learning_rate: float = 0.0003
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef_start: float = 0.05
    ent_coef_end: float = 0.01
    ent_coef_anneal_steps: int = 200000
    net_arch: List[int] = field(default_factory=lambda: [256, 128, 64])
    min_rr: float = 1.5
    sl_buffer_lr: float = 0.10
    sl_min_atr_gate: float = 0.3
    sl_max_lr_gate: float = 1.5
    max_trades_per_day: int = 3
    london_range_atr_max: float = 4.0
    sweep_asymmetry_mode: bool = True
    sweep_low_entry_confidence: float = 1.0
    sweep_high_entry_confidence: float = 0.6
    sweep_low_min_rr: float = 1.5
    sweep_high_min_rr: float = 2.0
    sweep_high_requires_choch: bool = True
    ny_open_reward_window_steps: int = 4
    ny_open_entry_bonus: float = 0.05
    opening_intent_enabled: bool = True
    opening_intent_clusters_path: str = "output/lens1_analysis/lens1_scenario_clusters.csv"
    opening_intent_min_confidence: float = 0.20
    opening_intent_bonus: float = 0.20
    opening_intent_missed_penalty: float = -0.12
    opening_intent_opposite_penalty: float = -0.20
    opening_intent_first_trade_only: bool = True
    mask_mode: str = "scenario"
    displacement_min: float = 1.2
    momentum_r2_max: float = 0.65
    target_fps: Optional[float] = None
    opening_intent_urgency_bonus: float = 0.18
    opening_intent_delay_penalty: float = -0.05
    opening_intent_force_entry_enabled: bool = False
    opening_intent_force_hold_mask: bool = False
    opening_intent_force_entry_confidence: float = 0.60
    no_trade_force_entry_enabled: bool = False
    no_trade_force_hold_mask: bool = False
    no_trade_force_min_step: int = 4
    no_trade_force_max_step: int = 12
    no_trade_force_entry_confidence: float = 0.35
    no_trade_force_min_confluence: float = 3.0
    no_trade_feasible_hold_penalty: float = -0.08
    no_trade_feasible_hold_penalty_growth: float = 1.0
    structural_eval_freq: int = 10000
    structural_min_win_rate: float = 55.0
    structural_min_profit_factor: float = 1.8
    structural_min_trades_per_episode: float = 0.35
    structural_min_ny_open_trade_rate: float = 0.02
    structural_min_first_trade_intent_match_rate: float = 0.35
    structural_max_first_trade_entry_delay_from_anchor_min: float = 180.0
    structural_min_pnl_sweep_low_days: float = 0.0
    hold_rate_watch_threshold: float = 0.85
    hold_rate_watch_after_steps: int = 200000
    hold_rate_warn_threshold: float = 0.95
    hold_rate_warn_after_steps: int = 150000
    model_save_path: str = "models/lens3"
    load_checkpoint: str = "best"
    resume_checkpoint: str = "none"
    checkpoint_freq: int = 10000
    eval_freq: int = 5000
    eval_episodes: int = 5
    n_envs: int = 4
    seed: int = 42


@dataclass
class RiskConfig:
    account_balance: float = 100000
    risk_pct: float = 0.01
    max_daily_loss_pct: float = 0.03
    max_open_positions: int = 1


@dataclass
class RewardConfig:
    correct_hold: float = 0.0
    missed_setup_penalty: float = -0.15
    smc_alignment_bonus: float = 0.30
    confluence_bonus: float = 0.10
    confluence_max: float = 0.60
    risk_violation_penalty: float = -0.10
    overtrading_penalty: float = -0.30
    lens1_align_bonus: float = 0.20
    lens1_align_penalty: float = -0.20
    episode_pnl_bonus: float = 0.50
    episode_winrate_bonus: float = 0.30
    episode_no_trade_penalty: float = -0.80
    episode_overtrade_penalty: float = -0.20
    participation_bonus: float = 0.15
    progressive_no_trade_step_penalty: float = -0.015


@dataclass
class IndicatorConfig:
    atr_period: int = 14
    rsi_period: int = 14
    momentum_period: int = 8
    volume_sma_period: int = 20


@dataclass
class Settings:
    data: DataConfig = field(default_factory=DataConfig)
    sessions: SessionConfig = field(default_factory=SessionConfig)
    smc: SMCConfig = field(default_factory=SMCConfig)
    lens1: Lens1Config = field(default_factory=Lens1Config)
    lens2: Lens2Config = field(default_factory=Lens2Config)
    lens3: Lens3Config = field(default_factory=Lens3Config)
    risk: RiskConfig = field(default_factory=RiskConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    project_root: str = ""


def load_settings(config_path: Optional[str] = None) -> Settings:
    """Load settings from YAML file into typed dataclasses."""
    if config_path is None:
        # Auto-detect: walk up from this file to find config/settings.yaml
        current = Path(__file__).resolve().parent.parent
        config_path = str(current / "config" / "settings.yaml")

    if not os.path.exists(config_path):
        print(f"[WARN] Config not found at {config_path}, using defaults")
        settings = Settings()
        settings.project_root = str(Path(config_path).resolve().parent.parent)
        return settings

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    settings = Settings(
        data=DataConfig(**raw.get("data", {})),
        sessions=SessionConfig(**raw.get("sessions", {})),
        smc=SMCConfig(**raw.get("smc", {})),
        lens1=Lens1Config(**raw.get("lens1", {})),
        lens2=Lens2Config(**raw.get("lens2", {})),
        lens3=Lens3Config(**raw.get("lens3", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        reward=RewardConfig(**raw.get("reward", {})),
        indicators=IndicatorConfig(**raw.get("indicators", {})),
    )
    settings.project_root = str(Path(config_path).resolve().parent.parent)
    return settings


def get_project_root(settings: Settings) -> Path:
    """Return resolved project root path."""
    return Path(settings.project_root).resolve()
