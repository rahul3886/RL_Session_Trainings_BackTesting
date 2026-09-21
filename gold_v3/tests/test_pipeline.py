"""
Tests for the pipeline — import checks and basic integration.
"""

from pathlib import Path

import pytest


class TestImports:
    """Verify all modules import without error."""

    def test_import_config(self):
        from src.config import load_settings, Settings

    def test_import_data_ingest(self):
        from src.data.ingest import ingest_data, fetch_mt5, load_csv

    def test_import_data_validator(self):
        from src.data.validator import validate_data, split_data

    def test_import_data_indicators(self):
        from src.data.indicators import add_indicators

    def test_import_smc_order_blocks(self):
        from src.smc.order_blocks import detect_order_blocks, get_nearest_ob

    def test_import_smc_fvg(self):
        from src.smc.fair_value_gaps import detect_fvgs, get_nearest_fvg

    def test_import_smc_liquidity(self):
        from src.smc.liquidity import detect_equal_levels, detect_sweep, map_liquidity_levels

    def test_import_smc_structure(self):
        from src.smc.structure import detect_swing_points, detect_choch, detect_mss

    def test_import_session_profiler(self):
        from src.session.profiler import slice_session, get_trading_days, compute_session_profile

    def test_import_session_features(self):
        from src.session.features import extract_session_features, build_lens1_state_vector, build_lens2_state_vector

    def test_import_lens1(self):
        from src.lens1.environment import Lens1LondonEnv
        from src.lens1.reward import Lens1RewardCalculator
        from src.lens1.agent import create_lens1_agent
        from src.lens1.encoder import run_full_analysis

    def test_import_lens2(self):
        from src.lens2.environment import Lens2NYEnv
        from src.lens2.reward import Lens2RewardCalculator
        from src.lens2.agent import create_lens2_agent
        from src.lens2.callbacks import TradingMetricsCallback

    def test_import_risk(self):
        from src.risk.manager import check_trade_risk, compute_structural_sl
        from src.risk.sizing import compute_position_size

    def test_import_backtester(self):
        from src.backtester import run_backtest

    def test_import_pipeline(self):
        from src.pipeline import run_data_pipeline, load_news_dates

    def test_import_model_registry(self):
        from src.model_registry import resolve_model_checkpoint, maybe_resolve_model_checkpoint


class TestConfigValues:
    def test_walk_forward_dates(self):
        from src.config import load_settings
        s = load_settings()
        assert s.data.train_start == "2022-01-01"
        assert s.data.train_end == "2024-01-01"
        assert s.data.validate_start == "2024-01-01"
        assert s.data.validate_end == "2024-07-01"
        assert s.data.test_start == "2024-07-01"
        assert s.data.test_end == "2025-01-01"

    def test_lens_dimensions(self):
        from src.config import load_settings
        s = load_settings()
        assert s.lens1.obs_dim == 31
        assert s.lens1.encoding_dim == 8
        assert s.lens2.obs_dim == 40
        assert s.lens2.obs_dim == s.lens1.encoding_dim + 32  # 8 + 32

    def test_reward_values(self):
        from src.config import load_settings
        s = load_settings()
        assert s.reward.correct_hold == 0.0
        assert s.reward.missed_setup_penalty < 0
        assert s.reward.smc_alignment_bonus > 0

    def test_resolve_model_checkpoint_selectors(self, tmp_path):
        from src.model_registry import resolve_model_checkpoint, maybe_resolve_model_checkpoint

        model_root = tmp_path / "models" / "lens2"
        (model_root / "best").mkdir(parents=True)
        (model_root / "structural_best").mkdir(parents=True)
        (model_root / "lens2_ppo_final.zip").write_text("final", encoding="utf-8")
        (model_root / "best" / "best_model.zip").write_text("best", encoding="utf-8")
        (model_root / "structural_best" / "structural_best_model.zip").write_text("structural", encoding="utf-8")
        (model_root / "lens2_890000_steps.zip").write_text("step", encoding="utf-8")

        assert resolve_model_checkpoint(tmp_path, "models/lens2", "lens2", "final").name == "lens2_ppo_final.zip"
        assert resolve_model_checkpoint(tmp_path, "models/lens2", "lens2", "best_eval").name == "best_model.zip"
        assert resolve_model_checkpoint(tmp_path, "models/lens2", "lens2", "structural_best").name == "structural_best_model.zip"
        assert resolve_model_checkpoint(tmp_path, "models/lens2", "lens2", "step:890000").name == "lens2_890000_steps.zip"
        assert maybe_resolve_model_checkpoint(tmp_path, "models/lens2", "lens2", "") is None


class TestPipelineBehaviour:
    def test_run_lens2_training_returns_validation_env(self, monkeypatch, tmp_path):
        from src.config import load_settings
        from src import pipeline
        import src.lens1.environment as lens1_environment
        import src.lens2.environment as lens2_environment
        import src.lens2.agent as lens2_agent
        import src.model_registry as model_registry

        created_lens2_envs = []

        class StubLens1Env:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

        class StubLens2Env:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs
                created_lens2_envs.append(self)

        monkeypatch.setattr(lens1_environment, "Lens1LondonEnv", StubLens1Env)
        monkeypatch.setattr(lens2_environment, "Lens2NYEnv", StubLens2Env)
        monkeypatch.setattr(model_registry, "maybe_resolve_model_checkpoint", lambda *args, **kwargs: None)
        monkeypatch.setattr(lens2_agent, "create_lens2_agent", lambda *args, **kwargs: "created-model")
        monkeypatch.setattr(lens2_agent, "train_lens2", lambda *args, **kwargs: "trained-model")

        settings = load_settings()
        data = {
            "train": "train-frame",
            "train_days": ["2024-01-15"],
            "validate": "validate-frame",
            "validate_days": ["2024-01-16"],
        }

        model, env = pipeline.run_lens2_training(
            settings,
            data,
            lens1_model=object(),
            project_root=tmp_path,
            news_dates=set(),
        )

        assert model == "trained-model"
        assert env is created_lens2_envs[-1]
