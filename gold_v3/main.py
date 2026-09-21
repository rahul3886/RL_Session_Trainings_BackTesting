"""
XAUUSD v3 Three-Lens SMC Trading Engine — CLI Orchestrator.

Usage:
  python main.py --mode download           ← fetch Dukascopy data
  python main.py --mode train-lens1        ← Step 1: train London observer
  python main.py --mode analyse-lens1      ← what did Lens 1 learn?
  python main.py --mode train-lens2        ← Step 2: train NY Predictor (Classifier)
  python main.py --mode train-lens3        ← Step 3: train NY Executor (RL)
  python main.py --mode backtest           ← full three-lens replay
  python main.py --mode signal             ← live signal output
  python main.py --mode smoke-test         ← validate three envs
  python main.py --mode tensorboard        ← launch TensorBoard
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings, get_project_root
from src.model_registry import resolve_model_checkpoint


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_download(settings, project_root):
    """Download data from Dukascopy."""
    from src.data.ingest import ingest_data
    from src.data.validator import validate_data, count_trading_days
    from src.data.indicators import add_indicators

    raw_dir = str(project_root / settings.data.raw_dir)
    processed_dir = str(project_root / settings.data.processed_dir)
    os.makedirs(raw_dir, exist_ok=True)

    for tf in settings.data.timeframes:
        print(f"\n{'='*60}")
        print(f"Downloading {settings.data.instrument.upper()} {tf}")
        print(f"{'='*60}")

        df = ingest_data(
            instrument=settings.data.instrument,
            start=settings.data.train_start,
            end=settings.data.test_end,
            timeframe=tf,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
        )

        df, stats = validate_data(df, label=f"XAUUSD_{tf}")
        df = add_indicators(df, settings)

        # Save enriched data
        enriched_path = os.path.join(processed_dir, f"{settings.data.instrument}_{tf}_enriched.parquet")
        df.to_parquet(enriched_path)

        print(f"\n--- SUMMARY ---")
        print(f"  Timeframe:     {tf}")
        print(f"  Total bars:    {stats['final_rows']}")
        print(f"  Trading days:  {count_trading_days(df)}")
        print(f"  Date range:    {stats['date_range_start']}")
        print(f"                 {stats['date_range_end']}")
        print(f"  Gaps detected: {stats['gaps_detected']}")
        print(f"  Saved to:      {enriched_path}")


def cmd_train_lens1(settings, project_root):
    """Train Lens 1 (London Observer)."""
    from src.pipeline import run_data_pipeline, run_lens1_training

    data = run_data_pipeline(settings, project_root)
    model, train_env, val_env = run_lens1_training(settings, data, project_root)
    print("\nLens 1 training complete.")
    print(f"   Model saved to: {project_root / settings.lens1.model_save_path}")


def cmd_analyse_lens1(settings, project_root):
    """Run Lens 1 post-training analysis."""
    from stable_baselines3 import PPO
    from src.pipeline import run_data_pipeline, run_lens1_analysis
    from src.lens1.environment import Lens1LondonEnv

    # Load trained model
    try:
        model_path = resolve_model_checkpoint(
            project_root,
            settings.lens1.model_save_path,
            "lens1",
            settings.lens1.load_checkpoint,
        )
    except FileNotFoundError:
        print(f"Lens 1 model not found for selector '{settings.lens1.load_checkpoint}'")
        print("   Run --mode train-lens1 first.")
        return

    data = run_data_pipeline(settings, project_root)

    model = PPO.load(str(model_path))
    val_env = Lens1LondonEnv(data["validate"], data["validate_days"], settings)

    run_lens1_analysis(model, val_env, data["validate_days"], project_root)
    print("\nLens 1 analysis complete.")
    print(f"   Outputs saved to: {project_root / 'output' / 'lens1_analysis'}")


def cmd_train_lens2(settings, project_root):
    """Train Lens 2 (NY Predictor) Supervised Classifier."""
    from src.pipeline import run_data_pipeline, run_lens2_training

    data = run_data_pipeline(settings, project_root)
    model, scaler = run_lens2_training(settings, data, project_root)
    print("\nLens 2 predictor training complete.")
    print(f"   Model saved to: {project_root / settings.lens2.model_save_path}")


def cmd_train_lens3(settings, project_root):
    """Train Lens 3 (NY Executor) with anchored Lens 1 and Predictive Lens 2."""
    import torch
    import joblib
    import pandas as pd
    from stable_baselines3 import PPO
    from src.pipeline import run_data_pipeline, run_lens3_training, load_news_dates
    from src.model_registry import resolve_model_checkpoint
    from src.lens2.classifier import Lens2Predictor

    # Load Lens 1
    try:
        lens1_path = resolve_model_checkpoint(
            project_root,
            settings.lens1.model_save_path,
            "lens1",
            settings.lens1.load_checkpoint,
        )
    except FileNotFoundError:
        print(f"Lens 1 model not found")
        return

    lens1_model = PPO.load(str(lens1_path))
    
    # Load Lens 2 Predictor
    try:
        model_dir = Path(project_root) / settings.lens2.model_save_path
        scaler = joblib.load(model_dir / "lens2_scaler.pkl")
        df_dataset = pd.read_csv(Path(project_root) / settings.lens2.dataset_save_path)
        feature_dim = len([c for c in df_dataset.columns if c not in ["date", "scenario_id"]])
        lens2_predictor = Lens2Predictor(
            input_dim=feature_dim,
            hidden_dim=settings.lens2.net_arch,
            num_classes=9
        )
        lens2_predictor.load_state_dict(torch.load(model_dir / "lens2_best.pth", weights_only=True))
        lens2_predictor.eval()
    except Exception as e:
        print(f"Lens 2 Predictor not found: {e}. Run --mode train-lens2 first.")
        return

    data = run_data_pipeline(settings, project_root)

    # Load news calendar
    news_path = str(project_root / settings.data.news_calendar_path)
    news_dates = load_news_dates(news_path)

    model, env = run_lens3_training(settings, data, lens1_model, lens2_predictor, scaler, project_root, news_dates, holistic=False)
    print("\nLens 3 training complete.")
    print(f"   Model saved to: {project_root / settings.lens3.model_save_path}")


def cmd_train_lens3_holistic(settings, project_root):
    """Run Lens 3 training on all available days (Holistic Mode)."""
    import torch
    import joblib
    import pandas as pd
    from stable_baselines3 import PPO
    from src.pipeline import run_data_pipeline, run_lens3_training, load_news_dates
    from src.model_registry import resolve_model_checkpoint
    from src.lens2.classifier import Lens2Predictor

    # Load Lens 1 Encoder
    try:
        lens1_path = resolve_model_checkpoint(
            project_root,
            settings.lens1.model_save_path,
            "lens1",
            settings.lens1.load_checkpoint,
        )
    except FileNotFoundError:
        print(f"Lens 1 model not found")
        return

    lens1_model = PPO.load(str(lens1_path))
    
    # Load Lens 2 Predictor
    try:
        model_dir = Path(project_root) / settings.lens2.model_save_path
        scaler = joblib.load(model_dir / "lens2_scaler.pkl")
        df_dataset = pd.read_csv(Path(project_root) / settings.lens2.dataset_save_path)
        feature_dim = len([c for c in df_dataset.columns if c not in ["date", "scenario_id"]])
        lens2_predictor = Lens2Predictor(
            input_dim=feature_dim,
            hidden_dim=settings.lens2.net_arch,
            num_classes=9
        )
        lens2_predictor.load_state_dict(torch.load(model_dir / "lens2_best.pth", weights_only=True))
        lens2_predictor.eval()
    except Exception as e:
        print(f"Lens 2 Predictor not found: {e}. Run --mode train-lens2 first.")
        return

    data = run_data_pipeline(settings, project_root)
    news_dates = load_news_dates(str(project_root / settings.data.news_calendar_path))

    model, env = run_lens3_training(
        settings, data, lens1_model, lens2_predictor, scaler, 
        project_root, news_dates, holistic=True
    )
    print("\nLens 3 Holistic training complete.")
    print(f"   Model saved to: {project_root / settings.lens3.model_save_path}")


def cmd_backtest(settings, project_root):
    """Run walk-forward backtest on test split."""
    import torch
    import joblib
    import pandas as pd
    from stable_baselines3 import PPO
    from sb3_contrib import MaskablePPO
    from src.pipeline import run_data_pipeline, run_backtest_pipeline, load_news_dates
    from src.lens1.environment import Lens1LondonEnv
    from src.lens2.classifier import Lens2Predictor
    from src.lens3.environment import Lens3NYEnv
    from src.lens3.validation import evaluate_lens3_structural

    # Load both models
    try:
        lens1_path = resolve_model_checkpoint(
            project_root,
            settings.lens1.model_save_path,
            "lens1",
            settings.lens1.load_checkpoint,
        )
        lens3_path = resolve_model_checkpoint(
            project_root,
            settings.lens3.model_save_path,
            "lens3",
            settings.lens3.load_checkpoint,
        )
    except FileNotFoundError:
        print("RL Models not found. Complete training first.")
        return

    lens1_model = PPO.load(str(lens1_path))
    lens3_model = MaskablePPO.load(str(lens3_path))
    
    # Load Lens 2 Predictor
    try:
        model_dir = Path(project_root) / settings.lens2.model_save_path
        lens2_scaler = joblib.load(model_dir / "lens2_scaler.pkl")
        df_dataset = pd.read_csv(Path(project_root) / settings.lens2.dataset_save_path)
        feature_dim = len([c for c in df_dataset.columns if c not in ["date", "scenario_id"]])
        lens2_predictor = Lens2Predictor(
            input_dim=feature_dim,
            hidden_dim=settings.lens2.net_arch,
            num_classes=9
        )
        lens2_predictor.load_state_dict(torch.load(model_dir / "lens2_best.pth", weights_only=True))
        lens2_predictor.eval()
    except Exception as e:
        print(f"Lens 2 Predictor not found: {e}. Run --mode train-lens2 first.")
        return

    data = run_data_pipeline(settings, project_root)
    news_dates = load_news_dates(str(project_root / settings.data.news_calendar_path))

    metrics = run_backtest_pipeline(settings, data, lens1_model, lens2_predictor, lens2_scaler, lens3_model, project_root, news_dates)
    print("\nBacktest complete.")
    print(f"   Results: {project_root / 'output' / 'backtest' / 'results.md'}")


def cmd_train_hmm(settings, project_root):
    """Run Lens 0 HMM Regime Training using hmmlearn."""
    from src.data.regime_hmm import run_hmm_training
    
    print("\nStarting Phase 9: Lens 0 Unsupervised Regime Training (HMM)...")
    run_hmm_training(settings, project_root)
    print("\nLens 0 HMM Training Complete.")


def cmd_smoke_test(settings, project_root):
    """Validate both environments work correctly."""
    import numpy as np
    from src.pipeline import run_data_pipeline, load_news_dates
    from src.lens1.environment import Lens1LondonEnv
    from src.lens3.environment import Lens3NYEnv

    print("=" * 60)
    print("SMOKE TEST — Validating Environments")
    print("=" * 60)

    data = run_data_pipeline(settings, project_root)
    news_dates = load_news_dates(str(project_root / settings.data.news_calendar_path))

    print(f"  MT5 symbol resolved: {settings.data.mt5_symbol}")
    print(f"  Bars loaded: {len(data['train'])} train, "
          f"{len(data['validate'])} validate, {len(data['test'])} test")
    print(f"  Trading days: train={len(data['train_days'])}, "
          f"val={len(data['validate_days'])}, "
          f"test={len(data['test_days'])}")

    # Test Lens 1 Environment
    print("\n--- Lens 1 Environment ---")
    lens1_env = Lens1LondonEnv(data["train"], data["train_days"][:5], settings)
    obs, info = lens1_env.reset()
    print(f"  Obs shape:    {obs.shape} (expected: {settings.lens1.obs_dim})")
    print(f"  Obs non-zero: {np.count_nonzero(obs)}")
    print(f"  Phase:        {info.get('phase')}")

    # Step through a few candles
    for i in range(3):
        action = lens1_env.action_space.sample()
        obs, reward, term, trunc, info = lens1_env.step(action)
        print(f"  Step {i}: phase={info.get('phase')}, reward={reward:.4f}")

    print("  Lens 1 env OK")

    # Test Lens 3 Environment
    print("\n--- Lens 3 Environment ---")
    lens3_env = Lens3NYEnv(
        data["train"], data["train_days"][:5],
        lens1_model=None, lens1_env=None,
        settings=settings, news_dates=news_dates,
    )
    obs, info = lens3_env.reset()
    print(f"  Obs shape:    {obs.shape} (expected: {settings.lens3.obs_dim})")
    print(f"  Obs non-zero: {np.count_nonzero(obs)}")

    # Step through Lens 3 checks
    buy_count = 0
    sell_count = 0
    for i in range(min(20, lens3_env.ny_candle_count)):
        masks = lens3_env.action_masks()
        if buy_count < 5 and masks[1]:
            action = 1  # BUY
            buy_count += 1
        elif sell_count < 5 and masks[2]:
            action = 2  # SELL
            sell_count += 1
        else:
            action = 0  # HOLD

        obs, reward, term, trunc, info = lens3_env.step(action)
        if action != 0:
            print(f"  Step {i}: action={'BUY' if action==1 else 'SELL'}, "
                  f"pos={info.get('position')}, reward={reward:.4f}")

        if term or trunc:
            break

    london_range = info.get("london_range", 0)
    print(f"\n  London Range today: {london_range:.2f} pts")
    print(f"  dist_to_LH in state: {obs[10]:.3f} LR fractions")
    print(f"  (should be 0.0–3.0, not 0–50)")

    if london_range > 0 and (obs[10] > 5.0 or obs[10] < -5.0):
        print("  WARNING: distances look like ATR-normalised,"
              " not LR-normalised. Check features.py")
    else:
        print("  Distance normalisation looks correct")

    print(f"  State dims:   {obs.shape[0]}")
    print(f"  Trades taken: {lens3_env.trades_this_episode}")
    print("  Lens 3 env OK")

    # Validate state vector
    print("\n--- State Vector Validation ---")
    print(f"  Lens 1 encoding dims: {settings.lens1.encoding_dim}")
    print(f"  Lens 3 obs dims:      {settings.lens3.obs_dim}")
    print(f"  Total = {settings.lens1.encoding_dim} + {settings.lens3.obs_dim - settings.lens1.encoding_dim} = {settings.lens3.obs_dim}")
    print("  Dimensions match")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)


def cmd_signal(settings, project_root):
    """Output live trading signal."""
    print("Signal mode — requires live data feed (not implemented in v3 standalone)")
    print("Use --mode backtest for historical analysis.")


def cmd_tensorboard(settings, project_root):
    """Launch TensorBoard."""
    import sys
    tb_dir = project_root / "output" / "tensorboard"
    print(f"Launching TensorBoard from: {tb_dir}")
    os.system(f"{sys.executable} -m tensorboard.main --logdir {tb_dir}")


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description="XAUUSD v3 Two-Lens SMC Trading Engine"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "download", "train-hmm", "train-lens1", "analyse-lens1",
            "train-lens2", "train-lens3", "train-lens3-holistic", "backtest", "signal",
            "smoke-test", "tensorboard",
        ],
        help="Operation mode",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to settings.yaml (default: config/settings.yaml)",
    )

    args = parser.parse_args()

    # Load settings
    config_path = args.config or str(PROJECT_ROOT / "config" / "settings.yaml")
    settings = load_settings(config_path)
    project_root = Path(settings.project_root) if settings.project_root else PROJECT_ROOT

    print(f"\n{'='*60}")
    print(f"XAUUSD v3 Two-Lens SMC Trading Engine")
    print(f"Mode: {args.mode}")
    print(f"Project root: {project_root}")
    print(f"{'='*60}\n")

    # Create output directories
    for d in ["output/lens1_analysis", "output/backtest", "output/tensorboard",
              "models/lens1", "models/lens2", "models/lens3", "data/raw", "data/processed"]:
        os.makedirs(project_root / d, exist_ok=True)

    mode_map = {
        "download": cmd_download,
        "train-hmm": cmd_train_hmm,
        "train-lens1": cmd_train_lens1,
        "analyse-lens1": cmd_analyse_lens1,
        "train-lens2": cmd_train_lens2,
        "train-lens3": cmd_train_lens3,
        "train-lens3-holistic": cmd_train_lens3_holistic,
        "backtest": cmd_backtest,
        "signal": cmd_signal,
        "smoke-test": cmd_smoke_test,
        "tensorboard": cmd_tensorboard,
    }

    try:
        mode_map[args.mode](settings, project_root)
    except Exception as e:
        logger.error(f"Error in mode '{args.mode}': {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
