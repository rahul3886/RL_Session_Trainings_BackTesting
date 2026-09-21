"""
Pipeline — orchestrates Lens 1 → Lens 2 training handoff.
Ensures sequential training: Lens 1 first, then Lens 2 with anchored Lens 1.
"""

import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import Settings, load_settings, get_project_root
from src.data.ingest import ingest_data
from src.data.validator import validate_data, split_data, count_trading_days
from src.data.indicators import add_indicators
from src.model_registry import maybe_resolve_model_checkpoint
from src.session.profiler import get_trading_days

logger = logging.getLogger(__name__)


def load_news_dates(calendar_path: str) -> set:
    """Load high-impact news dates from CSV."""
    news_dates = set()

    if not os.path.exists(calendar_path):
        logger.warning(f"News calendar not found: {calendar_path}")
        return news_dates

    try:
        df = pd.read_csv(calendar_path)
        if "date" in df.columns:
            for d in df["date"]:
                news_dates.add(str(d).strip())
        logger.info(f"Loaded {len(news_dates)} news dates from calendar")
    except Exception as e:
        logger.warning(f"Failed to load news calendar: {e}")

    return news_dates


def run_data_pipeline(settings: Settings, project_root: Path) -> dict:
    """
    Run the full data pipeline: download → validate → add indicators → split.

    Returns:
        dict with 'df_15m', 'df_5m', 'train', 'validate', 'test', 'trading_days'
    """
    logger.info("=" * 60)
    logger.info("DATA PIPELINE")
    logger.info("=" * 60)

    raw_dir = str(project_root / settings.data.raw_dir)
    processed_dir = str(project_root / settings.data.processed_dir)

    result = {}

    for tf in settings.data.timeframes:
        logger.info(f"\n--- Loading {tf} data ---")
        df = ingest_data(
            instrument=settings.data.instrument,
            start=settings.data.train_start,
            end=settings.data.test_end,
            timeframe=tf,
            raw_dir=raw_dir,
            processed_dir=processed_dir,
        )

        # Validate
        df, stats = validate_data(df, label=f"XAUUSD_{tf}")

        # Add indicators
        df = add_indicators(df, settings)

        result[f"df_{tf.lower()}"] = df
        logger.info(f"  {tf}: {len(df)} bars, {count_trading_days(df)} trading days")

    # Use 15M as primary timeframe
    df_primary = result.get("df_m15", result.get("df_m5"))

    if df_primary is None:
        raise RuntimeError("No data loaded for any timeframe")

    # Split
    train, validate, test = split_data(
        df_primary,
        settings.data.train_start,
        settings.data.train_end,
        settings.data.validate_start,
        settings.data.validate_end,
        settings.data.test_start,
        settings.data.test_end,
    )

    # Get trading days for each split
    session_cfg = settings.sessions
    train_days = get_trading_days(
        train, session_cfg.london_start, session_cfg.london_end,
        session_cfg.ny_start, session_cfg.ny_end,
    )
    validate_days = get_trading_days(
        validate, session_cfg.london_start, session_cfg.london_end,
        session_cfg.ny_start, session_cfg.ny_end,
    )
    test_days = get_trading_days(
        test, session_cfg.london_start, session_cfg.london_end,
        session_cfg.ny_start, session_cfg.ny_end,
    )

    result.update({
        "df_primary": df_primary,
        "train": train,
        "validate": validate,
        "test": test,
        "train_days": train_days,
        "validate_days": validate_days,
        "test_days": test_days,
    })

    logger.info(f"\nPipeline summary:")
    logger.info(f"  Train days:    {len(train_days)}")
    logger.info(f"  Validate days: {len(validate_days)}")
    logger.info(f"  Test days:     {len(test_days)}")
    logger.info(f"  Total:         {len(train_days) + len(validate_days) + len(test_days)}")

    return result


def run_lens1_training(settings: Settings, data: dict, project_root: Path):
    """Run Lens 1 training pipeline."""
    from src.lens1.environment import Lens1LondonEnv
    from src.lens1.agent import create_lens1_agent, train_lens1

    logger.info("=" * 60)
    logger.info("STEP 2: TRAINING LENS 1 — LONDON OBSERVER")
    logger.info("=" * 60)

    # Create environments
    train_env = Lens1LondonEnv(
        data["train"], data["train_days"], settings
    )
    validate_env = Lens1LondonEnv(
        data["validate"], data["validate_days"], settings
    )

    resume_path = maybe_resolve_model_checkpoint(
        project_root,
        settings.lens1.model_save_path,
        "lens1",
        settings.lens1.resume_checkpoint,
    )
    # Create agent
    model = create_lens1_agent(
        train_env,
        validate_env,
        settings,
        str(project_root),
        resume_from=str(resume_path) if resume_path else None,
    )

    # Train
    model = train_lens1(
        model,
        total_timesteps=settings.lens1.total_timesteps,
        eval_env=validate_env,
        settings=settings,
        project_root=str(project_root),
        reset_num_timesteps=resume_path is None,
    )

    return model, train_env, validate_env


def run_lens1_analysis(model, env, trading_days: list, project_root: Path):
    """Run Lens 1 post-training analysis."""
    from src.lens1.encoder import run_full_analysis

    output_dir = str(project_root / "output" / "lens1_analysis")
    return run_full_analysis(model, env, trading_days, output_dir)


def run_lens2_training(settings: Settings, data: dict, project_root: Path):
    """Run Lens 2 supervised classifier training."""
    from src.lens2.dataset import generate_golden_dataset
    from src.lens2.train import train_lens2_classifier
    
    logger.info("=" * 60)
    logger.info("STEP 2.5: GENERATING GOLDEN DATASET & TRAINING LENS 2 PREDICTOR")
    logger.info("=" * 60)
    
    # Generate the dataset first
    generate_golden_dataset(data["df_primary"], settings)
    
    # Train the PyTorch classifier
    train_lens2_classifier(settings, project_root)
    
    # Load the trained scaler and model for inference later
    import torch
    import joblib
    from src.lens2.classifier import Lens2Predictor
    
    model_dir = Path(project_root) / settings.lens2.model_save_path
    scaler = joblib.load(model_dir / "lens2_scaler.pkl")
    
    # Identify feature dim by looking at the dataset columns minus date/scenario_id
    df_dataset = pd.read_csv(Path(project_root) / settings.lens2.dataset_save_path)
    feature_dim = len([c for c in df_dataset.columns if c not in ["date", "scenario_id"]])
    
    model = Lens2Predictor(
        input_dim=feature_dim,
        hidden_dim=settings.lens2.net_arch,
        num_classes=9
    )
    model.load_state_dict(torch.load(model_dir / "lens2_best.pth", weights_only=True))
    model.eval()
    
    return model, scaler


def run_lens3_training(
    settings: Settings,
    data: dict,
    lens1_model,
    lens2_predictor,
    lens2_scaler,
    project_root: Path,
    news_dates: set = None,
    holistic: bool = False,
):
    """Run Lens 3 training pipeline with anchored Lens 1 and predictive Lens 2."""
    from src.lens1.environment import Lens1LondonEnv
    from src.lens3.environment import Lens3NYEnv
    from src.lens3.agent import create_lens3_agent, train_lens3

    logger.info("=" * 60)
    logger.info("STEP 3: TRAINING LENS 3 — NY EXECUTOR")
    logger.info("=" * 60)

    # Merge all data if holistic mode is enabled
    if holistic:
        import pandas as pd
        logger.info("HOLISTIC MODE ENABLED (Merging all data splits for strategy verification)")
        df_all = pd.concat([data["train"], data["validate"], data["test"]]).sort_index()
        days_all = sorted(list(set(data["train_days"] + data["validate_days"] + data["test_days"])))
        train_df = df_all
        train_days = days_all
    else:
        train_df = data["train"]
        train_days = data["train_days"]

    # Create Lens 1 env for encoding extraction
    lens1_env = Lens1LondonEnv(
        train_df, train_days, settings
    )

    # Create environment factory for vectorization
    def train_env_factory():
        return Lens3NYEnv(
            train_df, train_days,
            lens1_model, lens1_env, settings, news_dates,
            lens2_predictor=lens2_predictor,
            lens2_scaler=lens2_scaler,
            df_m1=data.get("df_m1"),
            df_m5=data.get("df_m5")
        )

    lens1_val_env = Lens1LondonEnv(
        data["validate"], data["validate_days"], settings
    )
    validate_env = Lens3NYEnv(
        data["validate"], data["validate_days"],
        lens1_model, lens1_val_env, settings, news_dates,
        lens2_predictor=lens2_predictor,
        lens2_scaler=lens2_scaler,
        df_m1=data.get("df_m1"),
        df_m5=data.get("df_m5")
    )

    from src.model_registry import maybe_resolve_model_checkpoint
    resume_path = maybe_resolve_model_checkpoint(
        project_root,
        settings.lens3.model_save_path,
        "lens3",
        settings.lens3.resume_checkpoint,
    )

    # Create agent
    model = create_lens3_agent(
        train_env_factory,
        validate_env,
        settings,
        str(project_root),
        resume_from=str(resume_path) if resume_path else None,
    )

    # Train
    model = train_lens3(
        model,
        total_timesteps=settings.lens3.total_timesteps,
        eval_env=validate_env,
        settings=settings,
        project_root=str(project_root),
        reset_num_timesteps=resume_path is None,
    )

    return model, validate_env


def run_backtest_pipeline(
    settings: Settings,
    data: dict,
    lens1_model,
    lens2_predictor,
    lens2_scaler,
    lens3_model,
    project_root: Path,
    news_dates: set = None,
):
    """Run walk-forward backtest on test split."""
    from src.backtester import run_backtest

    logger.info("=" * 60)
    logger.info("WALK-FORWARD BACKTEST — TEST SPLIT")
    logger.info("=" * 60)

    output_dir = str(project_root / "output" / "backtest")

    metrics = run_backtest(
        data["test"],
        data["test_days"],
        lens1_model,
        lens2_predictor,
        lens2_scaler,
        lens3_model,
        df_m1=data.get("df_m1"),
        df_m5=data.get("df_m5"),
        settings=settings,
        news_dates=news_dates,
        output_dir=output_dir,
    )

    return metrics
