"""
Evaluate selected Lens 2 checkpoints on the validation split and save
the richer opening-intent diagnostics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.pipeline import run_data_pipeline, load_news_dates
from src.lens1.environment import Lens1LondonEnv
from src.lens2.environment import Lens2NYEnv
from src.lens2.validation import evaluate_lens2_structural


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Lens 2 checkpoints on validation.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=[
            "models/lens2/lens2_340000_steps.zip",
            "models/lens2/lens2_500000_steps.zip",
        ],
        help="Checkpoint paths relative to project root unless absolute.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/lens2_validation/checkpoint_reviews",
        help="Directory to save validation outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    settings = load_settings(str(PROJECT_ROOT / "config" / "settings.yaml"))
    data = run_data_pipeline(settings, PROJECT_ROOT)
    news_dates = load_news_dates(str(PROJECT_ROOT / settings.data.news_calendar_path))
    lens1_model = PPO.load(str(PROJECT_ROOT / settings.lens1.model_save_path / "lens1_ppo_final.zip"))

    lens1_val_env = Lens1LondonEnv(data["validate"], data["validate_days"], settings)
    validate_env = Lens2NYEnv(
        data["validate"], data["validate_days"],
        lens1_model, lens1_val_env, settings, news_dates,
    )

    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint_arg in args.checkpoints:
        checkpoint_path = Path(checkpoint_arg)
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path

        if not checkpoint_path.exists():
            print(f"Skipping missing checkpoint: {checkpoint_path}")
            continue

        model = MaskablePPO.load(str(checkpoint_path))
        label = checkpoint_path.stem
        metrics = evaluate_lens2_structural(
            model,
            validate_env,
            settings=settings,
            output_dir=str(output_dir),
            label=label,
        )
        print(
            f"{label}: pf={metrics['profit_factor']:.2f}, "
            f"win_rate={metrics['win_rate']:.1f}%, "
            f"first_trade_wr={metrics['first_trade_win_rate']:.1f}%, "
            f"first_trade_ny_open={metrics['first_trade_is_ny_open_rate']:.1%}, "
            f"avg_first_delay={metrics['avg_first_trade_entry_delay_from_anchor_min']:.1f} min"
        )


if __name__ == "__main__":
    main()
