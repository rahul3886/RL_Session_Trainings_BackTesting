"""
Resume Lens 2 training from the latest saved checkpoint.
"""

import argparse
from pathlib import Path
import re
import sys

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings
from src.pipeline import run_data_pipeline, load_news_dates
from src.lens1.environment import Lens1LondonEnv
from src.lens2.environment import Lens2NYEnv
from src.lens2.agent import train_lens2, mask_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Resume Lens 2 training from a checkpoint.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path relative to project root unless absolute. Defaults to latest step checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = PROJECT_ROOT
    settings = load_settings(str(project_root / "config" / "settings.yaml"))

    data = run_data_pipeline(settings, project_root)
    news_dates = load_news_dates(str(project_root / settings.data.news_calendar_path))
    lens1_model = PPO.load(str(project_root / settings.lens1.model_save_path / "lens1_ppo_final.zip"))

    train_lens1_env = Lens1LondonEnv(data["train"], data["train_days"], settings)
    train_env = Lens2NYEnv(
        data["train"], data["train_days"],
        lens1_model, train_lens1_env, settings, news_dates,
    )

    val_lens1_env = Lens1LondonEnv(data["validate"], data["validate_days"], settings)
    validate_env = Lens2NYEnv(
        data["validate"], data["validate_days"],
        lens1_model, val_lens1_env, settings, news_dates,
    )

    wrapped_train = ActionMasker(train_env, mask_fn)
    vec_train = DummyVecEnv([lambda: wrapped_train])

    model_dir = project_root / settings.lens2.model_save_path
    candidates = []
    for path in model_dir.glob("lens2_*_steps.zip"):
        match = re.match(r"lens2_(\d+)_steps\.zip", path.name)
        if match:
            candidates.append((int(match.group(1)), path))

    if not candidates:
        raise RuntimeError("No Lens 2 step checkpoints found to resume from.")

    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = project_root / checkpoint_path
        if not checkpoint_path.exists():
            raise RuntimeError(f"Checkpoint not found: {checkpoint_path}")
        match = re.match(r"lens2_(\d+)_steps\.zip", checkpoint_path.name)
        resume_step = int(match.group(1)) if match else 0
    else:
        resume_step, checkpoint_path = max(candidates, key=lambda item: item[0])
    model = MaskablePPO.load(str(checkpoint_path), env=vec_train)

    remaining = max(int(settings.lens2.total_timesteps) - int(model.num_timesteps), 0)
    print(f"Resuming Lens 2 from {checkpoint_path.name}")
    print(f"Loaded num_timesteps={model.num_timesteps}")
    print(f"Remaining timesteps={remaining}")

    if remaining <= 0:
        print("No remaining timesteps. Nothing to resume.")
        return

    train_lens2(
        model,
        total_timesteps=remaining,
        eval_env=validate_env,
        settings=settings,
        project_root=str(project_root),
        reset_num_timesteps=False,
    )


if __name__ == "__main__":
    main()
