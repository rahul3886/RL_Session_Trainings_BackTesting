"""
Expert Baseline Validation Script.
Evaluates the 1M-step model using perfectly labeled Oracle Scenario IDs (S1-S8).
This confirms the model's true potential before moving to predictive inference.
"""
import os
import sys
import logging
from pathlib import Path

# Ensure src is in path
sys.path.append(os.getcwd())

from src.config import load_settings
from sb3_contrib import MaskablePPO
from src.pipeline import run_data_pipeline, load_news_dates
from src.lens2.validation import evaluate_lens2_structural
from src.lens2.environment import Lens2NYEnv
from src.lens1.environment import Lens1LondonEnv
from sb3_contrib.common.wrappers import ActionMasker

def mask_fn(env):
    return env.action_masks()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    project_root = Path("c:/RAHUL BOGI/RL_Session Trainings/xauusd_v3")
    settings = load_settings(project_root / "config" / "settings.yaml")
    
    # Force settings for Expert Baseline
    settings.lens2.oracle_mode = True
    settings.lens2.obs_dim = 56
    settings.lens2.n_envs = 1

    # Load data
    logger.info("Loading and processing data...")
    data = run_data_pipeline(settings, project_root)
    news_dates = load_news_dates(str(project_root / settings.data.news_calendar_path))
    
    # Load model
    model_path = project_root / "models" / "lens2" / "lens2_ppo_final.zip"
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}")
        return

    # Use training split for baseline (to see how well it mastered the teacher)
    # Plus validation split for general expert potential
    splits = ["train", "validate"]
    
    for split in splits:
        logger.info(f"--- Running Expert Baseline on {split} split ---")
        
        days = data[f"{split}_days"]
        l1_env = Lens1LondonEnv(data[split], days, settings)
        l2_env = Lens2NYEnv(
            data[split], days,
            lens1_model=None, 
            lens1_env=l1_env,
            settings=settings,
            news_dates=news_dates
        )
        
        # Wrap for masking
        wrapped_env = ActionMasker(l2_env, mask_fn)
        
        # Load model
        model = MaskablePPO.load(model_path, env=wrapped_env)
        
        output_dir = project_root / "output" / "expert_baseline" / split
        metrics = evaluate_lens2_structural(
            model, l2_env, settings=settings, 
            output_dir=str(output_dir), 
            label=f"expert_{split}"
        )
        
        logger.info(f"Expert {split} Results: WR={metrics['win_rate']:.1f}% PF={metrics['profit_factor']:.2f}")

if __name__ == "__main__":
    main()
