"""
Lens 3 MaskablePPO Agent — The Executor.
Uses sb3-contrib MaskablePPO for action-masked discrete trading.
"""

import os
import json
import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch.multiprocessing as mp

from src.lens3.environment import Lens3NYEnv
from src.lens3.validation import evaluate_lens3_structural

logger = logging.getLogger(__name__)


class UpdateEntCoefCallback(BaseCallback):
    """Linear entropy coefficient annealing for Lens 3."""

    def __init__(
        self,
        start_value: float = 0.05,
        end_value: float = 0.01,
        anneal_steps: int = 200000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.start_value = start_value
        self.end_value = end_value
        self.anneal_steps = anneal_steps

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.anneal_steps, 1.0)
        new_ent_coef = self.start_value + (self.end_value - self.start_value) * progress
        self.model.ent_coef = new_ent_coef

        if self.num_timesteps % 5000 == 0:
            self.logger.record("policy/ent_coef", new_ent_coef)

        return True


class Lens3MetricsCallback(BaseCallback):
    """
    Log Lens 2-specific trading metrics to TensorBoard.
    """

    def __init__(self, settings=None, verbose: int = 0):
        super().__init__(verbose)
        lens3_cfg = None if settings is None else settings.lens3
        self.episode_pnls = []
        self.episode_trades = []
        self.episode_wins = []
        self.episode_losses = []
        self.hold_count = 0
        self.total_count = 0
        self.ob_entries = 0
        self.fvg_entries = 0
        self.sweep_entries = 0
        self.lens1_aligned = 0
        self.total_entries = 0
        self.episode_records = []
        self.hold_rate_watch_threshold = 0.85 if lens3_cfg is None else lens3_cfg.hold_rate_watch_threshold
        self.hold_rate_watch_after_steps = 200000 if lens3_cfg is None else lens3_cfg.hold_rate_watch_after_steps
        self.hold_rate_warn_threshold = 0.95 if lens3_cfg is None else lens3_cfg.hold_rate_warn_threshold
        self.hold_rate_warn_after_steps = 150000 if lens3_cfg is None else lens3_cfg.hold_rate_warn_after_steps

    def _on_step(self) -> bool:
        self.total_count += 1

        # Track holds
        actions = self.locals.get("actions", [])
        if len(actions) > 0:
            if actions[0] == 0:
                self.hold_count += 1

        # Track episode completions from info
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            # Extract London Range
            lr = info.get("london_range", 0)
            if lr > 0:
                self.logger.record("session/london_range_pts", lr)
                
            if i < len(dones) and dones[i]:
                pnl = info.get("pnl", 0.0)
                trades = info.get("trades", 0)
                self.episode_pnls.append(pnl)
                self.episode_trades.append(trades)
                self.episode_records.append({
                    "pnl": float(pnl),
                    "trades": int(trades),
                    "sweep_day": int(info.get("sweep_day", 0)),
                    "sweep_high_day": int(info.get("sweep_high_day", 0)),
                    "sweep_low_day": int(info.get("sweep_low_day", 0)),
                    "ny_open_trade_taken": int(info.get("ny_open_trade_taken", 0)),
                    "first_trade_pnl": float(info.get("first_trade_pnl", np.nan)),
                    "first_trade_is_ny_open": int(info.get("first_trade_is_ny_open", 0)),
                    "first_trade_matches_intent": int(info.get("first_trade_matches_intent", 0)),
                    "first_trade_entry_delay_from_anchor_min": float(info.get("first_trade_entry_delay_from_anchor_min", np.nan)),
                    "first_trade_duration_min": float(info.get("first_trade_duration_min", np.nan)),
                })

        # Log every 5000 steps
        if self.num_timesteps % 5000 == 0:
            # Hold rate
            if self.total_count > 0:
                hold_rate = self.hold_count / self.total_count
                self.logger.record("policy/hold_rate_500", hold_rate)

                if self.num_timesteps >= self.hold_rate_warn_after_steps and hold_rate > self.hold_rate_warn_threshold:
                    logger.warning(
                        "HOLD RATE %.1f%% at step %s. Agent may be trapped in Do-Nothing mode.",
                        hold_rate * 100.0,
                        self.num_timesteps,
                    )
                elif self.num_timesteps >= self.hold_rate_watch_after_steps and hold_rate > self.hold_rate_watch_threshold:
                    logger.warning(
                        "HOLD RATE %.1f%% at step %s. Trading participation is still too low.",
                        hold_rate * 100.0,
                        self.num_timesteps,
                    )

            # Episode metrics
            if len(self.episode_pnls) > 0:
                recent = min(50, len(self.episode_pnls))
                recent_pnls = self.episode_pnls[-recent:]
                recent_trades = self.episode_trades[-recent:]

                self.logger.record("trade/episode_pnl", np.mean(recent_pnls))
                self.logger.record("trade/trades_per_episode", np.mean(recent_trades))

                wins = sum(1 for p in recent_pnls if p > 0)
                total_with_trades = sum(1 for t in recent_trades if t > 0)
                if total_with_trades > 0:
                    self.logger.record("trade/win_rate", wins / total_with_trades)

            if len(self.episode_records) > 0:
                recent_records = self.episode_records[-50:]
                self._log_split_metrics(recent_records)

            # Reset counters
            self.hold_count = 0
            self.total_count = 0

        return True

    def _log_split_metrics(self, records) -> None:
        sweep_high_days = [r["pnl"] for r in records if r["sweep_high_day"]]
        sweep_low_days = [r["pnl"] for r in records if r["sweep_low_day"]]
        sweep_days = [r["pnl"] for r in records if r["sweep_day"]]
        continuation_days = [r["pnl"] for r in records if not r["sweep_day"]]
        first_trade_pnls = [r["first_trade_pnl"] for r in records if not np.isnan(r["first_trade_pnl"])]
        first_trade_ny_open = [r["first_trade_is_ny_open"] for r in records if not np.isnan(r["first_trade_pnl"])]
        first_trade_matches_intent = [r["first_trade_matches_intent"] for r in records if not np.isnan(r["first_trade_pnl"])]
        first_trade_entry_delays = [r["first_trade_entry_delay_from_anchor_min"] for r in records if not np.isnan(r["first_trade_entry_delay_from_anchor_min"])]
        first_trade_durations = [r["first_trade_duration_min"] for r in records if not np.isnan(r["first_trade_duration_min"])]

        self.logger.record("trade/pnl_sweep_high_days", float(np.mean(sweep_high_days)) if sweep_high_days else 0.0)
        self.logger.record("trade/pnl_sweep_low_days", float(np.mean(sweep_low_days)) if sweep_low_days else 0.0)
        self.logger.record("trade/pnl_continuation_days", float(np.mean(continuation_days)) if continuation_days else 0.0)
        self.logger.record("trade/win_rate_sweep_days", float(np.mean([p > 0 for p in sweep_days])) if sweep_days else 0.0)
        self.logger.record("trade/win_rate_continuation", float(np.mean([p > 0 for p in continuation_days])) if continuation_days else 0.0)
        self.logger.record("trade/ny_open_trade_rate", float(np.mean([r["ny_open_trade_taken"] for r in records])) if records else 0.0)
        self.logger.record("trade/first_trade_pnl", float(np.mean(first_trade_pnls)) if first_trade_pnls else 0.0)
        self.logger.record("trade/first_trade_win_rate", float(np.mean([p > 0 for p in first_trade_pnls])) if first_trade_pnls else 0.0)
        self.logger.record("trade/first_trade_is_ny_open_rate", float(np.mean(first_trade_ny_open)) if first_trade_ny_open else 0.0)
        self.logger.record("trade/first_trade_matches_intent_rate", float(np.mean(first_trade_matches_intent)) if first_trade_matches_intent else 0.0)
        self.logger.record("trade/avg_first_trade_entry_delay_anchor_min", float(np.mean(first_trade_entry_delays)) if first_trade_entry_delays else 0.0)
        self.logger.record("trade/avg_first_trade_duration_min", float(np.mean(first_trade_durations)) if first_trade_durations else 0.0)


class StructuralEvalCallback(BaseCallback):
    """Evaluate saved structure on the validation split and keep the best passing checkpoint."""

    def __init__(
        self,
        eval_env: Lens3NYEnv,
        settings=None,
        save_path: str = ".",
        output_dir: str = ".",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        cfg = settings.lens3 if settings else None
        self.eval_env = eval_env
        self.settings = settings
        self.eval_freq = 10000 if cfg is None else cfg.structural_eval_freq
        self.save_path = save_path
        self.output_dir = output_dir
        self.best_profit_factor = float("-inf")
        self.history = []
        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.eval_env is None or self.num_timesteps == 0:
            return True
        if self.num_timesteps % self.eval_freq != 0:
            return True

        metrics = evaluate_lens3_structural(
            self.model,
            self.eval_env,
            settings=self.settings,
        )
        self._record_metrics(metrics)
        self._append_history(metrics)

        if metrics.get("acceptance_passed", False) and metrics["profit_factor"] > self.best_profit_factor:
            self.best_profit_factor = metrics["profit_factor"]
            best_path = os.path.join(self.save_path, "structural_best_model")
            self.model.save(best_path)
            payload = dict(metrics)
            payload["timesteps"] = self.num_timesteps
            with open(os.path.join(self.save_path, "structural_best_metrics.json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.info(
                "Saved new structural-best Lens 2 checkpoint at %s timesteps (pf=%.2f, win_rate=%.1f%%).",
                self.num_timesteps,
                metrics["profit_factor"],
                metrics["win_rate"],
            )

        return True

    def _on_training_end(self) -> None:
        if self.eval_env is None:
            return
        metrics = evaluate_lens3_structural(
            self.model,
            self.eval_env,
            settings=self.settings,
            output_dir=self.output_dir,
            label="final",
        )
        self._append_history(metrics, training_end=True)
        self._write_history()

    def _record_metrics(self, metrics: dict) -> None:
        self.logger.record("validation/win_rate", metrics["win_rate"])
        self.logger.record("validation/profit_factor", metrics["profit_factor"])
        self.logger.record("validation/trades_per_episode", metrics["trades_per_episode"])
        self.logger.record("validation/pnl_sweep_low_days", metrics["pnl_sweep_low_days"])
        self.logger.record("validation/pnl_continuation_days", metrics["pnl_continuation_days"])
        self.logger.record("validation/win_rate_sweep_days", metrics["win_rate_sweep_days"])
        self.logger.record("validation/win_rate_continuation", metrics["win_rate_continuation"])
        self.logger.record("validation/ny_open_trade_rate", metrics["ny_open_trade_rate"])
        self.logger.record("validation/first_trade_win_rate", metrics["first_trade_win_rate"])
        self.logger.record("validation/first_trade_is_ny_open_rate", metrics["first_trade_is_ny_open_rate"])
        self.logger.record("validation/first_trade_matches_intent_rate", metrics["first_trade_matches_intent_rate"])
        self.logger.record("validation/avg_first_trade_entry_delay_anchor_min", metrics["avg_first_trade_entry_delay_from_anchor_min"])
        self.logger.record("validation/avg_first_trade_duration_min", metrics["avg_first_trade_duration_min"])
        self.logger.record("validation/ny_open_first_trade_win_rate", metrics["ny_open_first_trade_win_rate"])
        self.logger.record("validation/acceptance_passed", float(metrics["acceptance_passed"]))

    def _append_history(self, metrics: dict, training_end: bool = False) -> None:
        row = dict(metrics)
        row["timesteps"] = self.num_timesteps
        row["training_end"] = int(training_end)
        row["acceptance_win_rate"] = metrics["acceptance"]["win_rate"]
        row["acceptance_profit_factor"] = metrics["acceptance"]["profit_factor"]
        row["acceptance_trades_per_episode"] = metrics["acceptance"]["trades_per_episode"]
        row["acceptance_pnl_sweep_low_days"] = metrics["acceptance"]["pnl_sweep_low_days"]
        self.history.append(row)
        self._write_history()

    def _write_history(self) -> None:
        if not self.history:
            return
        pd.DataFrame(self.history).to_csv(
            os.path.join(self.output_dir, "structural_eval_history.csv"),
            index=False,
        )


def mask_fn(env: Lens3NYEnv) -> np.ndarray:
    """Mask function for ActionMasker wrapper."""
    return env.action_masks()


def create_lens3_agent(
    env_factory,
    eval_env: Optional[Lens3NYEnv] = None,
    settings=None,
    project_root: str = ".",
    resume_from: Optional[str] = None,
) -> MaskablePPO:
    """Create Lens 3 MaskablePPO agent with multi-env support."""
    cfg = settings.lens3 if settings else None
    
    n_envs = 1 if cfg is None else cfg.n_envs
    learning_rate = 0.0003 if cfg is None else cfg.learning_rate
    n_steps = 2048 if cfg is None else cfg.n_steps
    batch_size = 64 if cfg is None else cfg.batch_size
    n_epochs = 10 if cfg is None else cfg.n_epochs
    gamma = 0.99 if cfg is None else cfg.gamma
    gae_lambda = 0.95 if cfg is None else cfg.gae_lambda
    clip_range = 0.2 if cfg is None else cfg.clip_range
    ent_coef_start = 0.05 if cfg is None else cfg.ent_coef_start
    net_arch = [256, 128, 64] if cfg is None else cfg.net_arch
    seed = 42 if cfg is None else cfg.seed

    tb_log = os.path.join(project_root, "output", "tensorboard", "lens2")
    os.makedirs(tb_log, exist_ok=True)

    def make_env(rank: int):
        def _init():
            env = env_factory()
            # Set unique seed for each environment
            env.seed(seed + rank)
            return ActionMasker(env, mask_fn)
        return _init

    if n_envs > 1:
        vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    else:
        vec_env = DummyVecEnv([make_env(0)])

    if resume_from and os.path.exists(resume_from):
        model = MaskablePPO.load(resume_from, env=vec_env)
        logger.info("Resumed Lens 3 MaskablePPO agent from %s", resume_from)
    else:
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            ent_coef=ent_coef_start,
            policy_kwargs=dict(
                net_arch=dict(pi=net_arch, vf=net_arch),
            ),
            tensorboard_log=tb_log,
            verbose=1,
            seed=seed,
        )

    logger.info(f"Created Lens 3 MaskablePPO agent (n_envs={n_envs}):")
    logger.info(f"  Policy: MlpPolicy, arch={net_arch}")
    logger.info(f"  LR={learning_rate}, n_steps={n_steps}, batch={batch_size}")

    return model


def train_lens3(
    model: MaskablePPO,
    total_timesteps: int,
    eval_env: Optional[Lens3NYEnv] = None,
    settings=None,
    project_root: str = ".",
    reset_num_timesteps: bool = True,
):
    """Train Lens 2 with entropy annealing and checkpointing."""
    cfg = settings.lens3 if settings else None
    ent_start = 0.05 if cfg is None else cfg.ent_coef_start
    ent_end = 0.01 if cfg is None else cfg.ent_coef_end
    ent_steps = 200000 if cfg is None else cfg.ent_coef_anneal_steps
    checkpoint_freq = 10000 if cfg is None else cfg.checkpoint_freq
    eval_freq = 5000 if cfg is None else cfg.eval_freq
    eval_episodes = 5 if cfg is None else cfg.eval_episodes
    save_path = os.path.join(project_root, "models", "lens2") if cfg is None else os.path.join(project_root, cfg.model_save_path)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "best"), exist_ok=True)

    callbacks = [
        UpdateEntCoefCallback(ent_start, ent_end, ent_steps),
        CheckpointCallback(save_freq=checkpoint_freq, save_path=save_path, name_prefix="lens2"),
        Lens3MetricsCallback(settings),
    ]

    if eval_env is not None:
        structural_cb = StructuralEvalCallback(
            eval_env,
            settings=settings,
            save_path=os.path.join(save_path, "structural_best"),
            output_dir=os.path.join(project_root, "output", "lens3_validation"),
        )
        callbacks.append(structural_cb)

        wrapped_eval = ActionMasker(eval_env, mask_fn)
        vec_eval = DummyVecEnv([lambda: wrapped_eval])
        eval_cb = MaskableEvalCallback(
            vec_eval,
            best_model_save_path=os.path.join(save_path, "best"),
            log_path=os.path.join(save_path, "logs"),
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            verbose=1,
        )
        callbacks.append(eval_cb)

    logger.info(f"Training Lens 3: {total_timesteps} timesteps")
    # Disable the live progress bar so detached/background runs stay stable.
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
        reset_num_timesteps=reset_num_timesteps,
    )

    final_path = os.path.join(save_path, "lens3_ppo_final")
    model.save(final_path)
    logger.info(f"Lens 3 training complete. Final model saved to {final_path}.zip")

    return model
