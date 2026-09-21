"""
Lens 1 PPO Agent — continuous action space for encoding.
Uses standard PPO (not MaskablePPO) since all actions are always valid.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, BaseCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv

from src.lens1.environment import Lens1LondonEnv

logger = logging.getLogger(__name__)


class UpdateEntCoefCallback(BaseCallback):
    """
    Custom callback to linearly anneal entropy coefficient.
    SB3 doesn't support dynamic ent_coef schedules natively.
    """

    def __init__(
        self,
        start_value: float = 0.05,
        end_value: float = 0.01,
        anneal_steps: int = 150000,
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


class Lens1EncodingMetricsCallback(BaseCallback):
    """
    Log Lens 1-specific metrics to TensorBoard.
    """

    def __init__(self, eval_env=None, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.episode_encodings = []

    def _on_step(self) -> bool:
        # Collect encoding from info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "encoding" in info:
                self.episode_encodings.append(info["encoding"])

        # Log every 5000 steps
        if self.num_timesteps % 5000 == 0 and len(self.episode_encodings) > 0:
            encodings = np.array(self.episode_encodings[-100:])
            if encodings.ndim == 2 and encodings.shape[0] > 1:
                # Encoding variance: are all dims being used?
                variance = np.var(encodings, axis=0).mean()
                self.logger.record("lens1/encoding_variance", variance)

                # Per-dimension variance
                for i in range(encodings.shape[1]):
                    self.logger.record(f"lens1/dim_{i}_var", np.var(encodings[:, i]))

        return True


def create_lens1_agent(
    train_env: Lens1LondonEnv,
    eval_env: Optional[Lens1LondonEnv] = None,
    settings=None,
    project_root: str = ".",
    resume_from: Optional[str] = None,
) -> PPO:
    """
    Create the Lens 1 PPO agent with continuous action space.
    """
    cfg = settings.lens1 if settings else None

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

    tb_log = os.path.join(project_root, "output", "tensorboard", "lens1")
    os.makedirs(tb_log, exist_ok=True)

    # Vectorise environment
    vec_train = DummyVecEnv([lambda: train_env])

    if resume_from:
        model = PPO.load(resume_from, env=vec_train)
        logger.info("Resumed Lens 1 PPO agent from %s", resume_from)
    else:
        model = PPO(
            "MlpPolicy",
            vec_train,
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
                activation_fn=torch.nn.Tanh,
            ),
            tensorboard_log=tb_log,
            verbose=1,
            seed=seed,
        )

    logger.info(f"Created Lens 1 PPO agent:")
    logger.info(f"  Policy: MlpPolicy, arch={net_arch}, activation=Tanh")
    logger.info(f"  LR={learning_rate}, n_steps={n_steps}, batch={batch_size}")
    logger.info(f"  ent_coef_start={ent_coef_start}")

    return model


def train_lens1(
    model: PPO,
    total_timesteps: int,
    eval_env: Optional[Lens1LondonEnv] = None,
    settings=None,
    project_root: str = ".",
    reset_num_timesteps: bool = True,
):
    """
    Train Lens 1 with entropy annealing and checkpointing.
    """
    cfg = settings.lens1 if settings else None
    ent_start = 0.05 if cfg is None else cfg.ent_coef_start
    ent_end = 0.01 if cfg is None else cfg.ent_coef_end
    ent_steps = 150000 if cfg is None else cfg.ent_coef_anneal_steps
    checkpoint_freq = 10000 if cfg is None else cfg.checkpoint_freq
    eval_freq = 5000 if cfg is None else cfg.eval_freq
    eval_episodes = 5 if cfg is None else cfg.eval_episodes
    save_path = os.path.join(project_root, "models", "lens1") if cfg is None else os.path.join(project_root, cfg.model_save_path)

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, "best"), exist_ok=True)

    callbacks = [
        UpdateEntCoefCallback(ent_start, ent_end, ent_steps),
        CheckpointCallback(save_freq=checkpoint_freq, save_path=save_path, name_prefix="lens1"),
        Lens1EncodingMetricsCallback(),
    ]

    if eval_env is not None:
        vec_eval = DummyVecEnv([lambda: eval_env])
        eval_cb = EvalCallback(
            vec_eval,
            best_model_save_path=os.path.join(save_path, "best"),
            log_path=os.path.join(save_path, "logs"),
            eval_freq=eval_freq,
            n_eval_episodes=eval_episodes,
            deterministic=True,
            verbose=1,
        )
        callbacks.append(eval_cb)

    logger.info(f"Training Lens 1: {total_timesteps} timesteps")
    # Disable the live progress bar so long background runs remain stable when
    # launched without an attached interactive console.
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
        reset_num_timesteps=reset_num_timesteps,
    )

    # Save final model
    final_path = os.path.join(save_path, "lens1_ppo_final")
    model.save(final_path)
    logger.info(f"Lens 1 training complete. Final model saved to {final_path}.zip")

    return model
