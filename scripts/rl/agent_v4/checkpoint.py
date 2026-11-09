"""Strict new-run/checkpoint identity. Legacy weights are never migrated."""
from dataclasses import asdict
from pathlib import Path
import os

import torch

from agent_v4 import SCHEMA, MODEL_VERSION, STATE_VERSION
from agent_v4.model import Agent, ModelConfig, PPOConfig, Trainer
from agent_v4.reward import RewardConfig


def save(path, model, trainer, experiment, iteration):
    state = {"schema": SCHEMA, "model_version": MODEL_VERSION,
             "state_version": STATE_VERSION, "model_config": asdict(model.config),
             "ppo_config": asdict(trainer.config), "experiment": experiment,
             "model_state": model.state_dict(), "iteration": iteration,
             "actor_optimizer": trainer.actor_optimizer.state_dict(),
             "critic_optimizer": trainer.critic_optimizer.state_dict(),
             "torch_rng_state": torch.get_rng_state()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        torch.save(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load(path: Path, experiment=None, train=False):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if (ckpt.get("schema") != SCHEMA or ckpt.get("model_version") != MODEL_VERSION or
            ckpt.get("state_version") != STATE_VERSION):
        raise ValueError("Not an Agent v4 checkpoint; v3 requires its original runner")
    if experiment is not None and ckpt["experiment"] != experiment:
        raise ValueError("Resume experiment changed (source/config/codec/lock)")
    config = PPOConfig(**ckpt["ppo_config"])
    if (ckpt["ppo_config"] != ckpt["experiment"].get("ppo_config") or
            ckpt["model_config"] != ckpt["experiment"].get("model_config")):
        raise ValueError("Checkpoint and experiment model/PPO config differ")
    reward = RewardConfig(**ckpt["experiment"]["reward"])
    if reward.mode == "clip-delta" and (config.gamma != 1 or config.gae_lambda != 1):
        raise ValueError("clip-delta default requires undiscounted Monte Carlo advantages")
    model = Agent(ModelConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"], strict=True)
    trainer = None
    if train:
        trainer = Trainer(model, config)
        trainer.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        trainer.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        torch.set_rng_state(ckpt["torch_rng_state"])
    return ckpt, model, trainer

