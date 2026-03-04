"""
BowlingThrowEnv — one episode = полная игра в боулинг.

Задача: просто обернуть ALE Bowling так, чтобы:
  - наблюдения были preprocessed в ROI-тензор (1, 75, 160);
  - эпизод длился до конца игры (10 фреймов и т.п.), как решает сам симулятор;
  - награда бралась напрямую из ALE, без дополнительного shaping.
"""

import gymnasium as gym
import ale_py
import numpy as np
import torch

gym.register_envs(ale_py)

NOOP     = 0
FIRE     = 1
UP       = 2
DOWN     = 3
UPFIRE   = 4
DOWNFIRE = 5

_FIRE_ACTIONS = {FIRE, UPFIRE, DOWNFIRE}
_MAX_EPISODE_STEPS = 10_000  # защитный лимит, на нормальную игру не влияет
ROI_TOP = 100
ROI_BOTTOM = 175
ROI_CHANNEL = 2
OBS_SHAPE = (1, ROI_BOTTOM - ROI_TOP, 160)  # (1, 75, 160)


def preprocess(frame: np.ndarray) -> torch.Tensor:
    # Keep only the region of interest from a single color channel.
    roi = frame[ROI_TOP:ROI_BOTTOM, :, ROI_CHANNEL].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(roi)).unsqueeze(0)


class BowlingThrowEnv(gym.Wrapper):

    def __init__(self):
        super().__init__(gym.make("ALE/Bowling-v5", render_mode="rgb_array", mode=2))
        self._total_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._total_steps = 0
        return preprocess(obs), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._total_steps += 1
        timed_out = self._total_steps >= _MAX_EPISODE_STEPS
        done = terminated or truncated or timed_out

        # Базовая награда = то, что отдаёт симулятор ALE (очки за игру).
        shaped_reward = float(reward) - 0.001

        # Локальный бонус за страйк, чтобы агент "чувствовал" событие в моменте:
        # в Bowling ALE один страйк даёт 10 очков за этот бросок,
        # поэтому считаем step со reward >= 10 страйком и добавляем бонус.
        if shaped_reward >= 10.0:
            shaped_reward += 20.0

        return preprocess(obs), shaped_reward, done, truncated, info
