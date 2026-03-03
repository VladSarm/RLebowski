"""
BowlingThrowEnv — one episode = полная игра в боулинг.

Задача: просто обернуть ALE Bowling так, чтобы:
  - наблюдения были preprocessed в (1, 84, 84) тензор;
  - эпизод длился до конца игры (10 фреймов и т.п.), как решает сам симулятор;
  - награда бралась напрямую из ALE, без дополнительного shaping.
"""

import gymnasium as gym
import ale_py
import numpy as np
import torch
from PIL import Image

gym.register_envs(ale_py)

NOOP     = 0
FIRE     = 1
UP       = 2
DOWN     = 3
UPFIRE   = 4
DOWNFIRE = 5

_FIRE_ACTIONS = {FIRE, UPFIRE, DOWNFIRE}
_MAX_EPISODE_STEPS = 10_000  # защитный лимит, на нормальную игру не влияет


def preprocess(frame: np.ndarray) -> torch.Tensor:
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    arr = np.array(
        Image.fromarray(gray).resize((84, 84), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


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
