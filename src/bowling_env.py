"""
BowlingThrowEnv  — one episode = one ball throw.
preprocess()     — RGB frame → grayscale torch.Tensor (1, 84, 84).
"""

import gymnasium as gym
import ale_py
import numpy as np
import torch
from PIL import Image

gym.register_envs(ale_py)

# Actions (Discrete 6)
NOOP     = 0
FIRE     = 1   # throw
UP       = 2
DOWN     = 3
UPFIRE   = 4   # throw while moving up
DOWNFIRE = 5   # throw while moving down

_FIRE_ACTIONS = {FIRE, UPFIRE, DOWNFIRE}

# Ball rolls ~210 steps after FIRE; 260 is a safety timeout post-FIRE
_THROW_TIMEOUT = 260
# Total episode budget: ~40 steps to position + 260 to roll
_MAX_EPISODE_STEPS = 300


def preprocess(frame: np.ndarray) -> torch.Tensor:
    """
    RGB (210, 160, 3) uint8  →  grayscale tensor (1, 84, 84) float32 [0, 1].
    Luminance-weighted conversion to grayscale, resized to 84×84.
    """
    gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    arr = np.array(
        Image.fromarray(gray).resize((84, 84), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # (1, 84, 84)


class BowlingThrowEnv(gym.Wrapper):
    """
    ALE/Bowling-v5 wrapper: one episode = one ball throw.

    Throw lifecycle:
      1. Positioning — agent moves the bowler (UP/DOWN) and picks the throw moment.
      2. FIRE / UPFIRE / DOWNFIRE — ball launched. Step counter starts.
         UP/DOWN after this phase position the bowler for the NEXT throw.
      3. Episode ends: reward > 0 (ball reached pins)
                        OR timeout _THROW_TIMEOUT steps after FIRE.

    Between throws reset() does NOT restart ALE — returns the current frame.
    After the full game ends (terminated=True) reset() restarts ALE.

    Observation : torch.Tensor (1, 84, 84)  float32  [0, 1]
    Action space: Discrete(6)  — NOOP FIRE UP DOWN UPFIRE DOWNFIRE
    Reward      : number of pins knocked down (0 for gutter ball or timeout)
    """

    def __init__(self):
        super().__init__(gym.make("ALE/Bowling-v5", render_mode="rgb_array", mode=2))
        self._thrown = False
        self._steps_since_fire = 0
        self._total_steps = 0
        self._game_over = False

    def reset(self, **kwargs):
        if self._game_over:
            obs, info = self.env.reset(**kwargs)
            self._game_over = False
        else:
            # First throw or continuation — must reset ALE first
            obs, info = self.env.reset(**kwargs)

        self._thrown = False
        self._steps_since_fire = 0
        self._total_steps = 0
        return preprocess(obs), info

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self._total_steps += 1

        just_thrown = action in _FIRE_ACTIONS and not self._thrown
        if just_thrown:
            self._thrown = True
            self._steps_since_fire = 0

        if self._thrown:
            self._steps_since_fire += 1

        throw_done = (
            self._total_steps >= _MAX_EPISODE_STEPS
            or (self._thrown and (reward > 0 or self._steps_since_fire >= _THROW_TIMEOUT))
        )

        if terminated or truncated:
            self._game_over = True
            throw_done = True

        # -0.01 per step + +1 for throwing + pin/gutter reward at episode end
        shaped_reward = -0.01
        if just_thrown:
            shaped_reward += 1.0
        if throw_done:
            if reward == 0:
                shaped_reward += -1.0
            else:
                shaped_reward += float(reward) + (20.0 if reward >= 10 else 0.0)
        return preprocess(obs), shaped_reward, throw_done, truncated, info
