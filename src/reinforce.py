import torch
import numpy as np
import bowling_env
import mlp
import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.vector import AsyncVectorEnv
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = mlp.PolicyNetwork().to(DEVICE)
env = bowling_env.BowlingThrowEnv()  # kept for eval in main.py
optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
gamma = 0.998


# ---------------------------------------------------------------------------
# AsyncVectorEnv requires:
#   1. A module-level class (picklable for subprocess)
#   2. A module-level factory function (not a lambda)
# The wrapper converts tensor obs → numpy so VectorEnv can batch them.
# ---------------------------------------------------------------------------

class _NumpyEnv(gym.Env):
    """BowlingThrowEnv with numpy observations for AsyncVectorEnv."""

    def __init__(self):
        self._env = bowling_env.BowlingThrowEnv()
        self.observation_space = spaces.Box(0.0, 1.0, (1, 84, 84), dtype=np.float32)
        self.action_space = self._env.action_space

    def reset(self, **kwargs):
        obs, info = self._env.reset(**kwargs)
        return obs.numpy(), info

    def step(self, action):
        obs, reward, done, trunc, info = self._env.step(int(action))
        return obs.numpy(), reward, done, trunc, info

    def close(self):
        self._env.close()


def _make_numpy_env():
    return _NumpyEnv()


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------

def _rollout_async(vec_env, n_envs):
    """
    Collect one complete episode per env.

    All envs step in parallel each tick. When env i finishes (done),
    it gets NOOP actions until the rest catch up.
    GPU forward pass is batched over all n_envs observations at once.

    Returns: log_probs_per_env, rewards_per_env, steps_per_env
    """
    obs_np, _ = vec_env.reset()                      # (n_envs, 1, 84, 84) float32

    active = np.ones(n_envs, dtype=bool)
    log_probs_per = [[] for _ in range(n_envs)]
    rewards_per   = [[] for _ in range(n_envs)]
    steps_per     = [0]  * n_envs

    while active.any():
        # One batched GPU forward pass for all n_envs
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
        log_probs_all = policy.get_action_log_probabilities(obs_t)   # (n_envs, 6)
        probs = log_probs_all.exp()
        actions = torch.multinomial(probs, 1).squeeze(1)              # (n_envs,)

        actions_np = actions.detach().cpu().numpy().astype(np.int32)
        actions_np[~active] = 0   # NOOP for already-finished envs

        obs_np, rewards, terminated, truncated, _ = vec_env.step(actions_np)

        for i in range(n_envs):
            if active[i]:
                log_probs_per[i].append(log_probs_all[i, actions[i]])
                rewards_per[i].append(float(rewards[i]))
                steps_per[i] += 1
                if terminated[i] or truncated[i]:
                    active[i] = False

    return log_probs_per, rewards_per, steps_per


def _rollout_single(single_env):
    """Collect one episode from a single env (no VectorEnv overhead)."""
    state, _ = single_env.reset()
    state = state.to(DEVICE)
    log_probs, rewards = [], []
    step_count = 0

    while True:
        log_probs_all = policy.get_action_log_probabilities(state.unsqueeze(0))
        probs = log_probs_all.exp()
        action = torch.multinomial(probs, 1).item()
        next_state, reward, done, _, _ = single_env.step(action)
        log_probs.append(log_probs_all[0, action])
        rewards.append(reward)
        state = next_state.to(DEVICE)
        step_count += 1
        if done:
            break

    return [log_probs], [rewards], [step_count]


def _compute_returns_and_update(log_probs_per, rewards_per):
    """Compute discounted returns, normalize, do one gradient update."""
    batch_log_probs = []
    batch_returns   = []
    ep_totals       = []

    for log_probs, rewards in zip(log_probs_per, rewards_per):
        G, returns = 0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        batch_log_probs.extend(log_probs)
        batch_returns.extend(returns)
        ep_totals.append(sum(rewards))

    returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=DEVICE)
    returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    log_probs_t = torch.stack(batch_log_probs)
    loss = -torch.sum(log_probs_t * returns_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), ep_totals


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def episode_termination(episodes, n_envs=1, save_period=None, checkpoint_dir=None):
    writer = SummaryWriter(log_dir="runs/reinforce")

    if n_envs > 1:
        vec_env = AsyncVectorEnv([_make_numpy_env] * n_envs)
    else:
        vec_env = None

    if checkpoint_dir:
        path = checkpoint_dir / "policy_ep0.pt"
        torch.save(policy.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

    episode = 0
    while episode < episodes:
        if n_envs > 1:
            lp, rw, st = _rollout_async(vec_env, n_envs)
        else:
            lp, rw, st = _rollout_single(env)

        loss_val, ep_totals = _compute_returns_and_update(lp, rw)
        episode += len(ep_totals)

        mean_reward = sum(ep_totals) / len(ep_totals)
        mean_steps  = int(sum(st) / len(st))

        writer.add_scalar("train/mean_return", mean_reward, episode)
        writer.add_scalar("train/loss",        loss_val,    episode)
        writer.add_scalar("train/mean_steps",  mean_steps,  episode)

        print(f"Episode {episode:4d} | Envs: {n_envs} | Steps: {mean_steps:4d} | "
              f"Return: {mean_reward:6.1f} | Loss: {loss_val:8.4f}")

        if save_period and checkpoint_dir and episode % save_period == 0:
            path = checkpoint_dir / f"policy_ep{episode}.pt"
            torch.save(policy.state_dict(), path)
            print(f"  Checkpoint saved: {path}")

    if vec_env is not None:
        vec_env.close()
    writer.close()
