import torch
import numpy as np
import random
from datetime import datetime
from pathlib import Path
import bowling_env
import mlp
import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.vector import AsyncVectorEnv
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

policy = mlp.PolicyNetwork().to(DEVICE)
env = bowling_env.BowlingThrowEnv()  # kept for eval in main.py
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
gamma = 0.999
ppo_clip_epsilon = 0.2
ppo_update_epochs = 10
ppo_num_mini_batches = 2


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
        self.observation_space = spaces.Box(0.0, 1.0, bowling_env.OBS_SHAPE, dtype=np.float32)
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


def set_seed(seed):
    """Set RNG seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------

def _rollout_async(vec_env, n_envs, reset_seeds=None):
    """
    Collect one complete episode per env.

    All envs step in parallel each tick. When env i finishes (done),
    it gets NOOP actions until the rest catch up.
    GPU forward pass is batched over all n_envs observations at once.

    Returns:
      obs_per_env, actions_per_env, old_log_probs_per_env, rewards_per_env, steps_per_env
    """
    if reset_seeds is None:
        obs_np, _ = vec_env.reset()                  # (n_envs, 1, 75, 160) float32
    else:
        obs_np, _ = vec_env.reset(seed=reset_seeds)  # (n_envs, 1, 75, 160) float32

    active = np.ones(n_envs, dtype=bool)
    obs_per = [[] for _ in range(n_envs)]
    actions_per = [[] for _ in range(n_envs)]
    old_log_probs_per = [[] for _ in range(n_envs)]
    rewards_per   = [[] for _ in range(n_envs)]
    steps_per     = [0]  * n_envs

    while active.any():
        # One batched GPU forward pass for all n_envs
        obs_before_step = obs_np.copy()
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            log_probs_all = policy.get_action_log_probabilities(obs_t)   # (n_envs, 6)
            probs = log_probs_all.exp()
            actions = torch.multinomial(probs, 1).squeeze(1)              # (n_envs,)

        actions_np = actions.detach().cpu().numpy().astype(np.int32)
        actions_np[~active] = 0   # NOOP for already-finished envs

        obs_np, rewards, terminated, truncated, _ = vec_env.step(actions_np)

        for i in range(n_envs):
            if active[i]:
                action_i = int(actions[i].item())
                obs_per[i].append(obs_before_step[i])
                actions_per[i].append(action_i)
                old_log_probs_per[i].append(float(log_probs_all[i, action_i].item()))
                rewards_per[i].append(float(rewards[i]))
                steps_per[i] += 1
                if terminated[i] or truncated[i]:
                    active[i] = False

    return obs_per, actions_per, old_log_probs_per, rewards_per, steps_per


def _compute_normalized_returns(rewards_per):
    """Compute discounted returns and normalize over the rollout batch."""
    batch_returns = []
    ep_totals       = []

    for rewards in rewards_per:
        G, returns = 0, []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        batch_returns.extend(returns)
        ep_totals.append(sum(rewards))

    returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=DEVICE)
    returns_t = (returns_t - returns_t.mean()) / (returns_t.std(unbiased=False) + 1e-8)
    return returns_t, ep_totals


def _flatten_obs_actions(obs_per, actions_per):
    """Flatten observations/actions and keep sample order stable."""
    batch_obs = []
    batch_actions = []

    for obs_list, actions_list in zip(obs_per, actions_per):
        if len(obs_list) != len(actions_list):
            raise ValueError("Mismatch between observations and actions in rollout.")
        batch_obs.extend(obs_list)
        batch_actions.extend(actions_list)

    obs_t = torch.tensor(np.asarray(batch_obs), dtype=torch.float32, device=DEVICE)
    actions_t = torch.tensor(batch_actions, dtype=torch.long, device=DEVICE)
    return obs_t, actions_t


def _flatten_rollout(obs_per, actions_per, old_log_probs_per):
    """Flatten per-env rollout data to tensors aligned by sample index."""
    obs_t, actions_t = _flatten_obs_actions(obs_per, actions_per)
    batch_old_log_probs = []

    for log_probs_list in old_log_probs_per:
        batch_old_log_probs.extend(log_probs_list)

    old_log_probs_t = torch.tensor(batch_old_log_probs, dtype=torch.float32, device=DEVICE)
    if old_log_probs_t.numel() != actions_t.numel():
        raise ValueError("Mismatch between old log_probs and actions in rollout.")
    return obs_t, actions_t, old_log_probs_t


def _ppo_update(obs_per, actions_per, old_log_probs_per, advantages_t):
    """Run multiple PPO-Clip policy updates on shuffled mini-batches."""
    obs_t, actions_t, old_log_probs_t = _flatten_rollout(obs_per, actions_per, old_log_probs_per)

    if advantages_t.numel() != actions_t.numel():
        raise ValueError("Mismatch between rollout samples and computed advantages.")

    batch_size = actions_t.size(0)
    if batch_size == 0:
        raise ValueError("Empty rollout batch.")

    policy_losses = []
    approx_kls = []
    clip_fractions = []

    for _ in range(ppo_update_epochs):
        shuffled_idx = torch.randperm(batch_size, device=DEVICE)
        mini_batch_indices = torch.tensor_split(shuffled_idx, ppo_num_mini_batches)

        for mb_idx in mini_batch_indices:
            if mb_idx.numel() == 0:
                continue

            mb_obs = obs_t[mb_idx]
            mb_actions = actions_t[mb_idx]
            mb_old_log_probs = old_log_probs_t[mb_idx]
            mb_advantages = advantages_t[mb_idx]

            log_probs_all = policy.get_action_log_probabilities(mb_obs)  # (mini_batch, n_actions)
            new_log_probs_t = log_probs_all.gather(1, mb_actions.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(new_log_probs_t - mb_old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1.0 - ppo_clip_epsilon, 1.0 + ppo_clip_epsilon)

            surr1 = ratio * mb_advantages
            surr2 = clipped_ratio * mb_advantages
            policy_loss = -torch.mean(torch.min(surr1, surr2))

            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

            with torch.no_grad():
                approx_kl = (mb_old_log_probs - new_log_probs_t).mean()
                clip_fraction = ((ratio - 1.0).abs() > ppo_clip_epsilon).float().mean()

            policy_losses.append(float(policy_loss.item()))
            approx_kls.append(float(approx_kl.item()))
            clip_fractions.append(float(clip_fraction.item()))

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "approx_kl": float(np.mean(approx_kls)),
        "clip_fraction": float(np.mean(clip_fractions)),
    }


def _reinforce_update(obs_per, actions_per, normalized_returns_t):
    """Run one REINFORCE policy update on the rollout batch."""
    obs_t, actions_t = _flatten_obs_actions(obs_per, actions_per)

    if normalized_returns_t.numel() != actions_t.numel():
        raise ValueError("Mismatch between rollout samples and normalized returns.")

    log_probs_all = policy.get_action_log_probabilities(obs_t)
    new_log_probs_t = log_probs_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)
    loss = -torch.sum(new_log_probs_t * normalized_returns_t)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return {
        "loss": float(loss.item()),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def episode_termination(
    episodes,
    n_envs=1,
    save_period=None,
    checkpoint_dir=None,
    algo="ppo",
    seed=42,
    tb_log_dir=None,
):
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1 for batch-only training.")
    if algo not in {"ppo", "reinforce"}:
        raise ValueError("algo must be one of: 'ppo', 'reinforce'.")

    set_seed(seed)
    if checkpoint_dir:
        checkpoint_dir = Path(checkpoint_dir)

    if tb_log_dir:
        log_dir = str(tb_log_dir)
    else:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/{algo}/{run_name}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"TensorBoard run: {log_dir}")
    vec_env = AsyncVectorEnv([_make_numpy_env] * n_envs)

    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir:
        path = checkpoint_dir / f"policy_{algo}_ep0.pt"
        torch.save(policy.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

    episode = 0
    rollout_idx = 0
    while episode < episodes:
        rollout_seeds = [seed + rollout_idx * n_envs + i for i in range(n_envs)]
        obs_per, actions_per, old_log_probs_per, rewards_per, st = _rollout_async(
            vec_env, n_envs, reset_seeds=rollout_seeds
        )
        returns_t, ep_totals = _compute_normalized_returns(rewards_per)

        if algo == "ppo":
            metrics = _ppo_update(obs_per, actions_per, old_log_probs_per, returns_t)
        else:
            metrics = _reinforce_update(obs_per, actions_per, returns_t)

        episode += len(ep_totals)
        rollout_idx += 1

        mean_reward = sum(ep_totals) / len(ep_totals)
        mean_steps  = int(sum(st) / len(st))
        ret_abs_mean = float(returns_t.abs().mean().item())
        ret_std = float(returns_t.std(unbiased=False).item())

        writer.add_scalar("train/mean_return", mean_reward, episode)
        writer.add_scalar("train/mean_steps",  mean_steps,  episode)

        if algo == "ppo":
            writer.add_scalar("train/policy_loss", metrics["policy_loss"], episode)
            writer.add_scalar("train/approx_kl",   metrics["approx_kl"], episode)
            writer.add_scalar("train/clip_fraction", metrics["clip_fraction"], episode)
            writer.add_scalar("train/adv_abs_mean", ret_abs_mean, episode)
            writer.add_scalar("train/adv_std", ret_std, episode)

            print(f"Episode {episode:4d} | Algo: {algo} | Envs: {n_envs} | Steps: {mean_steps:4d} | "
                  f"Return: {mean_reward:6.1f} | PolicyLoss: {metrics['policy_loss']:8.4f} | "
                  f"KL: {metrics['approx_kl']:.5f} | ClipFrac: {metrics['clip_fraction']:.3f} | "
                  f"AdvAbs: {ret_abs_mean:.3f} | AdvStd: {ret_std:.3f}")
        else:
            writer.add_scalar("train/loss", metrics["loss"], episode)

            print(f"Episode {episode:4d} | Algo: {algo} | Envs: {n_envs} | Steps: {mean_steps:4d} | "
                  f"Return: {mean_reward:6.1f} | Loss: {metrics['loss']:8.4f}")

        if save_period and checkpoint_dir and episode % save_period == 0:
            path = checkpoint_dir / f"policy_{algo}_ep{episode}.pt"
            torch.save(policy.state_dict(), path)
            print(f"  Checkpoint saved: {path}")

    vec_env.close()
    writer.close()
