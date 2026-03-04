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


def _rollout_async_ppo(vec_env, n_envs):
    """
    Collect trajectories for PPO from a vectorized env.

    Returns:
      - traj:      list of (state, action, reward, old_log_prob)
      - ep_totals: list of total rewards per env
      - steps_per: list of step counts per env
    """
    obs_np, _ = vec_env.reset()                      # (n_envs, 1, 84, 84) float32

    active = np.ones(n_envs, dtype=bool)
    traj = []
    ep_totals = [0.0] * n_envs
    steps_per = [0] * n_envs

    while active.any():
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
        log_probs_all = policy.get_action_log_probabilities(obs_t)   # (n_envs, 6)
        probs = log_probs_all.exp()
        actions = torch.multinomial(probs, 1).squeeze(1)              # (n_envs,)

        actions_np = actions.detach().cpu().numpy().astype(np.int32)
        actions_np[~active] = 0   # NOOP for finished envs

        obs_np, rewards, terminated, truncated, _ = vec_env.step(actions_np)

        for i in range(n_envs):
            if active[i]:
                state_i = obs_t[i]                     # (1,84,84) on DEVICE
                action_i = actions[i]                  # scalar tensor
                reward_i = torch.tensor(float(rewards[i]), device=DEVICE)
                logp_i = log_probs_all[i, actions[i]] # scalar tensor

                traj.append((state_i.detach(), action_i.detach(), reward_i.detach(), logp_i.detach()))
                ep_totals[i] += float(rewards[i])
                steps_per[i] += 1

                if terminated[i] or truncated[i]:
                    active[i] = False

    return traj, ep_totals, steps_per


def _rollout_single_ppo(single_env):
    """
    Collect one full episode trajectory for PPO from a single env.

    Returns:
      - traj:      list of (state, action, reward, old_log_prob)
      - ep_totals: [total_reward]
      - steps_per: [step_count]
    """
    state, _ = single_env.reset()
    state = state.to(DEVICE)
    traj = []
    ep_total = 0.0
    step_count = 0

    while True:
        log_probs_all = policy.get_action_log_probabilities(state.unsqueeze(0))  # (1,6)
        probs = log_probs_all.exp()
        action = torch.multinomial(probs, 1).item()

        next_state, reward, done, _, _ = single_env.step(action)
        next_state = next_state.to(DEVICE)

        logp = log_probs_all[0, action]
        reward_t = torch.tensor(float(reward), device=DEVICE)

        traj.append((state.detach(), torch.tensor(action, device=DEVICE), reward_t.detach(), logp.detach()))
        ep_total += float(reward)
        step_count += 1

        state = next_state
        if done:
            break

    return traj, [ep_total], [step_count]


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
    advantage = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)
    log_probs_t = torch.stack(batch_log_probs)
    loss = -torch.sum(log_probs_t * advantage)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), ep_totals


def ppo(traj, batch_size, itter, clip_eps: float = 0.2):
    """
    PPO update with a linear time-baseline (GAE-style advantage via LS fit).

    Assumes traj is a list of (state, action, reward, old_log_prob) tuples,
    collected sequentially across all envs / episodes in the batch.

    - Returns R_t are discounted with global gamma.
    - Baseline b_t = w1 * t + w0 (t = step index), where w is fitted by
      least-squares over all steps in traj.
    - Advantage A_t = R_t - b_t, normalized over the whole batch.
    - PPO-clip objective with epsilon = clip_eps (default 0.2).
    """
    if len(traj) == 0:
        return

    # Flatten trajectory into tensors
    states = [elem[0] for elem in traj]
    actions = [elem[1] for elem in traj]
    rewards = [elem[2] for elem in traj]
    old_log_probs = [elem[3] for elem in traj]

    states_t = torch.stack(states).to(DEVICE)          # (T, *obs_shape)
    actions_t = torch.stack(actions).to(DEVICE)        # (T,)
    rewards_t = torch.stack(rewards).to(DEVICE)        # (T,)
    old_log_probs_t = torch.stack(old_log_probs).to(DEVICE)  # (T,)

    T = rewards_t.size(0)

    # ------------------------------------------------------------------
    # 1) Discounted returns R_t
    # ------------------------------------------------------------------
    returns = torch.zeros_like(rewards_t, device=DEVICE)
    G = torch.tensor(0.0, device=DEVICE)
    for t in reversed(range(T)):
        G = rewards_t[t] + gamma * G
        returns[t] = G

    # ------------------------------------------------------------------
    # 2) Linear baseline b_t = w1 * t + w0 by least-squares over all steps
    #    phi_t = [t, 1], t = 0,1,...,T-1
    # ------------------------------------------------------------------
    t_idx = torch.arange(T, dtype=torch.float32, device=DEVICE).unsqueeze(1)  # (T,1)
    ones = torch.ones_like(t_idx)
    Phi = torch.cat([t_idx, ones], dim=1)  # (T,2)

    # w = (Phi^T Phi)^(-1) Phi^T R   (2x2 system)
    # Add small ridge term for stability.
    XtX = Phi.T @ Phi  # (2,2)
    Xty = Phi.T @ returns  # (2,)
    ridge = 1e-6 * torch.eye(2, device=DEVICE)
    w = torch.linalg.solve(XtX + ridge, Xty)  # (2,)

    baseline = Phi @ w  # (T,)

    # ------------------------------------------------------------------
    # 3) Advantage = R_t - b_t, normalized over batch
    # ------------------------------------------------------------------
    advantages = returns - baseline
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # ------------------------------------------------------------------
    # 4) PPO update: multiple epochs over shuffled mini-batches
    # ------------------------------------------------------------------
    for _ in range(itter):
        idx = torch.randperm(T, device=DEVICE)

        for start in range(0, T, batch_size):
            end = min(start + batch_size, T)
            mb_idx = idx[start:end]

            mb_states = states_t[mb_idx]
            mb_actions = actions_t[mb_idx]
            mb_old_logp = old_log_probs_t[mb_idx]
            mb_adv = advantages[mb_idx]

            # New log-probs under current policy
            log_probs_all = policy.get_action_log_probabilities(mb_states)  # (B,6)
            mb_actions_long = mb_actions.long().unsqueeze(1)                # (B,1)
            mb_logp = log_probs_all.gather(1, mb_actions_long).squeeze(1)   # (B,)

            ratio = (mb_logp - mb_old_logp).exp()

            unclipped = ratio * mb_adv
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_adv
            loss = -torch.sum(torch.min(unclipped, clipped))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


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


def episode_termination_ppo(
    episodes,
    n_envs=20,
    save_period=None,
    checkpoint_dir=None,
    batch_size: int = 64,
    ppo_epochs: int = 5,
    clip_eps: float = 0.15,
):
    """
    Training loop for PPO using the same env setup as REINFORCE.

    Each iteration:
      - Collect one episode per env (n_envs episodes total).
      - Flatten all steps into traj and run PPO update.
    """
    writer = SummaryWriter(log_dir="runs/ppo")

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
            traj, ep_totals, steps_per = _rollout_async_ppo(vec_env, n_envs)
        else:
            traj, ep_totals, steps_per = _rollout_single_ppo(env)

        # PPO update using all steps from all envs
        bs = min(batch_size, len(traj))
        ppo(traj, bs, ppo_epochs, clip_eps=clip_eps)

        episode += len(ep_totals)

        mean_reward = sum(ep_totals) / len(ep_totals)
        mean_steps  = int(sum(steps_per) / len(steps_per))

        writer.add_scalar("train/mean_return", mean_reward, episode)
        writer.add_scalar("train/mean_steps",  mean_steps,  episode)

        print(f"[PPO] Episode {episode:4d} | Envs: {n_envs} | Steps: {mean_steps:4d} | "
              f"Return: {mean_reward:6.1f}")

        if save_period and checkpoint_dir and episode % save_period == 0:
            path = checkpoint_dir / f"policy_ep{episode}.pt"
            torch.save(policy.state_dict(), path)
            print(f"  Checkpoint saved: {path}")

    if vec_env is not None:
        vec_env.close()
    writer.close()
