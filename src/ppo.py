import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.vector import AsyncVectorEnv
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

import bowling_env
import mlp

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = mlp.PolicyNetwork().to(DEVICE)
env = bowling_env.BowlingThrowEnv()  # kept for eval in main.py
optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
gamma = 0.998

class _NumpyEnv(gym.Env):
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

def _rollout_async(vec_env, n_envs):
    obs_np, _ = vec_env.reset()                      # (n_envs, 1, 84, 84) float32

    active = np.ones(n_envs, dtype=bool)
    log_probs_per = [[] for _ in range(n_envs)]
    rewards_per = [[] for _ in range(n_envs)]
    steps_per = [0]  * n_envs

    obs_per = [[] for _ in range(n_envs)]
    actions_per = [[] for _ in range(n_envs)]

    while active.any():
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=DEVICE)
        
        with torch.no_grad():
            log_probs_all = policy.get_action_log_probabilities(obs_t)   # (n_envs, 6)
        probs = log_probs_all.exp()
        actions = torch.multinomial(probs, 1).squeeze(1)              # (n_envs,)

        actions_np = actions.detach().cpu().numpy().astype(np.int32) # (n_envs,)
        actions_np[~active] = 0   # NOOP for already-finished envs

        obs_np, rewards, terminated, truncated, _ = vec_env.step(actions_np)

        for i in range(n_envs):
            if active[i]:
                log_probs_per[i].append(float(log_probs_all[i, actions[i]].item()))
                rewards_per[i].append(float(rewards[i]))
                steps_per[i] += 1
                obs_per[i].append(obs_t[i].detach().cpu().numpy())
                actions_per[i].append(int(actions[i].item()))
                if terminated[i] or truncated[i]:
                    active[i] = False

    # make tensors obs_t, actions_t, log_probs_t
    batch_obs = []
    batch_actions = []
    batch_log_probs = []

    for i in range(n_envs):
        assert len(obs_per[i]) == len(actions_per[i]) and len(obs_per[i]) == len(log_probs_per[i]) and len(obs_per[i]) == len(rewards_per[i]) # tau_i
        batch_obs.extend(obs_per[i])
        batch_actions.extend(actions_per[i])
        batch_log_probs.extend(log_probs_per[i])

    obs_t = torch.tensor(np.asarray(batch_obs), dtype=torch.float32, device=DEVICE)
    actions_t = torch.tensor(batch_actions, dtype=torch.long, device=DEVICE)
    log_probs_t = torch.tensor(batch_log_probs, dtype=torch.float32, device=DEVICE)
    
    return rewards_per, steps_per, obs_t, actions_t, log_probs_t

def _phi_avgpool_8x8(obs_t: torch.Tensor) -> torch.Tensor:
    """
    obs_t: (N, 1, 84, 84) float32
    returns Phi: (N, 65) where 64 pooled features + bias term 1
    """
    x = F.adaptive_avg_pool2d(obs_t, (8, 8))   # (N,1,8,8)
    x = x.flatten(1)                           # (N,64)
    ones = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
    Phi = torch.cat([x, ones], dim=1)          # (N,65) bias included
    return Phi

def _ls_baseline_advantages(obs_t: torch.Tensor,
                            returns_raw_t: torch.Tensor,
                            ridge: float = 1e-2) -> torch.Tensor:
    """
    returns_raw_t: (N,) Monte-Carlo returns G_t (NOT normalized)
    Output: advantages (N,) = G_t - Phi w, then normalized.
    """
    Phi = _phi_avgpool_8x8(obs_t)                       # (N,65)
    y = returns_raw_t.unsqueeze(1)                      # (N,1)

    # Solve w = (Phi^T Phi + ridge I)^(-1) Phi^T y
    A = Phi.T @ Phi
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    A = A + ridge * I
    b = Phi.T @ y
    w = torch.linalg.solve(A, b)                        # (65,1)

    baseline = (Phi @ w).squeeze(1)                     # (N,)
    adv = returns_raw_t - baseline                      # (N,)

    # normalize advantages 
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    return adv.detach()

def _compute_returns(rewards_per):
    n_envs = len(rewards_per)
    returns_per = [[] for _ in range(n_envs)]
    ep_totals = []

    for i in range(n_envs):
        G = 0
        for r in reversed(rewards_per[i]):
            G = r + gamma * G
            returns_per[i].insert(0, G)
        ep_totals.append(sum(rewards_per[i]))
    
    # make tensor returns_t
    batch_returns = []
    for i in range(n_envs):
        batch_returns.extend(returns_per[i])
    returns_t = torch.tensor(batch_returns, dtype=torch.float32, device=DEVICE)
    # normalize
    #returns_t = (returns_t - returns_t.mean()) / (returns_t.std(unbiased=False) + 1e-8)
    return returns_t, ep_totals



def _ppo_update(obs_t, actions_t, old_log_probs_t, advantages_t, n_updates_per_iteration=5, minibatch_size=256, clip_eps=0.2, max_grad_norm=5):
    old_log_probs_t = old_log_probs_t.detach()
    advantages_t = advantages_t.detach()
    batch_size = obs_t.shape[0]
    inds = np.arange(batch_size)

    losses = []
    for _ in range(n_updates_per_iteration):
        np.random.shuffle(inds)
        for start in range(0, batch_size, minibatch_size):
            mb = inds[start:start + minibatch_size]
            logp_all = policy.get_action_log_probabilities(obs_t[mb]) # obs_t[mb] of shape (mb_size, 1, 84, 84), logp_all of shape (mb_size, n_actions=6)
            # logp_new[i]=logp_all[i,actions[i]]
            logp_new = logp_all.gather(dim=1, index=actions_t[mb].unsqueeze(1)).squeeze(1) # (mb_size,)
            ratio = torch.exp(logp_new - old_log_probs_t[mb])

            unclipped = ratio * advantages_t[mb]
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_t[mb]
            surrogate = torch.min(unclipped, clipped).mean()

            loss = -surrogate  # maximize surrogate

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            losses.append(float(loss.item()))

    return float(np.mean(losses))


def episode_termination(episodes, n_envs=2, save_period=None, checkpoint_dir=None):
    writer = SummaryWriter(log_dir="runs/reinforce")

    assert n_envs > 1, 'Implemented for n_envs > 1'
    vec_env = AsyncVectorEnv([_make_numpy_env] * n_envs)

    if checkpoint_dir:
        path = checkpoint_dir / "policy_ep0.pt"
        torch.save(policy.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

    episode = 0
    while episode < episodes:
        # rollout under π_old (current policy) 
        rw, st, obs_t, actions_t, log_probs_t = _rollout_async(vec_env, n_envs) # log_probs_per, rewards_per, steps_per, obs_per, actions_per
        # compute A^{π_old} estimate 
        returns_t, ep_totals = _compute_returns(rw)
        advantages_t = _ls_baseline_advantages(obs_t, returns_t)
        # PPO-clip optimization
        loss_val = _ppo_update(obs_t, actions_t, log_probs_t, advantages_t)

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
        
        episode += n_envs

    if vec_env is not None:
        vec_env.close()
    writer.close()
