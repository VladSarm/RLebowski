import torch
import bowling_env
import mlp
from torch.utils.tensorboard import SummaryWriter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy = mlp.PolicyNetwork().to(DEVICE)
env = bowling_env.BowlingThrowEnv()  # kept for eval in main.py
optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
gamma = 0.998


def episode_termination(episodes, n_envs=1, save_period=None, checkpoint_dir=None):
    writer = SummaryWriter(log_dir="runs/reinforce")
    envs = [bowling_env.BowlingThrowEnv() for _ in range(n_envs)]

    if checkpoint_dir:
        path = checkpoint_dir / "policy_ep0.pt"
        torch.save(policy.state_dict(), path)
        print(f"  Checkpoint saved: {path}")

    episode = 0
    while episode < episodes:
        batch_log_probs = []
        batch_returns = []
        batch_rewards = []
        batch_steps = []

        for env_i in envs:
            if episode >= episodes:
                break
            state, _ = env_i.reset()
            state = state.to(DEVICE)
            log_probs, rewards = [], []
            step_count = 0

            while True:
                log_probs_all = policy.get_action_log_probabilities(state.unsqueeze(0))
                probs = log_probs_all.exp()
                action = torch.multinomial(probs, 1).item()
                next_state, reward, done, _, _ = env_i.step(action)
                log_probs.append(log_probs_all[0, action])
                rewards.append(reward)
                state = next_state.to(DEVICE)
                step_count += 1
                if done:
                    break

            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G/10)

            batch_log_probs.extend(log_probs)
            batch_returns.extend(returns)
            batch_rewards.append(sum(rewards))
            batch_steps.append(step_count)
            episode += 1

        returns_tensor = torch.tensor(batch_returns, dtype=torch.float32, device=DEVICE)
        # Normalize: actions above average get positive signal, below — negative
        returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)
        log_probs_tensor = torch.stack(batch_log_probs)
        loss = -torch.sum(log_probs_tensor * returns_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mean_reward = sum(batch_rewards) / len(batch_rewards)
        mean_steps = int(sum(batch_steps) / len(batch_steps))

        writer.add_scalar("train/mean_return", mean_reward, episode)
        writer.add_scalar("train/loss", loss.item(), episode)
        writer.add_scalar("train/mean_steps", mean_steps, episode)

        print(f"Episode {episode:4d} | Steps: {mean_steps:4d} | Return: {mean_reward:6.1f} | Loss: {loss.item():8.4f}")

        if save_period and checkpoint_dir and episode % save_period == 0:
            path = checkpoint_dir / f"policy_ep{episode}.pt"
            torch.save(policy.state_dict(), path)
            print(f"  Checkpoint saved: {path}")

    writer.close()
