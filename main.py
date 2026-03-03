import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, 'src')

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def train_reinforce(episodes: int, gamma: float, lr: float, checkpoint: str = None, save_period: int = 10, n_envs: int = 1):
    """Train PPO-Clip (actor-only) on Bowling with batch-only vector rollouts."""
    from reinforce import episode_termination, policy

    if n_envs < 1:
        raise ValueError("n_envs must be >= 1.")

    print(f"Training PPO-Clip on Bowling")
    print(f"  Device: {DEVICE}")
    print(f"  Episodes: {episodes}")
    print(f"  Gamma: {gamma}")
    print(f"  LR: {lr}")
    print(f"  PPO mode: actor-only, full-batch")
    print(f"  Parallel envs per update: {n_envs}")
    print(f"  Save period: {save_period} episodes")
    print()

    # Load checkpoint if provided
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.exists():
            print(f"Loading checkpoint from {checkpoint_path}...")
            state_dict = torch.load(checkpoint_path, map_location=DEVICE)
            policy.load_state_dict(state_dict)
            policy.to(DEVICE)
        else:
            print(f"Warning: checkpoint {checkpoint_path} not found, starting fresh")

    # Apply hyperparameters
    import reinforce
    reinforce.gamma = gamma
    for param_group in reinforce.optimizer.param_groups:
        param_group['lr'] = lr

    episode_termination(episodes, n_envs=n_envs, save_period=save_period, checkpoint_dir=CHECKPOINT_DIR)

    # Save final checkpoint
    checkpoint_path = CHECKPOINT_DIR / f"policy_ppo_clip_ep{episodes}_gamma{gamma}_lr{lr:.0e}.pt"
    torch.save(policy.state_dict(), checkpoint_path)
    print(f"\n✓ Final checkpoint saved to {checkpoint_path}")


def eval_policy(checkpoint: str, episodes: int = 5):
    """Просмотр обученной политики в действии."""
    import reinforce
    from lebowski_character import draw_lebowski_game, replace_bowler
    import matplotlib.pyplot as plt
    import gymnasium as gym
    import ale_py

    gym.register_envs(ale_py)

    print(f"Loading policy from {checkpoint}...")
    state_dict = torch.load(checkpoint, map_location=DEVICE)
    reinforce.policy.load_state_dict(state_dict)
    reinforce.policy.to(DEVICE)
    reinforce.policy.eval()

    dude = draw_lebowski_game()

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.set_title(f"Policy Evaluation ({episodes} episodes)")
    ax.axis("off")

    episode_rewards = []

    for ep in range(episodes):
        state, _ = reinforce.env.reset()
        state = state.to(DEVICE)
        rewards = []

        with torch.no_grad():
            while True:
                probs = reinforce.policy.get_action_probabilities(state.unsqueeze(0))
                action = torch.multinomial(probs, 1).item()
                next_state, reward, done, _, _ = reinforce.env.step(action)
                next_state = next_state.to(DEVICE)
                rewards.append(reward)

                # Рисуем с нашим персонажем
                frame = replace_bowler(reinforce.env.env.render(), dude)
                ax.clear()
                ax.imshow(frame)
                ax.axis("off")
                ax.set_title(f"Episode {ep+1}/{episodes} | Return: {sum(rewards):.1f}")
                plt.pause(0.01)

                state = next_state
                if done:
                    break

        total = sum(rewards)
        episode_rewards.append(total)
        print(f"Episode {ep+1}: Return = {total:.1f}")

    print(f"\nAverage return: {sum(episode_rewards) / len(episode_rewards):.2f}")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO-Clip agent on Bowling")

    subparsers = parser.add_subparsers(dest="command", help="Command: train or eval")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train PPO-Clip (batch-only)")
    train_parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    train_parser.add_argument("--gamma", type=float, default=0.998, help="Discount factor")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint")
    train_parser.add_argument("--save-period", type=int, default=10, help="Save checkpoint every N episodes")
    train_parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments per PPO update (>=1)")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained policy")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    eval_parser.add_argument("--episodes", type=int, default=5, help="Number of eval episodes")

    args = parser.parse_args()

    if args.command == "train":
        train_reinforce(
            episodes=args.episodes,
            gamma=args.gamma,
            lr=args.lr,
            checkpoint=args.checkpoint,
            save_period=args.save_period,
            n_envs=args.n_envs,
        )
    elif args.command == "eval":
        eval_policy(
            checkpoint=args.checkpoint,
            episodes=args.episodes,
        )
    else:
        parser.print_help()
