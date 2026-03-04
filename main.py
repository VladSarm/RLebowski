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


def train_reinforce(
    episodes: int,
    gamma: float,
    lr: float,
    algo: str = "ppo",
    checkpoint: str = None,
    save_period: int = 10,
    n_envs: int = 1,
    seed: int = 42,
    ppo_clip: float = 0.2,
    ppo_epochs: int = 10,
    ppo_mini_batches: int = 2,
    checkpoint_dir: str | None = None,
    tb_log_dir: str | None = None,
):
    """Train policy with selectable algorithm (PPO or REINFORCE) on batch vector rollouts."""
    from reinforce import episode_termination, policy

    if n_envs < 1:
        raise ValueError("n_envs must be >= 1.")

    checkpoint_root = Path(checkpoint_dir) if checkpoint_dir else CHECKPOINT_DIR
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    print(f"Training {algo.upper()} on Bowling")
    print(f"  Device: {DEVICE}")
    print(f"  Episodes: {episodes}")
    print(f"  Gamma: {gamma}")
    print(f"  LR: {lr}")
    print(f"  Algo: {algo}")
    print(f"  Seed: {seed}")
    print(f"  Parallel envs per update: {n_envs}")
    print(f"  Save period: {save_period} episodes")
    print(f"  Checkpoint dir: {checkpoint_root}")
    print(f"  TensorBoard dir: {tb_log_dir if tb_log_dir else '(auto)'}")
    if algo == "ppo":
        print(f"  PPO clip: {ppo_clip}")
        print(f"  PPO epochs: {ppo_epochs}")
        print(f"  PPO mini-batches: {ppo_mini_batches}")
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
    reinforce.ppo_clip_epsilon = ppo_clip
    reinforce.ppo_update_epochs = ppo_epochs
    reinforce.ppo_num_mini_batches = ppo_mini_batches
    for param_group in reinforce.optimizer.param_groups:
        param_group['lr'] = lr

    episode_termination(
        episodes,
        n_envs=n_envs,
        save_period=save_period,
        checkpoint_dir=checkpoint_root,
        algo=algo,
        seed=seed,
        tb_log_dir=tb_log_dir,
    )

    # Save final checkpoint
    checkpoint_path = checkpoint_root / f"policy_{algo}_ep{episodes}_gamma{gamma}_lr{lr:.0e}.pt"
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
    parser = argparse.ArgumentParser(description="PPO/REINFORCE agent on Bowling")

    subparsers = parser.add_subparsers(dest="command", help="Command: train or eval")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train PPO or REINFORCE (batch-only)")
    train_parser.add_argument("--algo", type=str, choices=["ppo", "reinforce"], default="ppo", help="Training algorithm")
    train_parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    train_parser.add_argument("--gamma", type=float, default=0.999, help="Discount factor")
    train_parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    train_parser.add_argument("--checkpoint", type=str, default=None, help="Load checkpoint")
    train_parser.add_argument("--save-period", type=int, default=10, help="Save checkpoint every N episodes")
    train_parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel environments per update (>=1)")
    train_parser.add_argument("--seed", type=int, default=42, help="Global RNG seed")
    train_parser.add_argument("--ppo-clip", type=float, default=0.2, help="PPO clip epsilon")
    train_parser.add_argument("--ppo-epochs", type=int, default=10, help="PPO learning epochs per rollout")
    train_parser.add_argument("--ppo-mini-batches", type=int, default=2, help="PPO mini-batches per epoch")
    train_parser.add_argument("--checkpoint-dir", type=str, default=None, help="Directory to store checkpoints for this run")
    train_parser.add_argument("--tb-log-dir", type=str, default=None, help="Directory for TensorBoard logs of this run")

    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Evaluate trained policy")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    eval_parser.add_argument("--episodes", type=int, default=5, help="Number of eval episodes")

    args = parser.parse_args()

    if args.command == "train":
        train_reinforce(
            algo=args.algo,
            episodes=args.episodes,
            gamma=args.gamma,
            lr=args.lr,
            checkpoint=args.checkpoint,
            save_period=args.save_period,
            n_envs=args.n_envs,
            seed=args.seed,
            ppo_clip=args.ppo_clip,
            ppo_epochs=args.ppo_epochs,
            ppo_mini_batches=args.ppo_mini_batches,
            checkpoint_dir=args.checkpoint_dir,
            tb_log_dir=args.tb_log_dir,
        )
    elif args.command == "eval":
        eval_policy(
            checkpoint=args.checkpoint,
            episodes=args.episodes,
        )
    else:
        parser.print_help()
