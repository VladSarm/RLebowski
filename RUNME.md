# RLebowski — Run & Code

How to install, run tests, start training, and evaluate policies. For background and algorithm details, see [README.md](README.md).

---

## Table of Contents
1. [Repository Structure](#repository-structure)
2. [Installation](#installation)
3. [Running Tests](#running-tests)
4. [Training](#training)
5. [Interactive Play](#interactive-play)
6. [Evaluation](#evaluation)

---

## Repository Structure
```text
RLebowski/
├── src/                       # Core RL components
│   ├── bowling_env.py         # Bowling environment wrapper
│   ├── lebowski_character.py  # Character-specific utilities
│   ├── reinforce.py           # REINFORCE & PPO algorithms
│   ├── mlp.py                 # Neural network policy
│   ├── play_interactive.py    # Interactive play mode
│   └── test_gym.py            # Basic tests
├── scripts/                   # Experiment runner
│   └── run_experiment_suite.py
├── main.py                    # Main CLI entry point
├── checkpoints/               # Saved policy checkpoints
├── runs/                      # TensorBoard logs
├── pyproject.toml             # Project dependencies
└── uv.lock                    # Dependency lock file
```

---

## Installation

The project uses `uv` for dependency management.

1. Install dependencies:
```bash
cd RLebowski
uv sync
```

2. Run with project environment:
```bash
uv run python main.py --help
uv run python main.py train --help
uv run python main.py eval --help
```

---

## Running Tests

Run basic tests to verify environment and policy setup:
```bash
cd RLebowski
uv run pytest src/test_gym.py -v
```

The tests cover:
- Gymnasium environment loading and stepping,
- policy initialization,
- basic forward pass sanity checks.

---

## Training

### Training CLI

Main command:
```bash
uv run python main.py train \
  --episodes <N> \
  --algo ppo \
  [options]
```

Main arguments:
- `--episodes` (required): Total episodes to train for
- `--algo` (required): Algorithm choice — `ppo` or `reinforce` (default: `ppo`)
- `--gamma`: Discount factor (default: `0.99`)
- `--lr`: Learning rate (default: `1e-3`)
- `--seed`: Random seed (default: `42`)
- `--checkpoint`: Load checkpoint from path (optional)
- `--checkpoint-dir`: Directory to save checkpoints (default: `checkpoints/`)
- `--tb-log-dir`: Directory for TensorBoard logs (default: `runs/`)
- `--save-period`: Save checkpoint every N episodes (default: `10`)
- `--n-envs`: Number of parallel environments (default: `1`)

PPO-specific:
- `--ppo-clip`: Clipping range ε (default: `0.2`)
- `--ppo-epochs`: Optimization epochs per batch (default: `10`)
- `--ppo-mini-batches`: Number of mini-batches per epoch (default: `2`)

### Examples

**Basic PPO training:**
```bash
uv run python main.py train \
  --episodes 100000 \
  --algo ppo \
  --gamma 0.999 \
  --lr 1e-3 \
  --seed 42
```

**PPO with parallel environments:**
```bash
uv run python main.py train \
  --episodes 50000 \
  --algo ppo \
  --n-envs 8 \
  --gamma 0.999 \
  --lr 3e-4 \
  --ppo-clip 0.2 \
  --ppo-epochs 10 \
  --tb-log-dir runs/ppo_baseline
```

**REINFORCE training:**
```bash
uv run python main.py train \
  --episodes 100000 \
  --algo reinforce \
  --gamma 0.999 \
  --lr 1e-4 \
  --n-envs 4
```

**Resume from checkpoint:**
```bash
uv run python main.py train \
  --episodes 100000 \
  --algo ppo \
  --checkpoint checkpoints/policy_ep50000.pt \
  --save-period 5
```

### Monitoring Training

View TensorBoard logs:
```bash
tensorboard --logdir runs/
```

Then open `http://localhost:6006` in your browser.

---

## Interactive Human Play

Play the Bowling game manually (human player):

```bash
uv run python src/play_interactive.py
```

Controls:
- **↑↓** — Move/steer
- **SPACE** — Throw
- **SHIFT+↑↓** — Throw + steer
- **R** — Reset game
- **Q** — Quit

This lets you manually test the game mechanics before or after training.

---

## Evaluation

### 1) Policy Evaluation with Visualization

Run evaluation on a trained checkpoint with visual display:
```bash
uv run python main.py eval \
  --checkpoint checkpoints/policy_ppo_ep10000.pt \
  --episodes 5
```

Shows:
- Rendered gameplay with agent's actions
- Return (total reward) per episode
- Average return across episodes
- Includes Lebowski character overlay

### 2) TensorBoard Logs

Automatically logged during training to monitor progress:
- Episode return over time
- Episode length
- Policy loss
- Value loss (PPO only)
- Training efficiency

View with:
```bash
tensorboard --logdir runs/
```

Then open `http://localhost:6006` in your browser.

---

## System Architecture

```text
                    +-------------------+
                    | main.py (CLI)     |
                    +---------+---------+
                              |
                 +------------+------------+
                 |                         |
          +------v------+          +------v-------+
          | BowlingEnv  |          | PPO / REFO  |
          | (gymnasium) |          | Algorithm   |
          +------+------+          +------+------+
                 |                        |
          +------v------+          +------v-------+
          | Atari Bowl  |<-------->| MLP Policy  |
          |  simulator  |   HTTP   | (torch)     |
          +-------------+          +-------------+
```

---
