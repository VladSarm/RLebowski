# 🎳 RLebowski

> **RLebowski** is a reinforcement learning research project for mastering **Atari Bowling** with modern policy gradient algorithms.

The repository combines:
- a **Gymnasium environment wrapper** for Atari Bowling,
- policy-gradient training with **PPO** and **REINFORCE**,
- PyTorch-based policy networks and optimization,
- TensorBoard integration for experiment tracking.

📌 **Installation, commands, and run instructions:** see [RUNME.md](RUNME.md).

---

## 📑 Table of Contents
1. [Project Overview](#project-overview)
2. [Why Bowling Is Interesting](#why-bowling-is-interesting)
3. [Algorithms](#algorithms)
4. [Environment & State Space](#environment--state-space)
5. [Training](#training)
6. [Experiments & Results](#experiments--results)
7. [Current Limitations](#current-limitations)

---

## 🧩 Project Overview

The project focuses on RL for a visual, procedural control task:
- **Environment**: Atari Bowling (Gymnasium-based).
- **Action space**: 6 discrete actions (movement, aim, throw).
- **State**: Raw pixel observations (84×84×4 with frame stacking).
- **Policy**: Multi-layer perceptron with configurable hidden layers.
- **Algorithms**: PPO (Proximal Policy Optimization) and REINFORCE (policy gradient).

**Main goals:**
- Implement clean, understandable implementations of PPO and REINFORCE.
- Achieve stable training on a continuous control task with visual input.
- Provide infrastructure for experiment tracking (TensorBoard).
- Demonstrate the practical differences between REINFORCE and PPO convergence.

---

## 💪 Why Bowling Is Interesting

Even for a simplified Atari task, bowling presents real RL challenges:

- **Visual processing**: Agent must interpret raw pixels to determine ball position, lane alignment, and pins.
- **Continuous control**: Actions are discrete but their effect on ball trajectory is highly non-linear.
- **Sparse rewards**: Success is sparse — you get a large reward only when you knock down pins.
- **Exploration-exploitation tradeoff**: Naive exploration (random actions) rarely produces meaningful scores; the agent must learn structured behavior.
- **Credit assignment**: Understanding which actions in a trajectory led to knocking down distant pins requires temporal reasoning.

From an engineering perspective:
- Pre-processing raw images for efficient learning.
- Balancing sample efficiency with computational cost.
- Stable gradient estimation under batch policy updates.

---

## 🎮 Simulator & Environment

![Bowling Gameplay](assets/not_trained-2026-03-04_20.44.34.gif)

### Environment Wrapper

The `BowlingThrowEnv` class wraps the Gymnasium Atari Bowling environment (`ALE/Bowling-v5`):
- Uses **mode=2** for the Bowling variant.
- ROI (Region of Interest) preprocessing: extracts the relevant game area from raw frames.
- Episode runs naturally until game completion (auto-termination by Bowling rules).
- All observations converted to torch tensors for GPU batch processing.

### State and Observation Format

- **Raw Atari output**: 210×160×3 RGB images per frame.
- **ROI extraction**: Crops rows 100–175 from a single color channel (channel 2).
- **Preprocessed state**: Shape `(1, 75, 160)` — single channel, height 75, width 160.
- **Network input**: Flattened to `1 * 75 * 160 = 12000` features.

The ROI captures the bowling lane and ball trajectory, removing irrelevant screen areas.

### Action Space (6 actions)

| Action Index | Meaning |
|---|---|
| 0 | NOOP (no operation) |
| 1 | FIRE (release/throw) |
| 2 | UP (vertical aim up) |
| 3 | DOWN (vertical aim down) |
| 4 | UPFIRE (aim up + throw) |
| 5 | DOWNFIRE (aim down + throw) |

No "power level" — just aim (vertical) + throw timing.

### Reward Structure

- **Base reward**: ALE native score (pins knocked down).
- **Step penalty**: –0.001 per step (encourages faster solutions).
- **Strike bonus**: +20 additional reward when strike detected (frame reward ≥ 10).
- **Episode termination**: Automatic when game ends (10 frames complete in Bowling mode).

---

## 🖼️ Image Processing Pipeline

### Frame Preprocessing Steps

Raw Atari Bowling frames undergo a series of preprocessing steps to extract relevant game information and reduce dimensionality:

**1. Color Channel Extraction (Grayscale)**
- Raw frame: `210×160×3` RGB pixels
- Extract single channel (blue, channel 2) which best highlights the bowling lane and pins
- Result: `210×160` grayscale values

**2. Region of Interest (ROI) Cropping**
- Crop rows **100–175** from the grayscale frame
- Removes score bar at top, irrelevant UI at bottom
- Focuses on the active bowling lane where ball trajectory and pins are visible
- Result: `75×160` region (see image below)

![ROI Extraction Example](assets/Picture_crop.jpg)

**3. Normalization**
- Scale pixel values from `[0, 255]` to `[0, 1]` by dividing by 255
- Improves numerical stability for neural network training
- Result: `1×75×160` tensor (1 channel, height, width)

```python
def preprocess(frame: np.ndarray) -> torch.Tensor:
    roi = frame[100:175, :, 2].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(roi)).unsqueeze(0)
```

### Convolutional Feature Extraction

Instead of flattening the `75×160` image directly (12,000 features), we use **CNN layers** to automatically learn spatial features:

**CNN Architecture:**
```
Input:  (batch, 1, 75, 160)  [ROI channel]
           ↓
Conv2d(1 → 16, kernel=3×3, stride=2, padding=2)
           ↓
    ReLU activation
           ↓
Conv2d(16 → 5, kernel=2×2, stride=1, padding=0)
           ↓
    ReLU activation
           ↓
Output: (batch, 5, 38, 80)  [5 feature maps]
           ↓
Flatten to (batch, 15200)
           ↓
MLP (15200 → 512 → 6 actions)
```

**Why CNN?**
- **Spatial locality**: Convolutional filters capture local patterns (ball edge, pin arrangement).
- **Translation invariance**: Same pin pattern recognized regardless of exact position on lane.
- **Dimensionality reduction**: Reduce 12,000 raw pixels to 15,200 learned features (compared to full-image MLPs which require more parameters).
- **Faster training**: Shared weights across spatial regions reduce overfitting.

**Feature Maps:**
- `Conv1` (16 filters): Detects low-level edges, textures, ball contours
- `Conv2` (5 filters): Combines low-level features into higher-level patterns (pins, lane structure)

---

## 📐 Algorithms

### Notation and Preliminaries
Notation follows standard REINFORCE lecture style. Random variables are denoted by capital letters, data points by lowercase letters.
- $\mathbb{S}$ — state space, $s \in \mathbb{S}$ — state; $\mathbb{A}$ — action space, $a \in \mathbb{A}$ — action.
- Environment: $S_{t+1} \sim p(\cdot\mid S_t, A_t)$
- Policy: $A_t \sim \pi^{\theta}(\cdot\mid S_t)$
- Reward: $R_t \sim p^{R}(\cdot\mid S_t, A_t)$
- Trajectory: $z_{0:\tau} = \{(s_0,a_0), (s_1,a_1), \dots, (s_{\tau-1},a_{\tau-1})\}$, where $\tau$ is the length of the episode (time horizon)

The **value function** is the expected discounted return starting from state $s$:

$$
V^\theta(s) =
\mathbb{E}_{\pi^\theta}
\left[
\sum_{t=0}^{\infty} \gamma^t R_t
\mid S_0 = s
\right]
$$

The **Q-function** is the expected discounted return starting from state $s$ and action $a$:

$$
Q^\theta(s,a) =
\mathbb{E}_{\pi^\theta}
\left[
\sum_{t=0}^{\infty} \gamma^t R_t
\mid S_0 = s, A_0 = a
\right]
$$

The **advantage function** measures how much better action $a$ is compared to the average action at state $s$:

$$
A^\theta(s,a) = Q^\theta(s,a) - V^\theta(s)
$$


### REINFORCE 
Policy objective (definition):
```math
J(\theta) = \mathbb{E}_{\pi^\theta}\left[\sum_{t=0}^{\tau-1} \gamma^t r(S_t, A_t)\right],
```
where $\gamma \in \left[0,1\right]$ is the discount factor.

Gradient estimator:
```math
\nabla_\theta J(\theta)
=
\mathbb{E}_{\pi^\theta}
\left[
\sum_{t=0}^{\tau-1}
\gamma^t R_t \nabla_\theta \log \pi^\theta (A_t \mid S_t)
\right]
```
where
```math
R_t = \sum_{k=t}^{\tau-1} \gamma^{k-t} r(S_k, A_k).
```

The idea is to perform gradient ascent
```math
\theta \longleftarrow \theta + \alpha \cdot \nabla_\theta J(\theta)
```

**Pros:**
- Unbiased gradient estimates.
- Simple implementation and interpretation.

**Cons:**
- High variance in gradient updates (sample inefficient).
- Can have unstable training curves.

### PPO (Proximal Policy Optimization)
Proximal policy optimization (PPO) is an advanced tweak of TRPO that employs policy PDF ratio and surrogate objective **clipping** instead of constraints or penalties.

The (statistical) surrogate objective in PPO reads:

```math
\hat{L}_{\text{PPO}} :=
\mathbb{E}_{T \sim \mathrm{Unif}[0, \tau_{\text{b}} - 1]}
\left[
\min \left\{
\frac{\pi_{\text{new}}(a_T \mid s_T)}{\pi_{\text{old}}(a_T \mid s_T)}
A^{\pi_{\text{old}}}(s_T, a_T),
\;
\mathrm{Clip}_{1-\epsilon}^{1+\epsilon}
\left(
\frac{\pi_{\text{new}}(a_T \mid s_T)}{\pi_{\text{old}}(a_T \mid s_T)}
\right)
A^{\pi_{\text{old}}}(s_T, a_T)
\right\}
\right]
```

where:
- $A^{\pi_{\text{old}}}(s_T, a_T)$ is the advantage estimate
- $\epsilon$ is the clipping range (typically 0.1 to 0.2)
- $\tau_b$  is a mini-batch size, not necessarily equal  $\tau$

As can be seen, both the surrogate objectives in PPO requires the advantage function. In practice, neither the value function, nor Q-function, nor advantage function is known. So, we use the advantage estimate.

**Advantages:**
- Lower variance through importance sampling.
- Clipping prevents overshooting; stable updates.
- Often converges faster and more reliably.

---

## 📊 Training

### Policy Network Architecture

Standard **3-layer MLP**:

```
Input (12000 features from flattened 1×75×160 ROI)
        ↓
  Dense(12000 → 512)
        ↓
      ReLU
        ↓
  Dense(512 → 256)
        ↓
      ReLU
        ↓
  Dense(256 → 6)  [policy head]
        ↓
    Softmax
        ↓
  π_θ(a|s) [action probabilities over 6 actions]
```

For PPO, an additional **value head**:
```
  Dense(256 → 1)
        ↓
      V(s)  [baseline/value estimate]
```

### Hyperparameters

Default settings from code:

**PPO & REINFORCE:**
- Learning rate: `1e-3`
- Discount factor γ: `0.999`
- Number if envs: `5`

**PPO-specific:**
- Clipping range ε: `0.2`
- Optimization epochs: `10` per batch
- Mini-batches per epoch: `2`
- Number if envs: `5`

### Optimization

Both algorithms use:
- **Optimizer**: Adam with default β₁=0.9, β₂=0.999
- **Batch updates**: Samples collected over N parallel environments
- **Loss**:
  - REINFORCE: `-E[log π_θ(a|s) · G_t]`
  - PPO: `L^CLIP(θ) - c_v L^VALUE(θ) + c_e H[π_θ]` (with optional entropy bonus)


---

## 📊 Experiments & Results

### REINFORCE vs PPO Trained Agents

**REINFORCE trained agent:**

![REINFORCE Trained](assets/reinforce_trained-2026-03-04_21.53.17.gif)

**PPO trained agent:**

![PPO Trained](assets/ppo_trained-2026-03-04_21.32.07.gif)

Both agents successfully learn to control the ball and knock down pins. PPO typically converges faster with more stable training dynamics compared to vanilla REINFORCE, which exhibits higher variance during learning.

### Algorithm Performance Comparison

![PPO vs REINFORCE](assets/PPO_VS_REINFORCE.jpg)

PPO achieves higher cumulative reward performance on the same training horizon.

---

#### Total Return

![PPO Total Return](assets/PPO_RETURN.jpg)

Total return with default hyperparameters.


### PPO Hyperparameter Tuning Study

We conducted several hyperparameter tuning experiments to investigate their effects on PPO training.

#### Epochs Per Batch

![PPO Epochs](assets/PPO_per_epoch.jpg)

Different numbers of gradient steps per batch. More optimization epochs make training smoother and result in higher final returns.

#### Mini-Batch Size

![PPO Mini-Batches](assets/PPO_minibatches.jpg)

Effect of mini-batch count. Larger mini-batches (10+) produce less noisy training but slower convergence, while fewer mini-batches (1-2) enable faster training.

#### Clipping Range (ε)

![PPO Clipping Range](assets/PPO_per_clip.jpg)

Effect of clipping parameter. The standard value ε = 0.2 provides a compromise between ε = 0.5 (noisy but fast growth) and ε = 0.005 (smooth but slow growth).

---

### 🧠 Value Function (PPO only)

The value network `V(s)` estimates the expected return from state s:

```
V(s) = E[G_t | s_t = s]
```

Trained with MSE loss:
```
L^VALUE(θ) = (V_θ(s_t) - G_t)^2
```

Advantages `A_t = G_t - V(s_t)` used for policy updates. A good value function reduces variance significantly.

---

## 🔧 Current Limitations



---

## 🚀 Future Directions

- Implement **A3C** (Asynchronous Advantage Actor-Critic) for true distributed training.
- Add **attention mechanisms** for better feature learning.
- Explore **curriculum learning** (graduated task difficulty).
- Implement **inverse models** and **curiosity-driven exploration**.
- Evaluate on other Atari environments.
- Multi-agent bowling with competitive/cooperative objectives.

---

## 📚 References

- Schulman et al. (2017): *Proximal Policy Optimization Algorithms* ([arXiv](https://arxiv.org/abs/1707.06347))
- Williams (1992): *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*
- Sutton & Barto (2018): *Reinforcement Learning: An Introduction* (2nd ed.)
- Gymnasium Documentation: https://gymnasium.farama.org/

---
