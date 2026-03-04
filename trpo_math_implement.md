### TRPO Mathematical Implementation Details

#### 1. Network Architectures

**Actor (Policy Network)**  
<!-- TODO: which version to use? -->
3‑layer MLP (2 hidden layers, 128 units each, tanh activations) producing action logits:

$$h^{(1)} = \tanh(s_t w_1 + b_1)$$
$$h^{(2)} = \tanh(h^{(1)} w_2 + b_2)$$
$$z = h^{(2)} w_3 + b_3$$
$$A_t \sim \pi^\theta(\bullet|S_t) = \text{Categorical}(\text{softmax}(z))$$

**Value function estimation**
1) **Baseline (Non‑TD)**  
Discounted empirical returns (no neural net):

$$g_t = r_t + \gamma(1-d_t)g_{t+1}$$

+ Advantage uses mean return from previous iteration $b_{\text{prev}}$:

$$\hat{A}_t = g_t - b_{\text{prev}}$$

2) **Value Network**  
Linear value function:

$$v^w(s_t) = s_t w_v + b_v$$

+ Trained by MSE between $v^w(s_t)$ and empirical returns.

+  Advantage Estimation (GAE)

$$\delta_t = r_t + \gamma v^w(s_{t+1}) - v^w(s_t)$$
$$\hat{A}_t = \sum_{l=0}^{N_T} (\gamma \lambda)^l \delta_{t+l}$$
$$\hat{A}_t \leftarrow \frac{\hat{A}_t - \mu(\hat{A})}{\sigma(\hat{A}) + \epsilon}$$

#### 2. Surrogate Objective

$$\mathcal{L}(\theta) = \mathbb{E}_{S_t \sim \rho^{\theta_{\text{old}}}} \left[ \mathbb{E}_{A_t \sim \pi^\theta(\bullet|S_t)} \left[ A^{\pi_{\theta_{\text{old}}}}(S_t, A_t) \right] \right]$$

Using importance sampling:

$$\mathcal{L}(\theta) = \mathbb{E}_{S_t \sim \rho^{\theta_{\text{old}}}, A_t \sim \pi^{\theta_{\text{old}}}} \left[ \frac{\pi^\theta(A_t|S_t)}{\pi^{\theta_{\text{old}}}(A_t|S_t)} A^{\pi_{\theta_{\text{old}}}}(S_t, A_t) \right]$$

Constrained optimization:

$$\max_{\theta} \hat{\mathcal{L}}(\theta) \quad \text{s.t.} \quad \bar{D}_{\text{KL}}(\theta_{\text{old}} \parallel \theta) \le \delta$$

where $\bar{D}_{\text{KL}} = \mathbb{E}_t \big[ D_{\text{KL}}(\pi^{\theta_{\text{old}}}(\bullet|S_t) \parallel \pi^\theta(\bullet|S_t)) \big]$.

From lecture NPG, the KL divergence for small $\Delta\theta = \theta - \theta_{\text{old}}$ is approximated quadratically using the Fisher Information Matrix $F$:

$$\bar{D}_{\text{KL}}(\theta_{\text{old}} \parallel \theta) \approx \frac{1}{2} \Delta\theta^T F \Delta\theta.$$

Maximizing the linearized surrogate $\hat{\mathcal{L}}(\theta)$ subject to this constraint yields the natural gradient update:

$$\Delta\theta = \sqrt{\frac{2\delta}{g^T F^{-1} g}} \, F^{-1} g.$$

#### 3. Solving the Constrained Optimization

**Fisher Information Matrix** $F$ is asymptotically the Hessian of $\bar{D}_{\text{KL}}$.  
Natural gradient direction $x \approx F^{-1}g$, where $g = \nabla_\theta \hat{\mathcal{L}}(\theta)\big|_{\theta_{\text{old}}}$.

**Hessian‑Vector Product** (without full matrix):

$$F v = \nabla_\theta \big( (\nabla_\theta \bar{D}_{\text{KL}})^T v \big) + \text{damping} \cdot v$$

Implemented via PyTorch autograd:
1. $u = \nabla_\theta \bar{D}_{\text{KL}}$
2. $y = u^T v$
3. $F v = \nabla_\theta y + \text{damping} \cdot v$

**Update Steps**:
1. Compute $g$.
2. Solve $F x = g$ via Conjugate Gradient → $x$.
3. Scale step: $\beta = \sqrt{\frac{2\delta}{x^T F x}}$, $\Delta \theta = \beta x$.

#### 5. Backtracking Line Search

Exponentially shrink step size $\alpha$ starting from 1:

$$\theta = \theta_{\text{old}} + \alpha \Delta \theta$$

Accept if:
- $\bar{D}_{\text{KL}} \le \delta$
- $\hat{\mathcal{L}}(\theta) > \hat{\mathcal{L}}(\theta_{\text{old}})$