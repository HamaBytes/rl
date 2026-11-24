# B4 — PPO and SAC for continuous control

Not in class. These two algorithms train most robot policies you'll read about. PPO (Schulman et al., 2017) is on-policy; SAC (Haarnoja et al., 2018) is off-policy.

## 1. PPO: Proximal Policy Optimization

Problem with plain actor-critic: one big gradient step can wreck the policy. PPO limits how far the new policy moves from the one that collected the data.

Ratio: r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

Clipped objective (maximize):

L_clip = E[ min( r_t A_t , clip(r_t, 1−ε, 1+ε) A_t ) ],  ε ≈ 0.2

**GAE** (Generalized Advantage Estimation) for A_t:

δ_t = r_t + γV(s_(t+1)) − V(s_t)
A_t = Σ_l (γλ)^l δ_(t+l),  λ ≈ 0.95

λ is the n-step/TD(λ) trade-off from B1: λ=0 → one-step TD advantage, λ=1 → Monte Carlo.

**Loop:** collect N steps with π_old (often many parallel envs) → compute GAE → several epochs of mini-batch updates on L_clip + value loss − entropy bonus → repeat.

Why robotics likes PPO: stable, simple, scales to thousands of parallel simulated robots on a GPU. Most legged-robot locomotion results train with PPO.

## 2. SAC: Soft Actor-Critic

Off-policy (replay buffer like DQN), so much more sample-efficient. Maximizes reward **plus entropy**:

J = E[ Σ_t r_t + α H(π(·|s_t)) ]

The entropy term keeps exploration alive and makes policies robust. α is often tuned automatically.

Key pieces:
- **Twin Q-networks**, take the min (fights overestimation, like Double DQN).
- Target: y = r + γ(1−d) [ min_i Q_targ,i(s', a') − α log π(a'|s') ],  a' ~ π(·|s')
- **Tanh-squashed Gaussian** policy so actions stay in bounds.
- Reparameterization trick: a = tanh(μ + σ·ε), ε ~ N(0,1), so gradients flow through the sample.

**When to use which (rule of thumb):**
- Cheap, massively parallel simulation → PPO.
- Expensive samples (real robot, slow sim) → SAC.

## 3. Tools

```bash
uv pip install "stable-baselines3[extra]" "gymnasium[mujoco]"
```

```python
import gymnasium as gym
from stable_baselines3 import PPO, SAC

env = gym.make('Pendulum-v1')
model = SAC('MlpPolicy', env, verbose=1, seed=0)
model.learn(total_timesteps=___)
model.save('sac_pendulum')
```

MuJoCo environments in Gymnasium 1.x: `HalfCheetah-v5`, `Hopper-v5`, `Walker2d-v5`, `Ant-v5`, `Humanoid-v5`. Run `[k for k in gym.registry if 'v5' in k]` to confirm on your version.

Read one clean implementation end to end: CleanRL's `ppo_continuous_action.py` and `sac_continuous_action.py`. Each is one file, a few hundred lines.

## 4. Exercises

**E1.** SB3 PPO vs SAC on Pendulum-v1, 3 seeds each, same number of environment steps (e.g. 50k). Plot episodic return vs steps. Which is more sample-efficient?

**E2.** Same on `HalfCheetah-v5` (use the RTX 2060 desktop; budget a few hours per run, GUESS). Record a video with `gymnasium.wrappers.RecordVideo` (needs `imageio`/`moviepy` depending on version).

**E3.** Write your own PPO for discrete CartPole from the CleanRL file as a guide, without copying: skeleton below. Then switch to a Gaussian policy for Pendulum.

```python
# per update:
# 1. rollout: for t in range(n_steps): store obs, action, logp_old, reward, done, value
# 2. GAE (backwards):
#    delta = r[t] + gamma * V[t+1] * (1 - done[t]) - V[t]
#    adv[t] = delta + gamma * lam * (1 - done[t]) * adv[t+1]
#    returns = adv + V
# 3. for epoch in range(n_epochs): for minibatch:
#    ratio = torch.exp(new_logp - ___)
#    pg_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * ___).mean()
#    v_loss = ((V_pred - returns) ** 2).mean()
#    loss = pg_loss + 0.5 * v_loss - ent_coef * entropy.mean()
```

**E4.** Hyperparameter sensitivity: PPO with clip ε ∈ {0.1, 0.2, 0.3} on one env, 3 seeds. One table. (Same method as A5.)

## 5. Self-check

1. What does the PPO clip prevent?
2. What does λ in GAE trade off?
3. Why does SAC use two Q-networks?
4. On-policy or off-policy: PPO? SAC? Consequence for sample efficiency?

---

### Answers

1. Policy updates that move too far from the data-collecting policy, which make the advantage estimates invalid and collapse performance.
2. Bias vs variance in the advantage (TD vs Monte Carlo).
3. Taking the min of two estimates reduces overestimation.
4. PPO on-policy (throws data away after each update, needs more samples); SAC off-policy (replay buffer, fewer samples).
