# B3 — Policy gradients and actor-critic

Not in class. Instead of learning Q and taking the argmax, learn the policy π_θ(a|s) directly. Works with **continuous actions** and **stochastic policies**, which is why robotics uses it.

## 1. The policy gradient theorem

Objective: J(θ) = E[ G_0 ] under π_θ.

∇_θ J(θ) = E[ Σ_t ∇_θ log π_θ(a_t|s_t) · G_t ]

Intuition: push up the log-probability of actions that were followed by high returns, push down the others.

## 2. REINFORCE (Williams, 1992)

Monte Carlo policy gradient: run an episode, compute G_t for each step, then

loss = − Σ_t log π_θ(a_t|s_t) · (G_t − b(s_t))

**Baseline b(s)**: subtracting it doesn't change the expected gradient but reduces variance a lot. The usual choice is a learned V(s).

Practical trick: normalize the returns in each batch (subtract mean, divide by std).

## 3. Actor-critic

Replace the Monte Carlo G_t with a bootstrapped estimate (TD again):

**Advantage** A(s,a) ≈ r + γV(s') − V(s)  (this is the TD error δ)

- **Actor** π_θ: loss = − log π_θ(a|s) · A (A treated as a constant)
- **Critic** V_φ: loss = (r + γV_φ(s') − V_φ(s))²

A2C = synchronous version with several environments in parallel.

The bias/variance trade-off from A3/B1 again: REINFORCE is unbiased and noisy, actor-critic is biased and steadier.

## 4. Continuous actions: Gaussian policy

The network outputs a mean μ_θ(s) (and a log std, often a free parameter). Sample a ~ N(μ, σ²). log π is the Gaussian log-density. Clip or squash with tanh to respect motor limits.

## 5. Skeleton: REINFORCE with baseline (CartPole, discrete)

```python
import torch, torch.nn as nn, gymnasium as gym, numpy as np

env = gym.make('CartPole-v1')
obs_dim, nA = env.observation_space.shape[0], env.action_space.n
policy = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, nA))
value  = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 1))
opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=3e-3)

def run_episode():
    obs, _ = env.reset()
    logps, rewards, states, done = [], [], [], False
    while not done:
        s = torch.as_tensor(obs, dtype=torch.float32)
        dist = torch.distributions.Categorical(logits=policy(s))
        a = dist.sample()
        obs, r, terminated, truncated, _ = env.step(int(a))
        logps.append(dist.log_prob(___)); rewards.append(r); states.append(s)
        done = terminated or truncated
    return logps, rewards, states

def returns_to_go(rewards, gamma=0.99):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = ___
        out.append(G)
    return torch.tensor(out[::-1], dtype=torch.float32)

# update:
# G = returns_to_go(rewards)
# V = value(torch.stack(states)).squeeze(1)
# adv = (G - V.detach())
# policy_loss = -(torch.stack(logps) * ___).sum()
# value_loss = ((G - V) ** 2).mean()
```

## 6. Exercises

**E1.** REINFORCE without baseline vs with baseline on CartPole, 5 seeds. Plot both. Measure the spread.

**E2.** Turn it into one-step actor-critic (advantage = TD error). Compare speed and stability with E1.

**E3.** Pendulum-v1 (continuous torque in [−2, 2]) with a Gaussian actor-critic. It's hard to make work by hand; getting it partly working is the lesson. Then move to B4.

## 7. Self-check

1. Why does a baseline not bias the gradient?
2. What is the advantage, and how do you estimate it with one step?
3. Two reasons robotics prefers policy-gradient methods over DQN.

---

### Answers

1. E[∇log π(a|s) · b(s)] = b(s) · ∇ Σ_a π(a|s) = b(s) · ∇1 = 0.
2. How much better a is than the average action in s: A = Q(s,a) − V(s) ≈ r + γV(s') − V(s).
3. Continuous actions; stochastic policies with smooth exploration (also: easier to add constraints and priors on actions).
