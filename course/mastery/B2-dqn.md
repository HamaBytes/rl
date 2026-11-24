# B2 — Deep Q-Networks (DQN)

Not in class (the docx ends with "the natural next step is DQN"). Paper: Mnih et al., *Human-level control through deep reinforcement learning*, Nature 2015.

Prerequisites: B1, and basic PyTorch (tensors, `nn.Module`, `optimizer.step()`). If PyTorch is new, do the official "Learn the Basics" tutorial first (about 2 h).

## 1. The idea

Q-learning where the table is a neural network Q(s, ·; θ) that outputs one value per action.

Loss on a batch of transitions (s, a, r, s', done):

y = r + γ (1 − done) max_a' Q(s', a'; θ⁻)
L(θ) = mean( Huber( y − Q(s, a; θ) ) )

## 2. The two tricks that make it work (against the deadly triad)

1. **Experience replay** (*mémoire de rejeu*): store transitions in a buffer, train on random mini-batches. Breaks the correlation between consecutive samples and reuses data.
2. **Target network** θ⁻: a frozen copy of θ used in y, updated every C steps (or softly: θ⁻ ← τθ + (1−τ)θ⁻). Stops the target from moving with every gradient step.

## 3. Improvements to know (names + one line each)

| Variant | Fix |
|---|---|
| Double DQN | max overestimates. Choose a* with the online net, evaluate it with the target net: y = r + γ Q(s', argmax_a Q(s',a;θ); θ⁻) |
| Dueling DQN | Split into V(s) + A(s,a) heads; learns state values faster |
| Prioritized replay | Sample transitions with large TD error more often |
| n-step returns | B1's n-step target inside DQN |
| Rainbow | All of the above combined (Hessel et al., 2018) |

**Limit:** the max over actions requires a **discrete** action space. Motor torques are continuous → B3/B4.

## 4. Skeleton (CartPole-v1, PyTorch)

```python
import random, collections
import numpy as np, torch, torch.nn as nn, gymnasium as gym

env = gym.make('CartPole-v1')
obs_dim, nA = env.observation_space.shape[0], env.action_space.n

class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, ___))
    def forward(self, x):
        return self.net(x)

q, q_target = QNet(), QNet()
q_target.load_state_dict(q.state_dict())
opt = torch.optim.Adam(q.parameters(), lr=1e-3)
buffer = collections.deque(maxlen=50_000)

def act(obs, eps):
    if random.random() < eps:
        return env.action_space.sample()
    with torch.no_grad():
        return int(q(torch.as_tensor(obs, dtype=torch.float32)).argmax())

def train_step(batch_size=64, gamma=0.99):
    batch = random.sample(buffer, batch_size)
    s, a, r, s2, done = map(np.array, zip(*batch))
    s, s2 = torch.as_tensor(s, dtype=torch.float32), torch.as_tensor(s2, dtype=torch.float32)
    a = torch.as_tensor(a).long()
    r, done = torch.as_tensor(r, dtype=torch.float32), torch.as_tensor(done, dtype=torch.float32)
    q_sa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        y = r + gamma * (1 - ___) * q_target(s2).max(1).values
    loss = nn.functional.smooth_l1_loss(q_sa, ___)
    opt.zero_grad(); loss.backward(); opt.step()

# main loop sketch:
# obs, _ = env.reset(seed=0); eps = 1.0
# for step in range(100_000):
#     a = act(obs, eps)
#     obs2, r, terminated, truncated, _ = env.step(a)
#     buffer.append((obs, a, r, obs2, float(___)))     # terminated only, NOT truncated
#     obs = obs2 if not (terminated or truncated) else env.reset()[0]
#     if len(buffer) > 1_000: train_step()
#     if step % 1_000 == 0: q_target.load_state_dict(q.state_dict())
#     eps = max(0.05, 1.0 - step / 20_000)
```

## 5. Exercises

**E1.** Get CartPole to an average episode length ≥ 475 over 100 greedy evaluation episodes (that's "solved" for v1). Log episode length per episode.

**E2. Ablations** (3 seeds each): no replay (train on the last transition only), no target network, both. Plot. This shows *why* the tricks exist better than any text.

**E3.** Implement Double DQN (two lines change). Log the average predicted max Q on a fixed set of states. Does it go down vs plain DQN?

**E4.** Compare your agent with Stable-Baselines3's `DQN("MlpPolicy", env)`. Only *after* your own works.

**E5 (stretch).** Read CleanRL's `dqn.py` (single file) and list 3 differences from yours.

## 6. Self-check

1. Why replay? Why a target network?
2. Why use `terminated` and not `truncated` in the `done` mask?
3. Why can't DQN control a motor torque directly?
4. What does Double DQN fix?

---

### Answers

1. Replay breaks correlation and reuses data; the target network keeps y stable for many steps.
2. Same as the docx: truncation is a time limit, the state still has a future.
3. It needs max over actions; with continuous actions that max is itself an optimization problem.
4. Overestimation bias from taking the max of noisy estimates.
