# 03 — Monte Carlo (*méthodes Monte Carlo*)

> **Goal:** learn values and a policy from complete episodes, with no model.
> **You need:** 01, 02. **Time:** about 2.5 h. **Class:** slides 32-43.
> **Reading:** Sutton & Barto ch. 5 (5.1-5.4); David Silver, lectures 4-5.

---

## 1. Concepts

### 1.1 Why model-free (*sans modèle*)

In real problems (robots, games, finance) you don't know P(s'|s,a). You only see experience: (s_t, a_t, r_(t+1), s_(t+1)). Two families learn from it:

- **Monte Carlo (this module):** wait for the end of the episode, then average the real returns.
- **TD (04, 05):** update after every step, using the current estimate of the next state.

### 1.2 MC prediction: estimate V^π or Q^π for a fixed π

V^π(s) ≈ average of all returns G_t observed after visiting s.

- **First-visit:** only the first visit to s in each episode counts. Unbiased.
- **Every-visit:** every visit counts. More data, slightly different statistics.

**Incremental mean** (no need to store all returns):

```
V(s) ← V(s) + (1/N(s)) [ G − V(s) ]
```

Replace 1/N by a constant α and it tracks problems that change over time. Notice the shape: *old + step × (target − old)*. TD has exactly the same shape; only the target changes. That's the bridge to 04.

### 1.3 MC control: find π*

GPI again (02, section 1.6): estimate Q^π by averaging returns, then make π greedy in Q.

You must keep exploring, or some (s,a) pairs never get a return. Two options:

- **Exploring starts** (*départs exploratoires*, slide 42): each episode starts from a random (s,a). Fine in theory, impractical for real systems (you can't teleport a robot).
- **ε-greedy / ε-soft policies** (standard, S&B ch. 5): behave ε-greedy, so every action has probability at least ε/|A|. Details in 04, section 1.3.

**Off-policy MC with importance sampling** exists (learn about π while following another policy b) but has high variance. Know the name; it isn't in your class.

### 1.4 Pros and limits (slide 43)

| *Avantages* | *Limites* |
|---|---|
| Simple, based on averaging | Needs episodes that end |
| No model | High variance (a whole episode of randomness in each G) |
| Learns from sampled experience | Slow convergence |
| Unbiased (first-visit) | Useless for never-ending tasks |

---

## 2. Worked example: first-visit vs every-visit

Episode: s0 →(r=1)→ s1 →(r=0)→ s0 →(r=2)→ end, γ = 1.

Compute returns backward: G at t=2 is 2; at t=1, 0 + 2 = 2; at t=0, 1 + 2 = 3.

- First-visit V(s0): only t = 0 counts → **3**.
- Every-visit V(s0): t = 0 and t = 2 → (3 + 2) / 2 = **2.5**.
- V(s1) = 2 either way (one visit).

---

## 3. Code: first-visit MC control on FrozenLake

```python
import numpy as np

def generate_episode(env, Q, epsilon, rng):
    """Returns a list of (state, action, reward)."""
    episode = []
    state, _ = env.reset()
    done = False
    while not done:
        action = ___                       # ε-greedy on Q[state], random tie-break
        next_state, reward, terminated, truncated, _ = env.step(action)
        episode.append((state, action, reward))
        done = terminated or truncated
        state = next_state
    return episode

def mc_control_first_visit(env, n_episodes, gamma=0.99, eps_start=1.0,
                           eps_min=0.05, eps_decay=0.9995, seed=0):
    rng = np.random.default_rng(seed)
    nA = env.action_space.n
    Q = np.zeros((env.observation_space.n, nA))
    N = np.zeros_like(Q)
    epsilon = eps_start
    for ep in range(n_episodes):
        episode = generate_episode(env, Q, epsilon, rng)
        first_index = {}
        for t, (s, a, _) in enumerate(episode):
            first_index.setdefault((s, a), t)
        G = 0.0
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = ___
            if first_index[(s, a)] == ___:          # first visit of (s,a)
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / ___
        epsilon = max(eps_min, epsilon * eps_decay)
    return Q
```

---

## 4. Exercises

**E1 (by hand).** Episode: s0 →(r=0)→ s1 →(r=0)→ s0 →(r=1)→ end, γ = 1. Compute V(s0) by first-visit and by every-visit. Why are they equal here, when they weren't in section 2?

**E2.** Run `mc_control_first_visit` on non-slippery FrozenLake (5,000 episodes), then slippery (50,000). Evaluate the greedy policy over 1,000 episodes. Compare with the DP optimum from 02 E3.

**E3.** Why is MC hopeless on a task where an episode lasts 10,000 steps and the reward only comes at the end? Two lines.

**E4.** (After 04.) Plot learning curves (moving average, window 500) for MC and Q-learning on slippery FrozenLake, same number of episodes, 5 seeds. Which learns faster? Which spreads more between seeds? Link the spread to "variance".

---

## 5. Self-check (closed book)

1. First-visit vs every-visit, one line each.
2. Why does MC need exploration, and what are the two ways to get it?
3. Write the incremental mean update. What do you change to get a TD update?
4. Is MC biased? Is its variance high or low? Why?

---

## Answers

- **E1:** returns at the visits to s0: t=0 → 1, t=2 → 1. First-visit 1, every-visit (1+1)/2 = 1. Equal because both visits happen to have the same return; with the rewards of section 2 they differ.
- **E3:** one update per 10,000 steps, and each G carries the noise of 10,000 random steps: credit assignment is almost impossible.
- **E4 (expected, measure it):** Q-learning usually learns faster on FrozenLake; MC curves usually spread more between seeds.
- **Self-check 2:** without exploration some (s,a) never get a return. Exploring starts, or ε-greedy behaviour.
- **Self-check 3:** V ← V + α(G − V). TD replaces G with r + γV(s').
- **Self-check 4:** first-visit MC is unbiased (G is a real sample of the return) but high variance (it sums the randomness of every step to the end).

**Next:** [04 — Q-learning step by step](04-q-learning.md)
