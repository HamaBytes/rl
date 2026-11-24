# A3 — Monte Carlo (*méthodes Monte Carlo*)

Class: slides 32-43. First model-free method: no P, no R, only sampled episodes.

## 1. Why model-free (*sans modèle*)

In real problems (robots, games, finance) you don't know P(s'|s,a). You only see experience tuples (s_t, a_t, r_(t+1), s_(t+1)). Two families learn from them:

- **Monte Carlo:** wait for the end of the episode, then average the real returns.
- **TD (A4):** update after every step, using the current estimate of the next state.

## 2. MC prediction (estimate V^π or Q^π for a fixed π)

V^π(s) ≈ average of all returns G_t observed after visiting s.

- **First-visit:** only the first time s appears in each episode counts. Unbiased.
- **Every-visit:** every occurrence counts. More data, slightly different statistics.

**Incremental mean** (don't store all returns):

V(s) ← V(s) + (1/N(s)) [ G − V(s) ]

Replace 1/N by a constant α and you get a form that tracks non-stationary problems. Notice it has exactly the shape of the TD update: *old + step × (target − old)*. Only the target changes. That's the key bridge to A4.

## 3. MC control (find π*)

GPI again: estimate Q^π by averaging returns, then make π greedy in Q.

You must keep exploring, otherwise some (s,a) pairs never get a return. Two options:

- **Exploring starts** (*départs exploratoires*, slide 42): every episode starts from a random (s,a). Works in theory, impractical for real systems (you can't teleport a robot).
- **ε-soft / ε-greedy policies** (not detailed in your slides, standard in Sutton & Barto ch. 5): behave ε-greedy, so every action has probability ≥ ε/|A|.

**Off-policy MC with importance sampling** exists (learn about π while following b), but it has high variance. Know the name; it isn't in your class.

## 4. Pros and limits (slide 43)

| Avantages | Limites |
|---|---|
| Simple, based on averaging | Needs episodes that end |
| No model | High variance (a whole episode of randomness in each G) |
| Works from sampled experience | Slow convergence |
| Unbiased (first-visit) | Useless for never-ending tasks |

## 5. Skeletons

```python
from collections import defaultdict
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
        G = 0.0
        visited = set()
        first_index = {}
        for t, (s, a, _) in enumerate(episode):
            first_index.setdefault((s, a), t)
        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = ___
            if first_index[(s, a)] == ___:          # first visit of (s,a)
                N[s, a] += 1
                Q[s, a] += (G - Q[s, a]) / ___
        epsilon = max(eps_min, epsilon * eps_decay)
    return Q
```

## 6. Exercises

**E1 (by hand).** Episode: s0 →(r=0)→ s1 →(r=0)→ s0 →(r=1)→ end, γ = 1. Compute V(s0) by first-visit and by every-visit from this single episode.

**E2.** Run `mc_control_first_visit` on FrozenLake non-slippery (5,000 episodes), then slippery (50,000). Evaluate greedy over 1,000 episodes. Compare with the DP optimum from A2.

**E3.** Plot the learning curve (moving average, window 500) for MC and for Q-learning (docx code) on slippery FrozenLake, same number of episodes, 5 seeds. Which learns faster? Which has a wider spread between seeds? Link the spread to "variance".

**E4.** Why is MC hopeless on a task where the episode lasts 10,000 steps and reward only comes at the end? (Answer in two lines.)

## 7. Self-check

1. First-visit vs every-visit in one line each.
2. Why does MC need exploration, and what are the two ways to get it?
3. Write the incremental mean update. What changes to get a TD update?
4. Is MC biased? Is its variance high or low? Why?

---

### Answers

- E1: returns from s0 visits: t=0 → G=1, t=2 → G=1. First-visit V(s0)=1. Every-visit V(s0)=(1+1)/2=1. (Same here; build your own example with different rewards to see them differ.)
- E3 (expected): Q-learning usually learns faster on FrozenLake; MC curves often spread more between seeds. Measure, don't trust me.
- E4: one update per 10,000 steps, and each G carries the noise of 10,000 random steps. Credit assignment is almost impossible.
- Self-check 3: V ← V + α(G − V). TD replaces G with r + γV(s').
- Self-check 4: first-visit MC is unbiased (G is a real sample of the return) but high variance (it sums the randomness of every step until the end).
