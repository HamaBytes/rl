# 04 — Q-learning step by step

> **Goal:** understand Q-learning well enough to explain every line of your own implementation before you write it.
> **You need:** 00, 01 (03 helps). **Time:** about 2.5 h. **Class:** slides 44-57 and the FrozenLake docx (steps 1-5).
> **Reading:** Sutton & Barto 6.5; David Silver, lecture 5.

---

## 1. Concepts

### 1.1 The intuition

A random agent never improves because it remembers nothing. Q-learning adds one thing: **a memory of how good each action was in each situation.**

The agent keeps asking:

> "If I'm in state **s** and take action **a**, how much total reward will I collect from now on, if I play well afterwards?"

That number is **Q(s,a)**. If you knew it perfectly, playing well would be easy: pick the action with the highest Q. Q-learning **estimates Q\* from experience**, one step at a time (it's the sampled version of the Bellman optimality equation from 01, section 1.5).

### 1.2 The Q-table

When states and actions are countable, store Q in a table: one **row** per state, one **column** per action.

| state | action 0 (left) | action 1 (none) | action 2 (right) |
|---|---|---|---|
| s₀ | −12.3 | −14.1 | −9.8 |
| s₁ | −7.0 | −8.2 | −10.5 |

In s₀ the best action is **right** (−9.8 is highest). At the start the table holds a guess, usually zeros; learning fixes the guesses.

FrozenLake: 16 states × 4 actions, so `Q = np.zeros((16, 4))`. MountainCar's states are continuous, so it needs a trick first (module 08).

### 1.3 Choosing actions: ε-greedy (*exploration vs exploitation*)

Always picking the best-known action means never trying anything new, so you can get stuck on a bad early guess. Always acting randomly means never using what you learned.

**ε-greedy:**
- with probability **ε**, a **random** action (*exploration*);
- otherwise the action with the **highest Q** (*exploitation*). Break ties at random, or the agent always favours action 0 while Q is all zeros.

ε usually starts high (1.0) and **decays** over episodes (to 0.01-0.05): explore early, exploit later.

### 1.4 The update rule

After every step (s, a) → r, s':

```
Q(s,a)  ←  Q(s,a)  +  α · [ r + γ · max_a' Q(s',a')  −  Q(s,a) ]
                            └──────── target ───────┘
                          └─────────────── TD error δ ───────────┘
```

1. **Target** = reward just received + discounted value of the best action from where I landed.
2. **TD error** δ = target − old guess: how wrong I was.
3. Move the old guess a fraction **α** of the way toward the target.

**If the episode really ended** (`terminated`), there's no future: target = r.
**If it was cut off** (`truncated`), the state still had a future: keep r + γ max Q(s'). Many implementations treat both the same; for your project, find out what the paper does (09).

### 1.5 The three knobs

| Symbol | Name | Controls | Too low | Too high |
|---|---|---|---|---|
| α | learning rate | how fast old guesses change | learns very slowly | unstable, forgets |
| γ | discount factor | how much the future counts | short-sighted | (close to 1 is normal for MountainCar) |
| ε | exploration rate | how often to act randomly | stuck on bad habits | never uses what it learned |

### 1.6 Initialization matters

In MountainCar every reward is −1, so all true Q-values are negative. Starting the table at **0** makes every untried action look *better than it is*, which pushes the agent to try it: **optimistic initialization**, free exploration. In FrozenLake (rewards 0 or 1) zeros are neutral instead.

### 1.7 The full algorithm

```
Q = zeros(n_states, n_actions)
for each episode:
    s = reset()
    repeat until terminated or truncated:
        a = ε-greedy(Q, s)
        s', r, terminated, truncated = step(a)
        target = r if terminated else r + γ · max Q[s']
        Q[s, a] += α · (target − Q[s, a])
        s = s'
    decay ε
    record the episode's total reward
```

### 1.8 How you'll know it works

- **Learning curve:** total reward per episode. Random = flat. A learning agent rises, then plateaus.
- Curves are noisy: plot a **moving average** (e.g. 100 episodes).
- Judge the final agent separately, greedy and without updates (module 06).

---

## 2. Worked examples

**A. MountainCar-style numbers.** α = 0.1, γ = 0.99. Q(s,a) = −5, r = −1, max Q(s',·) = −4.

- target = −1 + 0.99 × (−4) = **−4.96**
- δ = −4.96 − (−5) = **0.04**
- new Q(s,a) = −5 + 0.1 × 0.04 = **−4.996**

Q went up slightly: this action was a bit better than the agent thought.

**B. The class docx example.** Q(s,a) = 0.5, r = 0, max Q(s',·) = 0.8, α = 0.1, γ = 0.9.

- target = 0 + 0.9 × 0.8 = **0.72**; δ = **0.22**; new Q = 0.5 + 0.022 = **0.522**.

---

## 3. Code: tabular Q-learning on FrozenLake

```python
import numpy as np
import gymnasium as gym

def choose_action(Q, s, epsilon, rng):
    if rng.random() < ___:
        return int(rng.integers(Q.shape[1]))
    best = np.flatnonzero(Q[s] == Q[s].___())   # all tied best actions
    return int(rng.choice(___))

def q_learning(env, n_episodes, alpha=0.1, gamma=0.99, eps_start=1.0,
               eps_min=0.05, eps_decay=0.999, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon, returns = eps_start, []
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            a = choose_action(Q, s, epsilon, rng)
            s2, r, terminated, truncated, _ = env.step(a)
            target = r + gamma * ___ * (not ___)
            Q[s, a] += alpha * (___ - Q[s, a])
            s = ___
            done = terminated or truncated
            total += r
        epsilon = max(eps_min, epsilon * ___)
        returns.append(total)
    return Q, returns

env = gym.make("FrozenLake-v1", is_slippery=False)
Q, returns = q_learning(env, 2000)
```

---

## 4. Exercises

**E1.** In the table of section 1.2, which action does a greedy agent pick in s₁? Why are all values negative in MountainCar?

**E2.** With ε = 0.1 and 3 actions, what is the probability the agent picks the greedy action? (A random pick can also land on it.)

**E3.** α = 0.5, Q(s,a) = −10, r = −1, and the car just reached the flag (`terminated=True`). New Q(s,a)?

**E4.** Worked example A with α = 1.0. New Q? What does α = 1 mean in words? And α = 0?

**E5.** Why might the MountainCar agent learn faster with Q initialized to 0 than to −100?

**E6 (code).** Fill in the skeleton. Train on non-slippery FrozenLake, plot the moving average of returns, print the greedy policy as arrows on the 4×4 grid.

**E7 (code).** Same on slippery FrozenLake, 20,000 episodes. Evaluate the greedy policy on 1,000 episodes. How far are you from the DP optimum (02 E3)?

---

## 5. Self-check (closed book)

1. What does Q(s,a) mean, in one sentence?
2. Write the Q-learning update from memory and name each part.
3. What changes in the update when `terminated` is True? When only `truncated` is True?
4. What do α, γ and ε each control?
5. Why is initializing Q to 0 "optimistic" in MountainCar?
6. Why does ε usually decay over time?
7. Why break ties at random in ε-greedy?

---

## Answers

- **E1:** action 0 (−7.0 is highest). Every step costs −1 until the flag, so every return is negative.
- **E2:** 0.9 + 0.1 × 1/3 ≈ 0.933.
- **E3:** target = −1 (no future); new Q = −10 + 0.5 × (−1 + 10) = −5.5.
- **E4:** −5 + 1.0 × 0.04 = −4.96, i.e. the target itself. α = 1: forget the old guess entirely. α = 0: never learn.
- **E5:** with 0, untried actions look better than tried ones (which drop below 0), so the agent keeps trying new actions. With −100, tried actions quickly look better than untried ones, so exploration dies early.
- **Self-check 3:** terminated → target = r. Truncated only → keep r + γ max Q(s').
- **Self-check 5:** true values are all negative, so 0 overestimates every untried action.
- **Self-check 6:** explore while knowledge is poor, exploit once it's good.
- **Self-check 7:** with an all-zero table, `argmax` always returns action 0, which biases exploration.

**Next:** [05 — The TD family](05-td-control.md). Project track: [08 — Discretization and tile coding](08-discretization-tile-coding.md).
