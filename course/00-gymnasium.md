# 00 — Gymnasium: the agent-environment loop

> **Goal:** understand the agent-environment loop well enough to write `random_agent.py` for MountainCar on your own.
> **You need:** Python, numpy. **Time:** about 1 h. **Class:** tools used in every TP.
> Environment names checked against gymnasium 1.4.0, the version in this lab.

---

## 1. Concepts

### 1.1 The loop

Every RL problem is the same loop:

```
        action
  Agent ────────► Environment
    ▲                  │
    └── observation, reward ◄─┘
```

1. The agent looks at the **observation** (what it can see of the state).
2. It picks an **action**.
3. The environment moves forward one timestep and returns a new observation and a **reward**.
4. Repeat until the episode ends.

Gymnasium gives every environment the same interface, so the same loop works for FrozenLake, MountainCar or a simulated robot.

### 1.2 The five functions you need

| Function | What it does | Returns |
|---|---|---|
| `gymnasium.make(id, render_mode=...)` | Creates an environment by name, e.g. `"MountainCar-v0"` | the env |
| `env.reset(seed=...)` | Starts a new episode | `(observation, info)` |
| `env.step(action)` | Advances one timestep | `(observation, reward, terminated, truncated, info)` |
| `env.render()` | Shows the current state | depends on `render_mode` |
| `env.close()` | Closes windows, frees resources | nothing |

The five values from `step()`:

1. **observation:** the new state data.
2. **reward:** feedback for that step.
3. **terminated:** `True` if the episode reached a *natural* end (goal reached, hole, crash).
4. **truncated:** `True` if the episode was *cut off from outside*, usually by a step limit.
5. **info:** a dict of debugging details. Usually ignored.

> **Remember:** `terminated` vs `truncated` is how you tell "reached the flag" apart from "ran out of time". It changes the learning target later (module 04, section 1.4).

**Render modes:** `"human"` opens a window (slow, only to watch). `"rgb_array"` returns pixels (for videos). `None`, the default, has no graphics and is the fastest: **use it for training.**

### 1.3 Spaces: the shape of actions and observations

`env.action_space` says which actions are allowed; `env.observation_space` says what observations look like. Useful methods: `.sample()` (a random valid value) and `.contains(x)`.

| Space | Meaning | Example |
|---|---|---|
| `Discrete(n)` | one integer in {0, ..., n−1} | FrozenLake squares, MountainCar actions |
| `Box(low, high, shape)` | bounded array of real numbers | MountainCar position and velocity, pixels |
| `MultiBinary(n)` | array of on/off switches | |
| `MultiDiscrete([...])` | several discrete choices | |

Composite spaces (`Dict`, `Tuple`, ...) exist; you won't need them for a while.

What your own machine printed for MountainCar:

```
Box([-1.2  -0.07], [0.6  0.07], (2,), float32)   # observation_space: position, velocity
Discrete(3)                                        # action_space: left, nothing, right
```

A Q-table needs *countable* rows. `Discrete` is countable; `Box` is not. Module 08 solves that gap.

### 1.4 Wrappers

A wrapper changes how an environment behaves (observations, rewards, actions) without editing its code. They stack.

- **`TimeLimit`:** sets `truncated=True` after a maximum number of steps.
- **`FlattenObservation`:** turns a multi-dimensional observation into a 1D array.
- **`ClipAction` / `RescaleAction`:** keep continuous actions inside legal bounds.

`env.unwrapped` gives you the original environment under all the layers (module 02 uses it to read FrozenLake's model).

> **Checked on gymnasium 1.4.0:** `gym.make("MountainCar-v0")` comes wrapped in `TimeLimit` with **200 steps**.

### 1.5 Environment families

| Family | Exact IDs in gymnasium 1.4.0 | Type |
|---|---|---|
| Toy Text | `FrozenLake-v1`, `Taxi-v4`, `CliffWalking-v1`, `Blackjack-v1` | small and discrete: tabular-friendly (Part 1) |
| Classic Control | `CartPole-v1`, `MountainCar-v0`, `Pendulum-v1`, `Acrobot-v1` | simple physics (Parts 2-3) |
| Box2D | `LunarLander-v3`, `BipedalWalker-v3`, `CarRacing-v3` | 2D physics, extra install |
| MuJoCo | `Hopper-v5`, `Ant-v5`, `Humanoid-v5` | robot locomotion (module 13), extra install |
| Atari | via the Arcade Learning Environment | games from pixels |

Robotics road: Toy Text → Classic Control → MuJoCo.

Later, not now: **vector environments** (`SyncVectorEnv`, `AsyncVectorEnv`) run several copies at once; libraries like Stable-Baselines3 and CleanRL sit on top of Gymnasium. In Parts 1 and 2 you write everything by hand, on purpose.

---

## 2. Worked example: one CartPole episode

The official example uses CartPole, so you adapt it yourself for MountainCar.

```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset(seed=42)

episode_over = False
total_reward = 0.0
while not episode_over:
    action = env.action_space.sample()          # random valid action
    observation, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    episode_over = terminated or truncated

print(f"Episode finished! Total reward: {total_reward}")
env.close()
```

Line by line: `reset` gives the first observation; each `step` returns the five values; the loop stops on **either** a natural end **or** the time limit.

> **Setup note:** `render_mode="human"` needs pygame. If it complains, run `uv add "gymnasium[classic-control]"` from `rl-lab/`. For training, leave `render_mode` out.

---

## 3. Code: `random_agent.py` for MountainCar

```python
import gymnasium as gym
import numpy as np

env = gym.make("___")                     # no render_mode: fast
n_episodes = 10
returns = []

for ep in range(n_episodes):
    obs, info = env.reset(seed=___)        # a different, recorded seed per episode
    done, total = False, 0.0
    while not done:
        action = ___                       # random valid action
        obs, reward, terminated, truncated, info = env.step(action)
        total += ___
        done = ___ or ___
    returns.append(total)
    print(ep, total, "reached flag" if ___ else "time limit")

print("mean return:", np.mean(returns))
env.close()
```

---

## 4. Exercises

**E1.** Run the CartPole example, then change it to run 10 episodes and print each total.

**E2.** Fill in `random_agent.py`. Run it. What return do you get on every episode? Did the car ever reach the flag?

**E3.** Print `env.observation_space.low`, `.high` and `env.action_space.n` for `FrozenLake-v1` and `MountainCar-v0`. Which one could you store directly in a table?

**E4.** Print `env.spec.max_episode_steps` for MountainCar-v0, FrozenLake-v1 and CartPole-v1.

---

## 5. Self-check (closed book, in `notes.md`)

1. What does `reset()` return? What does `step()` return?
2. Difference between `terminated` and `truncated`? Give a MountainCar example of each.
3. Which `render_mode` do you use for training, and why?
4. MountainCar's action space is `Discrete(3)`. What are the three actions?
5. Its observation space is a `Box` of shape `(2,)`. What are the two numbers?
6. Why can't a Q-table use MountainCar observations directly?
7. MountainCar has a 200-step limit and gives −1 per step. What total reward does an agent get if it never reaches the flag?
8. Why does a seed matter in a scientific report?

---

## Answers

- **E2:** −200 every time; a random agent essentially never reaches the flag (the engine is weaker than gravity, see module 01, section 1.6).
- **E3:** FrozenLake: `Discrete(16)` observations, `Discrete(4)` actions, so a 16×4 table works directly. MountainCar: a `Box`, so it doesn't.
- **E4:** 200 for MountainCar-v0, 100 for FrozenLake-v1 (4×4), 500 for CartPole-v1 (checked on gymnasium 1.4.0).
- **Self-check 1:** `reset()` → `(observation, info)`. `step()` → `(observation, reward, terminated, truncated, info)`.
- **Self-check 2:** terminated = the car reached the flag (position ≥ 0.5). Truncated = the 200-step limit hit first.
- **Self-check 3:** `None`: no drawing, so training runs much faster.
- **Self-check 4:** 0 = push left, 1 = no push, 2 = push right.
- **Self-check 5:** position in [−1.2, 0.6] and velocity in [−0.07, 0.07].
- **Self-check 6:** they are continuous; there are infinitely many possible rows.
- **Self-check 7:** −200.
- **Self-check 8:** RL is random; a seed makes a run reproducible, and several seeds show whether a result is luck.

**Next:** [01 — MDP, return, Bellman equations](01-mdp-bellman.md)
