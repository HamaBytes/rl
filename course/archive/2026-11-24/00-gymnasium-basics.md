# Course 00: Gymnasium basics

*From hama's notes, 23 Nov 2026. Environment names checked against gymnasium 1.4.0 (the version in this lab).*

**Goal of this lesson:** understand the agent–environment loop well enough to write `random_agent.py` for MountainCar on your own.

---

## 1. The big idea: the agent–environment loop

Every RL problem is the same loop:

```
        action
  Agent ────────► Environment
    ▲                  │
    └──── observation, reward ◄─┘
```

1. The agent looks at the **observation** (what it can see of the state).
2. It picks an **action**.
3. The environment moves forward one timestep and returns a new observation plus a **reward**.
4. Repeat until the episode ends.

Gymnasium is a library that gives every environment the same interface, so the same loop works for CartPole, MountainCar or a robot simulator.

---

## 2. The 5 functions you need

| Function | What it does | Returns |
|---|---|---|
| `gymnasium.make(id, render_mode=...)` | Creates an environment by its registered name, e.g. `"MountainCar-v0"` | the env |
| `env.reset(seed=..., options=...)` | Starts a new episode | `(observation, info)` |
| `env.step(action)` | Advances one timestep | `(observation, reward, terminated, truncated, info)` |
| `env.render()` | Shows the current state | depends on `render_mode` |
| `env.close()` | Closes windows and frees resources | nothing |

### The 5 values from `step()`

1. **observation**: the new state data the agent sees.
2. **reward**: a number giving feedback for that step.
3. **terminated**: `True` if the episode reached a *natural* end (the task succeeded or failed).
4. **truncated**: `True` if the episode was *cut off from outside*, usually by a step time limit.
5. **info**: a dict of extra debugging details. You can usually ignore it.

> **Key point for MountainCar:** `terminated` vs `truncated` is how you tell *"reached the flag"* apart from *"ran out of time"*. Your mission depends on this.

### Render modes

- `"human"`: opens a window. It's slow, so use it only to *watch*.
- `"rgb_array"`: returns the image as a pixel array, e.g. to make videos.
- `None` (default): no graphics. It's the fastest, so **use it for training.**

---

## 3. The basic loop (official example, CartPole)

This is the example from the Gymnasium docs. It uses **CartPole**, not MountainCar, on purpose: you'll adapt it yourself in the mission.

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

print(f"Episode finished! Total Reward: {total_reward}")
env.close()
```

Read it line by line and make sure you can answer:
- Why is the loop condition `terminated or truncated`, and not just `terminated`?
- This runs **one** episode. What would you change to run 10?
- Where is the seed used, and why does a seed matter for a scientific report?

> **Setup note:** `render_mode="human"` needs pygame. If you get an error about pygame, run `uv add "gymnasium[classic-control]"` from `rl-lab/` (the docs say `pip install`, but this lab uses uv). For training, leave `render_mode` out.

---

## 4. Spaces: the shape of actions and observations

Every environment describes what is valid with:
- `env.action_space`: what actions are allowed
- `env.observation_space`: what observations look like

Useful methods: `.sample()` gives a random valid value, and `.contains(x)` checks whether `x` is valid.

### Basic spaces

| Space | Meaning | Example |
|---|---|---|
| `Discrete(n)` | One integer in `{0, 1, ..., n-1}` | button presses |
| `Box(low, high, shape)` | A bounded array of continuous numbers | sensor values, velocity, pixels |
| `MultiBinary(n)` | An array of on/off switches | |
| `MultiDiscrete([...])` | Several discrete choices, each with its own range | |
| `Text` | A bounded string | |

### Composite spaces
`Dict`, `Tuple` (combine simpler spaces), `Sequence`, `Graph` and `OneOf`. You won't need these for MountainCar.

### What you already saw for MountainCar (from your own machine)

```
Box([-1.2  -0.07], [0.6  0.07], (2,), float32)   # observation_space
Discrete(3)                                        # action_space
```

**Think:** a Q-table needs *countable* rows. `Discrete(3)` is countable. `Box` is continuous. That gap is exactly what the paper's "tabular discretization" solves.

---

## 5. Wrappers

A wrapper changes how an environment behaves (its observations, rewards or actions) **without editing the environment's code**. Wrappers can be stacked.

Common ones:
- **`TimeLimit`**: sets `truncated=True` after a maximum number of steps.
- **`FlattenObservation`**: flattens a multi-dimensional observation into one 1D array.
- **`ClipAction` / `RescaleAction`**: keep continuous actions inside legal bounds, or rescale them.

`wrapped_env.unwrapped` gives you the original environment under all the layers.

> **Checked on gymnasium 1.4.0:** `gym.make("MountainCar-v0")` already comes wrapped in `TimeLimit` with **max_episode_steps = 200**. Keep that number in mind when you see your random agent's total reward.

---

## 6. Built-in environment families

| Family | Examples (exact IDs in gymnasium 1.4.0) | Type |
|---|---|---|
| Classic Control | `CartPole-v1`, `MountainCar-v0`, `Pendulum-v1`, `Acrobot-v1` | simple physics |
| Toy Text | `FrozenLake-v1`, `Taxi-v4`, `CliffWalking-v1`, `Blackjack-v1` | small, already discrete: tabular-friendly |
| Box2D | `LunarLander-v3`, `BipedalWalker-v3`, `CarRacing-v3` | 2D physics with collisions |
| MuJoCo | `Ant-v5`, `HalfCheetah-v5`, `Hopper-v5`, `Humanoid-v5` | robotics / locomotion |
| Atari | via the Arcade Learning Environment | games from pixels |

> **Corrections to the original notes:** it's `Blackjack-v1` (not v2) and `Taxi-v4` (not v3) in this version. Box2D, MuJoCo and Atari need extra installs; Classic Control and Toy Text work out of the box.

**Robotics path:** Classic Control (now), then MuJoCo (`Hopper`, `Humanoid`) later. That's the road from MountainCar toward robots.

---

## 7. Scaling up (for later, not for this project)

- **Vector environments**: run several env copies at once. `SyncVectorEnv` runs them one after another; `AsyncVectorEnv` runs them in parallel processes on several CPU cores.
- **RL libraries** built on Gymnasium: Stable-Baselines3, CleanRL, Tianshou, Ray RLlib. **PettingZoo** is the multi-agent version.

For this project you write tabular Q-learning **by hand**, with no library. That's the point: you have to understand what those libraries hide.

---

## 8. Self-check (answer in `notes.md` without looking)

1. What does `reset()` return? What does `step()` return?
2. Difference between `terminated` and `truncated`? Give a MountainCar example of each.
3. Which `render_mode` do you use for training, and why?
4. MountainCar's action space is `Discrete(3)`. What are the 3 actions?
5. Its observation space is a `Box` of shape `(2,)`. What are the 2 numbers?
6. Why can't a Q-table use MountainCar observations directly?
7. MountainCar has a 200-step limit and gives -1 per step. What total reward does an agent get if it never reaches the flag?

**Next:** Mission 1, "Meet the car" (`random_agent.py`, then `heuristic_agent.py`).
