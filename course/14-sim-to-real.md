# 14 — RL for real robots: sim-to-real

> **Goal:** know why real robots are hard for RL, the standard toolbox for moving a policy from simulation to hardware, and a project path that uses your embedded skills.
> **You need:** 13. **Time:** 6-10 h (more engineering judgment, less code). **Class:** not in class (after 16 Jan). Closest to your humanoid-factory goal.
> Tool names and versions here move fast: check current docs before installing anything.

---

## 1. Concepts

### 1.1 Why robots are hard for RL

| Problem | What it means |
|---|---|
| Samples are expensive | A real robot runs in real time, wears out, needs resets |
| Exploration is dangerous | Random torques break hardware or hurt people |
| Reality gap | Simulated friction, masses, motor delays and sensors never match reality |
| Partial observability | Sensors are noisy and miss things: the Markov property (01) breaks |
| Reward design | Wrong rewards produce strange "solutions" (*reward hacking*) |

The standard answer: **train in simulation, transfer to the real robot.**

### 1.2 Sim-to-real toolbox

1. **Domain randomization** (Tobin et al., 2017): randomize masses, friction, motor strength, latency, sensor noise and visuals during training. The policy must work across all of them, so reality looks like "one more variation".
2. **System identification:** measure the real robot (motor constants, delays) and set the simulator to match.
3. **Actuator models:** learn a network mapping commands to real motor torques and put it in the sim (Hwangbo et al., *Science Robotics* 2019, ANYmal).
4. **Teacher-student (privileged learning):** a teacher trained with perfect sim information (terrain, friction) is distilled into a student that sees only real sensors.
5. **Massive parallel simulation:** thousands of robots on one GPU (Rudin et al., 2021).
6. **Action smoothing and limits:** low-pass filters, action-rate penalties, torque limits; real motors don't like jerky commands.
7. **Curriculum:** start easy (flat ground), increase difficulty as the policy improves.

### 1.3 Beyond pure RL (know the names)

- **Imitation learning:** behaviour cloning from demonstrations, then RL fine-tuning.
- **Model-based RL:** learn the dynamics, plan or train inside the model (Dreamer family, MBPO). Very sample-efficient.
- **Offline RL:** learn only from a fixed dataset.
- **Safe RL:** constrained MDPs, safety layers.
- **Robot foundation models / vision-language-action models:** large pretrained policies fine-tuned per robot; an active research area in 2025-2026.

### 1.4 Simulators (GUESS on which fits you best; verify current status)

| Tool | Notes |
|---|---|
| MuJoCo (+ Gymnasium) | Free, accurate contacts, used in 13. Start here |
| MuJoCo Playground / MJX | GPU-parallel MuJoCo from Google DeepMind |
| Isaac Lab (NVIDIA) | GPU-parallel, strong for legged robots and humanoids; needs a good NVIDIA GPU (your 2060 may limit it) |
| PyBullet | Older, simple, fine for small projects |
| Gazebo + ROS 2 | Standard robot software stack, less for RL training |

### 1.5 A project path that fits your IoT and embedded skills

1. **Sim only:** Hopper or Walker2d with SB3 PPO, then your own PPO (13).
2. **Domain randomization experiment:** Pendulum or cart-pole with random mass and friction, tested on unseen parameters. Robustness vs no randomization is a small paper-style result on its own.
3. **Real hardware, small:** a reaction-wheel or self-balancing robot, or a Furuta pendulum, on an STM32/ESP32. Model it in MuJoCo, train in sim, deploy (a small MLP runs fine on a microcontroller).
4. **Measure the reality gap:** sim vs real trajectories; add randomization or system ID until transfer works.
5. **Bigger:** a small quadruped kit or low-cost arm, then humanoid simulation (`Humanoid-v5`, Isaac Lab).

Step 3 is where your embedded background becomes an advantage most RL students don't have.

---

## 2. Worked example: a reward for a self-balancing robot

| Term | Sign | Purpose | How it could be hacked |
|---|---|---|---|
| alive bonus per step | + | stay upright | robot stays upright but never moves |
| tilt angle² | − | stay vertical | |
| wheel velocity² | − | don't drift away | |
| (action − previous action)² | − | smooth motor commands | too large a weight → robot barely reacts |
| falling (tilt beyond a limit) | − large, and `terminated` | hard failure | |

Rule: every term must say *what you want*, not *how to do it*, and you list the hack before training, not after.

---

## 3. Code: a domain-randomization wrapper for Pendulum

```python
import numpy as np
import gymnasium as gym

class RandomizePendulum(gym.Wrapper):
    """Each reset draws a new mass and max torque around the nominal values."""
    def __init__(self, env, mass_range=0.3, torque_range=0.2, seed=0):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)
        self.m0 = env.unwrapped.m
        self.t0 = env.unwrapped.max_torque
        self.mass_range, self.torque_range = mass_range, torque_range

    def reset(self, **kwargs):
        u = self.env.unwrapped
        u.m = self.m0 * (1 + self.rng.uniform(-___, ___))
        u.max_torque = self.t0 * (1 + self.rng.uniform(-___, ___))
        return self.env.reset(**kwargs)

env = RandomizePendulum(gym.make("Pendulum-v1"))
```

Check the attribute names on your version first (`vars(gym.make("Pendulum-v1").unwrapped)`) (on gymnasium 1.4.0 they are `m`, `l`, `g`, `max_torque`). Note that changing `max_torque` may not change the action space bounds; decide whether that matters for your test.

---

## 4. Exercises

**E1.** Read Tobin et al. 2017 and write a half-page summary: problem, method, result, limits.

**E2.** Pendulum with randomized mass (±30%) and torque limit, SAC, vs plain SAC. Test both on 10 unseen parameter settings. Table: mean return ± std.

**E3.** Write the reward table (like section 2) for a 2-joint arm that must reach a target point.

**E4.** Export a trained SB3 policy's MLP weights to a C array and run inference in plain C on your laptop: the first step toward running it on an MCU.

---

## 5. Self-check (closed book)

1. Name four sources of the reality gap.
2. Explain domain randomization in two sentences.
3. Why train a teacher with privileged information?
4. When would you choose SAC over PPO for a robot?

---

## Answers

1. Friction, masses and inertias, motor dynamics and delays, sensor noise, contact models, visual appearance.
2. Train across many randomized simulator parameters so the policy can't rely on any single setting. Reality then behaves like one more sample from that distribution.
3. Learning is far easier with perfect information; the student then learns to imitate it from realistic sensors only.
4. When samples are expensive (training on the real robot or in a slow simulator), because SAC reuses data from its replay buffer.

**Next:** [15 — Reading list and research path](15-reading-and-research.md)
