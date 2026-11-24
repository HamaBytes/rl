# B5 — RL for real robots: sim-to-real

Not in class. This is the part closest to your humanoid-factory goal. Less code, more engineering judgment. Tool names and versions here move fast: check the current docs before you install anything.

## 1. Why robots are hard for RL

| Problem | What it means |
|---|---|
| Samples are expensive | A real robot runs in real time, wears out, and needs resets |
| Exploration is dangerous | Random torques break hardware or hurt people |
| Reality gap | Simulated friction, masses, motor delays and sensors never match reality |
| Partial observability | Sensors are noisy and miss things (the Markov property from A1 breaks) |
| Reward design | Wrong rewards produce strange "solutions" (reward hacking) |

The standard answer: **train in simulation, transfer to the real robot.**

## 2. Sim-to-real toolbox

1. **Domain randomization** (Tobin et al., 2017): randomize masses, friction, motor strength, latency, sensor noise, visuals during training. The policy has to work across all of them, so reality looks like "one more variation".
2. **System identification:** measure the real robot (motor constants, delays) and set the simulator to match.
3. **Actuator models:** learn a network that maps commands to real motor torques and put it in the sim (Hwangbo et al., *Science Robotics* 2019, ANYmal).
4. **Teacher-student (privileged learning):** a teacher policy trained with perfect sim information (terrain, friction) is distilled into a student that sees only real sensors.
5. **Massive parallel simulation:** thousands of robots on one GPU (Rudin et al., *Learning to Walk in Minutes Using Massively Parallel Deep RL*, 2021).
6. **Action smoothing and limits:** low-pass filters, action rate penalties, torque limits; real motors don't like jerky commands.
7. **Curriculum:** start on flat ground, increase difficulty as the policy improves.

## 3. Beyond pure RL (know the names)

- **Imitation learning:** behaviour cloning from demonstrations, then RL fine-tuning.
- **Model-based RL:** learn the dynamics model, plan or train inside it (Dreamer family, MBPO). Very sample-efficient.
- **Offline RL:** learn only from a fixed dataset, no new interaction.
- **Safe RL:** constrained MDPs, safety layers.
- **Robot foundation models / vision-language-action models:** large pretrained policies fine-tuned per robot. An active research area in 2025-2026.

## 4. Simulators (GUESS on which fits you best; verify current status)

| Tool | Notes |
|---|---|
| MuJoCo (+ Gymnasium) | Free, accurate contact physics, what B4 uses. Start here |
| MuJoCo Playground / MJX | GPU-parallel MuJoCo from Google DeepMind |
| Isaac Lab (NVIDIA) | GPU-parallel, strong for legged robots and humanoids; needs a good NVIDIA GPU (your 2060 may be limiting) |
| PyBullet | Older, simple, still fine for small projects |
| Gazebo + ROS 2 | Standard for the robot software stack, less for RL training |

## 5. A project path that fits your skills (IoT and embedded)

Each step reuses the previous one. Dates are up to you; this is post-16-Jan material.

1. **Sim only:** Hopper or Walker2d with SB3 PPO, then with your own PPO (B4).
2. **Domain randomization experiment:** train Pendulum or a cart-pole with random mass and friction, test on "unseen" parameters. Plot robustness vs no randomization. This alone is a small paper-style result.
3. **Real hardware, small:** a reaction-wheel or self-balancing robot, or a Furuta pendulum, controlled by an STM32/ESP32. Model it in MuJoCo, train in sim, deploy the policy (a small MLP runs fine on a microcontroller once exported).
4. **Measure the reality gap:** compare sim vs real trajectories; add randomization or system ID until transfer works.
5. **Bigger:** a small quadruped kit or a low-cost arm, then humanoid simulation (`Humanoid-v5`, Isaac Lab humanoids).

Step 3 is where your embedded background becomes an advantage most RL students don't have.

## 6. Exercises

**E1.** Read Tobin et al. 2017 (domain randomization) and write a half-page summary: problem, method, result, limits.

**E2.** Pendulum with randomized mass (±30%) and torque limit noise, SAC, vs plain SAC. Test both on 10 unseen parameter settings. Table: mean return ± std.

**E3.** Write the reward function for a two-wheeled self-balancing robot: list each term, its sign, and how it could be hacked.

**E4.** Export a trained SB3 policy's MLP weights to a C array and run inference in plain C on your laptop. (First step toward running it on an MCU.)

## 7. Self-check

1. Name four sources of the reality gap.
2. Explain domain randomization in two sentences.
3. Why train a teacher with privileged information?
4. When would you choose SAC over PPO for a robot?

---

### Answers

1. Friction, masses/inertias, motor dynamics and delays, sensor noise, contact models, visual appearance.
2. Train across many randomized simulator parameters so the policy can't rely on any single setting. Reality then behaves like one more sample from that distribution.
3. Learning is far easier with perfect information; the student then learns to imitate it from realistic sensors only.
4. When samples are expensive (training on the real robot or in a slow simulator), because SAC reuses data from its replay buffer.
