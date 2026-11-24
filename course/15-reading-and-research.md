# 15 — Reading list and research path

> **Goal:** know what to read, in what order, and which research directions fit you.
> **You need:** nothing; use it alongside every module. This is a reference page: no code, exercises or answers.
> All of these are well-known, but they're listed from memory, not from a fresh search: check links and editions yourself.

---

## 1. Core references

| Resource | Use it for | With module |
|---|---|---|
| Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (free PDF on incompleteideas.net) | The textbook. Ch. 3-6 = Part 1; ch. 9-10 = 08, 10; ch. 7, 12 = 10; ch. 13 = 12 | 01-12 |
| David Silver's UCL RL course (lectures on YouTube) | Clear lectures that follow Sutton & Barto | 01-05 revision, 10 |
| OpenAI Spinning Up | Deep RL introduction, PPO/SAC explained with code | 12-13 |
| Hugging Face Deep RL Course (free) | Hands-on notebooks, DQN to PPO | 11-13 |
| CleanRL (GitHub) | Single-file reference implementations | 11-13 |
| Stable-Baselines3 docs | Using solid implementations | 13-14 |

## 2. Papers, in reading order

1. Mnih et al., 2015, *Human-level control through deep reinforcement learning* (DQN), Nature. → 11
2. Van Hasselt et al., 2016, *Deep Reinforcement Learning with Double Q-learning*. → 11
3. Schulman et al., 2017, *Proximal Policy Optimization Algorithms*. → 13
4. Haarnoja et al., 2018, *Soft Actor-Critic*. → 13
5. Tobin et al., 2017, *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*. → 14
6. Hwangbo et al., 2019, *Learning agile and dynamic motor skills for legged robots*, Science Robotics. → 14
7. Rudin et al., 2021, *Learning to Walk in Minutes Using Massively Parallel Deep RL*. → 14
8. Henderson et al., 2018, *Deep Reinforcement Learning that Matters* (why seeds and evaluation matter). → read alongside 06

**How to read a paper in 45 minutes:** abstract → figures → conclusion → method only if you'll implement it. Then write 5 lines: problem, idea, result, limit, what you'd try next.

## 3. Research angles that fit you (OPINION)

- **Sim-to-real on low-cost embedded hardware** (MCU-run policies, small balancing robots). Your IoT/embedded profile is rare among RL students.
- **Robustness and evaluation** (seeds, randomization, reproducibility): cheap to run, publishable as careful empirical studies.
- **RL and security of connected things** (your other interest): adversarial attacks on RL policies, or RL for intrusion response. Keep it as a later option, not a parallel track.

## 4. How this connects to your other work

- The MountainCar graded paper (09) uses Part 1 material. Part 1 will make that paper stronger, but grading stays separate.
- Kayoum (robotics/embedded PhD) asked for **one precise topic** before a paper discussion. A domain-randomization study on a small balancing robot (14: E2 and project step 3) is the kind of precise topic that would fit. A suggestion for after 16 Jan, not now.

**Back to:** [the index](README.md)
