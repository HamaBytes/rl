# B6 — Reading list and research path

Ordered by when to use it. All of these are well-known, but check links and editions yourself: I'm listing them from memory, not from a fresh search.

## Core references

| Resource | Use it for | When |
|---|---|---|
| Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (free PDF on the authors' site) | The textbook. Ch. 3-6 = Track A, ch. 7, 9-13 = B1-B3 | Now (ch. 3-6) |
| David Silver's UCL RL course (lectures on YouTube) | Clear lectures, matches Sutton & Barto | Track A revision |
| OpenAI Spinning Up | Deep RL intro, PPO/SAC explained with code | B3-B4 |
| Hugging Face Deep RL Course (free) | Hands-on notebooks, DQN to PPO | B2-B4 |
| CleanRL (GitHub) | Single-file reference implementations | B2-B4 |
| Stable-Baselines3 docs | Using solid implementations | B4-B5 |

## Papers (in reading order)

1. Mnih et al., 2015, *Human-level control through deep reinforcement learning* (DQN), Nature.
2. Van Hasselt et al., 2016, *Deep RL with Double Q-learning*.
3. Schulman et al., 2017, *Proximal Policy Optimization Algorithms*.
4. Haarnoja et al., 2018, *Soft Actor-Critic*.
5. Tobin et al., 2017, *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*.
6. Hwangbo et al., 2019, *Learning agile and dynamic motor skills for legged robots*, Science Robotics.
7. Rudin et al., 2021, *Learning to Walk in Minutes Using Massively Parallel Deep RL*.
8. Henderson et al., 2018, *Deep Reinforcement Learning that Matters* (why seeds and evaluation matter: read this one alongside A5).

How to read a paper in 45 minutes: abstract → figures → conclusion → method only if you'll implement it. Write 5 lines: problem, idea, result, limit, what you'd try next.

## Research angles that fit you (OPINION)

- **Sim-to-real on low-cost embedded hardware** (MCU-run policies, small balancing robots). Your IoT/embedded profile is rare among RL students.
- **Robustness and evaluation** (seeds, randomization, reproducibility): cheap to run, publishable as careful empirical studies.
- **RL and security of connected things** (your other interest): adversarial attacks on RL policies, or RL for intrusion response. Keep it as a later option, not a parallel track.

## How this connects to your other work

- The MountainCar graded paper uses Track A material (tabular Q-learning, discretization). Track A will make that paper stronger, but grading stays separate.
- Kayoum (robotics/embedded PhD) asked for **one precise topic** before a paper discussion. A domain-randomization study on a small balancing robot (B5 E2 + project step 3) is the kind of precise topic that would fit. That's a suggestion for after 16 Jan, not now.
