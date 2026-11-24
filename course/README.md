# Reinforcement Learning: one course, one path

*Rebuilt 24 Nov 2026 from your earlier courses (00-02 and the "mastery" modules). Nothing was deleted: the old versions are in [archive/2026-11-24/](archive/2026-11-24/).*

**What this is:** one numbered path from "what is an environment" to "RL on a real robot". It covers your class (Chapitre 5, B. Rym: the two slide decks and the FrozenLake docx), your graded MountainCar project, and the material you'll need for robotics after the exams.

**How to use it:** you write all the code. Each module gives the ideas, a worked example, a code skeleton with `___` blanks, exercises and a closed-book self-check. Answers are at the very end of each module. Look at them only after you've tried.

**Language:** English text, with the French terms your exam uses in *italics*.

---

## The path

### Part 1 — Foundations and class (finish before exams, 7-16 Jan)

| # | Module | Class slides | Time (GUESS) |
|---|---|---|---|
| 00 | [Gymnasium: the agent-environment loop](00-gymnasium.md) | (tools) | 1 h |
| 01 | [MDP, return, Bellman equations](01-mdp-bellman.md) | 1-21 | 2.5 h |
| 02 | [Dynamic programming](02-dynamic-programming.md) | 22-31 | 3 h |
| 03 | [Monte Carlo](03-monte-carlo.md) | 32-43 | 2.5 h |
| 04 | [Q-learning step by step](04-q-learning.md) | 44-57 + docx | 2.5 h |
| 05 | [The TD family: TD(0), SARSA, Expected SARSA, convergence](05-td-control.md) | 44-57 | 3 h |
| 06 | [Evaluating an agent](06-evaluation.md) | docx part 3 | 2.5 h |
| 07 | [Exam question bank](07-exam-question-bank.md) | all | 1.5 h per pass |

### Part 2 — Your graded MountainCar project (due ~28 Dec)

| # | Module | Time (GUESS) |
|---|---|---|
| 08 | [Continuous states: discretization and tile coding](08-discretization-tile-coding.md) | 2.5 h |
| 09 | [MountainCar project guide: from the paper to your 3 hypotheses](09-mountaincar-project.md) | 2 h + the project itself |

Part 2 builds on 00, 01 and 04. You can do it in parallel with Part 1 as soon as 04 is done.

### Part 3 — Beyond class, toward robotics (after 16 Jan, no deadline)

| # | Module | Why it matters for you |
|---|---|---|
| 10 | [Beyond tables: n-step, traces, function approximation](10-beyond-tables.md) | The bridge from class to deep RL |
| 11 | [Deep Q-Networks (DQN)](11-dqn.md) | First neural agent, the base of most deep RL |
| 12 | [Policy gradients and actor-critic](12-policy-gradients.md) | Needed for continuous actions (motors) |
| 13 | [PPO and SAC for continuous control](13-ppo-sac.md) | The two workhorses of robot learning |
| 14 | [RL for real robots: sim-to-real](14-sim-to-real.md) | Your humanoid-factory direction |
| 15 | [Reading list and research path](15-reading-and-research.md) | Books, courses, papers, project ideas |

Rough size for Part 3: 6 to 10 hours per module. One module every 2 to 3 weeks alongside S2 is realistic; all of it in a month is not.

### Other files here

- [rl-lab-part2-prep-2026-11-25.md](rl-lab-part2-prep-2026-11-25.md): prep sheet for the class TP of 25 Nov (kept as it was).
- [archive/](archive/): earlier versions, for reference only.

---

## Suggested schedule (a suggestion: your Command Center workbook is the master plan)

Uses your free slots (Wednesday afternoons, weekends) and keeps one rest day.

| When | Module |
|---|---|
| Sat 29 Nov or Sun 30 Nov | 00 (quick review) + 01 |
| Wed 3 Dec | 02 |
| Wed 10 Dec | 03 |
| Weekend 13-14 Dec | 04 + 05 |
| Wed 17 Dec | 06 |
| Alongside, before ~20 Dec | 08 + 09 for the MountainCar project |
| Late December, then week of 4 Jan | 07 closed book, twice |

---

## How every module is laid out

Same order everywhere, so you always know where to look:

1. **Header:** goal, what you need first, time, matching class slides.
2. **Concepts:** the ideas and formulas, numbered sections.
3. **Worked example:** one calculation done fully.
4. **Code:** a skeleton with `___` blanks for you to fill.
5. **Exercises:** E1, E2, ... by hand first, then code.
6. **Self-check:** closed book, answer in `notes.md`.
7. **Answers:** short answers to the exercises and self-check.
8. **Next:** the module that follows.

Modules 07 (question bank) and 15 (reading list) are reference pages and skip some of these sections.

---

## The one-line map of RL

```
Do you know the model (P, R)?
  yes → Dynamic programming (02)
  no  → learn from experience:
          wait for the end of the episode → Monte Carlo (03)
          update every step (bootstrap)   → TD: Q-learning (04), SARSA (05)
Continuous state?            → discretize or tile-code (08)
Too many states for a table? → function approximation (10) → DQN (11)
Continuous actions (motors)? → policy gradients (12) → PPO / SAC (13) → real robots (14)
```

## Tools

- Parts 1 and 2: `numpy`, `gymnasium`, `matplotlib`. Nothing else.
- Part 3: add `torch` (from 11), then `stable-baselines3` and `gymnasium[mujoco]` (13). Your RTX 2060 desktop helps from 11 on; the laptop CPU is enough until 13.

## Notation used everywhere

| Symbol | Meaning |
|---|---|
| s, a, r, s' | state, action, reward, next state |
| π(a\|s) | policy: probability of action a in state s |
| G_t | return from time t |
| V(s), Q(s,a) | state value, action value |
| α, γ, ε | learning rate, discount factor, exploration rate |
| δ | TD error = target − current estimate |
| `terminated` / `truncated` | episode really ended / cut off by a time limit |

## About sources

Section titles of Sutton & Barto (2nd ed., free PDF on incompleteideas.net) and David Silver's UCL lectures were checked for the old courses. Anything marked *(unverified)* or *(check)* was written from memory: check it in the source before you quote it in a report.
