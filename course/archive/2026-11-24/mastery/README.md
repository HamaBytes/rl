# RL Mastery Course — roadmap

Built from your class material (Chapitre 5, B. Rym: the two pptx decks and the FrozenLake docx), plus what class doesn't cover and you'll need for robotics.

**How to use it:** you write all the code. Each module gives the idea, the formulas, skeletons with `___` blanks, exercises, and a self-check. Short answers sit at the bottom of each module. Read the answer only after you've tried.

**Language:** English text, with the French terms your exam uses in *italics* or in brackets.

**Your existing courses (reused, not repeated):**
- [00-gymnasium-basics.md](../00-gymnasium-basics.md): the Gymnasium API, spaces, wrappers
- [01-q-learning.md](../01-q-learning.md): Q-learning intuition, ε-greedy, the update rule
- [02-q-learning-deep.md](../02-q-learning-deep.md): MDP, Bellman derivations, MC vs TD, convergence, discretization, tile coding

Those three were written for your MountainCar graded project. This course stays on class examples (FrozenLake, Gridworld, CliffWalking, CartPole) so the two don't mix.

---

## Track A — Class and exam (finish before exams, 7-16 Jan)

| # | Module | Class slides | Time (GUESS) |
|---|---|---|---|
| A1 | [MDP, return, Bellman](A1-mdp-bellman.md) | 1-21 | 2 h |
| A2 | [Programmation dynamique](A2-dynamic-programming.md) | 22-31 | 3 h |
| A3 | [Monte Carlo](A3-monte-carlo.md) | 32-43 | 2.5 h |
| A4 | [TD: TD(0), SARSA, Q-learning, Expected SARSA](A4-td-sarsa-qlearning.md) | 44-57 + docx | 3 h |
| A5 | [Evaluating an agent + exam question bank](A5-evaluation-and-exam.md) | docx part 3 | 2.5 h |

About 13 hours in total. A suggested fit with your free slots (Wednesday afternoons, weekends, one rest day kept):

- Sat 29 Nov or Sun 30 Nov: A1
- Wed 3 Dec: A2
- Wed 10 Dec: A3
- Weekend 13-14 Dec: A4
- Wed 17 Dec: A5
- Late December: redo the A5 question bank closed book once, then once more in the week of 4 Jan.

This is a suggestion. Your Command Center workbook is the master plan, so put these slots there if you keep them.

## Track B — Beyond class, toward robotics (after 16 Jan)

| # | Module | Why it matters for you |
|---|---|---|
| B1 | [Beyond tables: n-step, function approximation](B1-beyond-tables.md) | The bridge between class and deep RL |
| B2 | [Deep Q-Networks (DQN)](B2-dqn.md) | First neural agent, the base of most of deep RL |
| B3 | [Policy gradients and actor-critic](B3-policy-gradients.md) | Needed for continuous actions (motors) |
| B4 | [PPO and SAC for continuous control](B4-ppo-sac-continuous-control.md) | The two workhorses of robot learning |
| B5 | [RL for real robots: sim-to-real](B5-robotics-sim-to-real.md) | Your humanoid-factory direction |
| B6 | [Reading list and research path](B6-reading-and-research.md) | Books, courses, papers, project ideas |

Track B has no deadline. Rough size: 6 to 10 hours per module. One module every 2 to 3 weeks alongside S2 is realistic. Doing all of it in one month is not.

## Tools

- Track A: `numpy`, `gymnasium`, `matplotlib`. Nothing else.
- Track B: add `torch`, then `stable-baselines3` and `gymnasium[mujoco]` in B4. Your RTX 2060 desktop helps from B2 on, but CPU is enough until B4.

## The one-line map of RL

```
Know the model (P, R)?  yes → Programmation dynamique (A2)
                        no  → learn from experience:
                               wait for the end of the episode → Monte Carlo (A3)
                               update every step (bootstrap)   → TD: SARSA / Q-learning (A4)
Too many states for a table?  → function approximation (B1) → DQN (B2)
Continuous actions (motors)?  → policy gradients (B3) → PPO / SAC (B4) → real robots (B5)
```
