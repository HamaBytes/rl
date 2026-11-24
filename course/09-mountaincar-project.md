# 09 — MountainCar project guide: from the paper to your 3 hypotheses

> **Goal:** reproduce the paper's Table II, then defend every design choice and turn the theory into three testable hypotheses.
> **You need:** 01 (1.6), 04, 05 (1.6), 06, 08. **Time:** about 2 h of reading and planning, then the project itself. **Due:** around 28 Dec (week 13).
> **Paper:** *Modelling a RL Agent for Mountain Car Using Q-Learning With Tabular Discretization*, IEEE MysuruCon 2022, DOI 10.1109/MysuruCon55714.2022.9972352. Target: Table II (5 configurations, best −108; "solved" threshold −110).

**Professor's rule:** complete the analysis of 3 hypotheses first. Only then add anything else.

---

## 1. Concepts

### 1.1 Mission order

| Mission | File | Done when |
|---|---|---|
| 1. Meet the car | `random_agent.py` (00) | −200 every episode, explained |
| 2. Discretization | `discretize()` (08) | hand-checked tests pass |
| 3. Q-learning | `q_learning.py` | learning curve rises above −200 |
| 4. Reproduce Table II | same file, 5 configs | your numbers vs the paper, side by side |
| 5. Hypotheses H1-H3 | same file, one knob at a time | each accepted or rejected with evidence |
| 6 (optional, after Table II) | H4: tile coding, or LightGBM Fitted Q-Iteration | only if 1-5 are done (~20 Dec) |

Keep everything in **one** `q_learning.py` with the knobs as arguments, so every experiment uses the same code.

### 1.2 Where your setup breaks the convergence conditions (05, section 1.6)

Be honest about this in the report: this table alone is a strong *Discussion / Limitations* paragraph.

| Condition | In your project | Consequence |
|---|---|---|
| 1. all pairs visited | some cells are unreachable (e.g. high speed at the far left), and ε decays toward ~0 | those Q-values never update. Fine if irrelevant, but they show up in a policy heatmap |
| 2. Robbins-Monro | the paper probably uses a constant α *(check)* | no strict convergence; Q keeps fluctuating, curves stay noisy |
| 3. Markov finite MDP | discretized cells are **not** Markov (08) | the guarantee doesn't apply at all: Q-learning on bins is a heuristic that usually works, not a proven method |

### 1.3 Why Q-learning and not SARSA (your professor may ask)

Q-learning is off-policy: it learns the greedy policy you'll evaluate, while exploring. SARSA would learn the value of the exploring policy (05, section 1.4). In MountainCar there's no "cliff", so the safety advantage of SARSA doesn't matter much. That's a reason, but it's also a possible extra comparison, not a core one.

### 1.4 Natural hypotheses from the theory

| Hypothesis | Knob | Theory it comes from |
|---|---|---|
| H1 | number of bins | 08, section 1.3 (sweet spot) |
| H2 | α and/or ε schedule | 04 1.5, 05 1.6 (exploration, step size) |
| H3 | γ | 01 E6 (small γ can't tell fast from slow) |
| H4 (optional) | tile coding vs single grid | 08, section 1.4 |

---

## 2. Worked example: writing one hypothesis properly

> **H1:** "Increasing the number of bins per dimension from 10 to 40 improves the average return over the last 100 training episodes up to some N\*, after which it decreases, within a fixed budget of E episodes."
> **Test:** vary bins only (10, 20, 30, 40), everything else fixed at the paper's best config, 5 seeds each, greedy evaluation on 100 episodes per seed.
> **Evidence that would reject it:** the score keeps increasing up to 40, or the differences are smaller than ~2 standard errors across seeds.

Each part matters: one knob, a fixed budget, a measurable quantity, seeds, and a rejection criterion written **before** you run.

---

## 3. Code: `q_learning.py` outline

```python
import argparse
import numpy as np
import gymnasium as gym
# from discretize import discretize   (your Mission 2 code)

def train(n_bins, alpha, gamma, eps_start, eps_end, eps_decay, n_episodes, seed):
    env = gym.make("MountainCar-v0")
    rng = np.random.default_rng(seed)
    Q = np.zeros((*n_bins, env.action_space.n))      # optimistic in MountainCar (04, 1.6)
    epsilon, returns = eps_start, []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed * 100_000 + ep)
        s = discretize(obs, n_bins)
        done, total = False, 0.0
        while not done:
            a = ___                                   # ε-greedy on Q[s], random tie-break
            obs2, r, terminated, truncated, _ = env.step(a)
            s2 = discretize(obs2, n_bins)
            target = ___                              # your terminal/truncated choice, documented
            Q[s + (a,)] += alpha * (target - Q[s + (a,)])
            s = ___
            done = ___
            total += r
        epsilon = ___                                 # the paper's schedule
        returns.append(total)
    env.close()
    return Q, returns

# main: parse the knobs with argparse, run 5 seeds, save returns with np.save,
# plot the moving average (window 100) with a std band, evaluate greedy (06).
```

`Q[s + (a,)]` works because `s` is a tuple `(i, j)`; `s + (a,)` is `(i, j, a)`.

---

## 4. Exercises (do 1, 2 and 5 before writing `q_learning.py`)

**E1. Extract the spec.** From the paper's method section, fill this table in `notes.md`:

| Config | pos bins | vel bins | α | γ | ε start → end, decay | episodes | Q init | result |
|---|---|---|---|---|---|---|---|---|

Anything the paper doesn't state is a **gap**. List the gaps: they're choices you must make and justify. A professor values that honesty.

**E2. Terminal handling.** Does the paper bootstrap on `truncated`? If it doesn't say, decide, and write one paragraph justifying it with 04 (1.4) and 05 (1.6).

**E3. Predict before you run.** Rank the 5 configs using 01 E6, 05 1.6 and 08 1.3, before you have any code. After you reproduce Table II, compare. A correct prediction shows understanding; a wrong one, explained, makes an even better discussion section.

**E4. Write H1, H2, H3** in the format of section 2, with a rejection criterion each.

**E5. Define "solved".** Is "−108" a single episode, or an average over the last 100? Training or greedy evaluation? Find it in the paper. If you compare different quantities, your reproduction is invalid.

**E6. Learning curve sanity check.** Random gives a flat −200. Yours should start at −200 and, after hundreds to thousands of episodes, rise (toward roughly −150, then higher). Don't expect −108 on the first run.

---

## 5. Self-check (closed book, before your oral)

1. Explain "Q-learning is sample-based value iteration on the Bellman optimality equation" in two sentences.
2. Which convergence conditions does your project violate, and what does that change in practice?
3. Why does γ matter so much in MountainCar?
4. Why is optimistic initialization free exploration here?
5. What is aliasing, and which hypothesis tests it?
6. Why would tile coding be a fair H4 and LightGBM a less direct one?

---

## Answers

- **Self-check 1:** the Bellman optimality equation says Q*(s,a) = r + γ max Q*(s',·). Q-learning never computes the expectation; it takes one sampled (r, s') and moves Q(s,a) a step α toward that sampled right-hand side.
- **Self-check 2:** all three (section 1.2). In practice: no guarantee, Q fluctuates, results vary across seeds, so report mean ± std.
- **Self-check 3:** the flag is ~100+ steps away; with small γ the returns of fast and slow (or failed) episodes become almost identical (01 E6).
- **Self-check 4:** all true values are negative, so zeros make every untried action look better than tried ones.
- **Self-check 5:** different real states sharing one cell; H1 (number of bins).
- **Self-check 6:** tile coding keeps the same algorithm and changes only the state representation, so it isolates one knob. LightGBM Fitted Q-Iteration changes the algorithm itself (batch, supervised regression), so it's a different method, not a variant.

**Next:** Part 3, after the exams: [10 — Beyond tables](10-beyond-tables.md).
