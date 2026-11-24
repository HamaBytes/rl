# 06 — Evaluating an agent (*évaluation d'un agent*)

> **Goal:** turn "my code runs" into "I can prove my agent learned", with numbers that survive a professor's questions.
> **You need:** 04, 05. **Time:** about 2.5 h. **Class:** docx part 3.
> **Reading (later):** Henderson et al., 2018, *Deep Reinforcement Learning that Matters*.

---

## 1. Concepts

### 1.1 Golden rules

1. **Never judge on training scores.** Training uses ε > 0, so scores are lowered on purpose. Evaluate separately: greedy (ε = 0), **no Q update**.
2. **Compare to a baseline** (random policy). A score means nothing alone.
3. **Several seeds** (at least 5). Report mean ± standard deviation.
4. **Change one hyperparameter at a time.**
5. **Enough evaluation episodes.** Slippery FrozenLake: 1,000, because success is a noisy 0/1 per episode.

### 1.2 Metrics

| *Métrique* | What it says |
|---|---|
| *Retour moyen* | Average cumulative reward; should rise then plateau |
| *Taux de succès* | Share of episodes reaching the goal (0/1 rewards) |
| *Longueur d'épisode* | Shorter = more efficient |
| *Vitesse de convergence* | Episodes needed to reach a good level |
| *Variance entre graines* | Stability: a result is only reliable if reproducible |

### 1.3 How many evaluation episodes is enough?

For a success rate p measured over n episodes, the standard error is about **√(p(1−p)/n)**. Two results closer than about two standard errors can't be told apart.

### 1.4 Hyperparameters (memorize)

| *Paramètre* | Too small | Too large |
|---|---|---|
| α | Very slow learning | Unstable values, oscillation |
| γ | Myopic, ignores the distant goal | Little incentive to shorten the path |
| ε | Freezes on a bad solution | Stays too random |
| *Épisodes* | Not enough time to learn | Wasted compute |

### 1.5 Diagnosis (memorize)

| *Symptôme* | Probable cause | Fix |
|---|---|---|
| Flat curve at 0 | Sparse reward, ε decays too fast, too few episodes | Slower ε decay, more episodes |
| Rises then oscillates or collapses | α too large | Lower α |
| Good in training, bad in evaluation | Too few eval episodes, or ε still high | Greedy eval, more episodes |
| Very different results across seeds | High variance | Average over seeds |
| Path uselessly long | γ = 1 and no step penalty | Slightly lower γ, or a small step penalty |

---

## 2. Worked example: is 74% better than 70%?

p = 0.74.

- n = 100: √(0.74 × 0.26 / 100) = √0.0019 ≈ **0.044**. A 4-point gap is within one standard error: noise.
- n = 1,000: √(0.74 × 0.26 / 1000) ≈ **0.014**. Now 4 points is almost three standard errors: probably real.

---

## 3. Code

```python
import numpy as np

def moving_average(x, window):
    return np.convolve(x, np.ones(___) / window, mode="valid")

def evaluate(env, Q, n_episodes=1000, seed=10_000):
    """Greedy, no learning. Returns the success rate (FrozenLake) or mean return."""
    total = 0.0
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            a = int(np.argmax(___))               # ε = 0
            s, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
    return total / ___

def run_experiment(train_fn, env, seeds=range(5), n_eval=1000, **kwargs):
    curves, scores = [], []
    for s in seeds:
        Q, returns = train_fn(env, seed=s, **kwargs)
        curves.append(moving_average(returns, ___))
        scores.append(evaluate(env, Q, n_episodes=___))
    curves = np.array(curves)
    return curves.mean(0), curves.std(0), np.mean(scores), np.std(scores)
```

Plot the mean curve with a shaded ± std band: `plt.fill_between(x, mean - std, mean + std, alpha=0.3)`.

---

## 4. Exercises

**E1 (by hand).** A random policy gets 1.5% success on slippery FrozenLake, your agent 72% ± 3% over 5 seeds (1,000 eval episodes each). Write the two sentences you'd put in a report.

**E2.** p = 0.5. How many evaluation episodes do you need for a standard error of 0.01?

**E3. The mini lab report (90 min, the best exam preparation here).** One page, in French, on slippery FrozenLake, comparing DP (02), MC (03), SARSA and Q-learning (04-05):
1. Table: method, episodes, success rate mean ± std over 5 seeds, 1,000 greedy eval episodes each.
2. One figure: learning curves (mean + std band) for the three model-free methods.
3. Five sentences of analysis: fastest, most stable, distance to the DP optimum, and why. Use the words *biais*, *variance*, *bootstrapping*, *on-policy*, *off-policy*.

---

## 5. Self-check (closed book)

1. Why evaluate with ε = 0 and no updates?
2. Why several seeds? Why a random baseline?
3. A curve rises then oscillates: cause and fix?
4. What is the standard error of a success rate, and what does it tell you?

---

## Answers

- **E1 (example):** "Q-learning reaches 72% ± 3% success (5 seeds, 1,000 greedy evaluation episodes each), against 1.5% for a random policy. The spread between seeds is small, so the result is reproducible."
- **E2:** √(0.25/n) = 0.01 → n = 2,500.
- **Self-check 1:** random actions lower the score and hide what the agent knows; updating during evaluation would change the agent being measured.
- **Self-check 2:** RL is random, one run can be lucky; a score only means something compared to a reference.
- **Self-check 3:** α too large → lower α.
- **Self-check 4:** √(p(1−p)/n): the noise in the measured rate; gaps smaller than ~2 standard errors aren't meaningful.

**Next:** [07 — Exam question bank](07-exam-question-bank.md)
