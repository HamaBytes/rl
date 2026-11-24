# Course 02: Q-learning in depth

*Written 23 Nov 2026, for the ML block of 24 Nov or later. Read Course 01 first; this one goes underneath it.*

**Goal:** be able to *derive* Q-learning, say *why* it works and *when* it doesn't, and defend every design choice in your reproduction of the paper (Table II, best score −108) in front of your professor.

**Time:** about 3 sessions of 1.5 h. Do the exercises on paper. There is no solution code in this file: you write all of it.

---

## Sources and how to use them

| Source | What to read / watch | Verified? |
|---|---|---|
| Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed. (2018), free PDF on the authors' site (incompleteideas.net) | Ch. 3 *Finite Markov Decision Processes* (3.1–3.7) | ✅ section titles checked against the book's table of contents |
| same | Ch. 5 *Monte Carlo Methods*, 5.1 *Monte Carlo Prediction* | ✅ |
| same | Ch. 6 *Temporal-Difference Learning*: 6.1 TD Prediction, 6.2 Advantages of TD, 6.4 Sarsa, 6.5 Q-learning, 6.6 Expected Sarsa, 6.7 Maximization Bias and Double Learning | ✅ |
| same | 9.5 *Feature Construction for Linear Methods*: 9.5.3 Coarse Coding (p. 215), 9.5.4 Tile Coding (p. 217) | ✅ |
| same | Ch. 10 *On-policy Control with Approximation*, 10.1 *Episodic Semi-gradient Control* (this is where the book solves Mountain Car) | ✅ title checked; the Mountain Car example number was **not** verified |
| David Silver, UCL RL course (2015), YouTube playlist `PLzuuYNsE1EZAXYR4FJ75jcJseBmo4KQ9-` | Lecture 2 *Markov Decision Processes*, Lecture 4 *Model-Free Prediction*, Lecture 5 *Model-Free Control*, Lecture 6 *Value Function Approximation* | ✅ titles checked. Timestamps **not** verified, so none are given here; use the slide titles to navigate |
| Gymnasium 1.4.0 source, `gymnasium/envs/classic_control/mountain_car.py` | the physics in §1 below | ✅ read directly from the installed code |

Things I state from memory and could not check at the source are marked *(unverified)*. Check them in the book before you quote them in your report.

---

## 0. Map of this course

```
1. MountainCar as an MDP          → what exactly are S, A, P, R, γ here?
2. Returns and value functions    → what are we trying to estimate?
3. Bellman equations (derived)    → the equation Q-learning is built on
4. Monte Carlo vs TD              → two ways to learn from experience
5. Sarsa vs Q-learning            → on-policy vs off-policy control
6. Convergence: when it works     → the conditions, and how your setup breaks them
7. Discretization trade-offs      → what the paper's "tabular discretization" really costs
8. Tile coding                    → the classic better answer for MountainCar
9. Exercises tied to the paper    → feeds your 3 hypotheses
```

---

## 1. MountainCar as a Markov Decision Process

An MDP is a tuple **(S, A, P, R, γ)** (Sutton & Barto ch. 3; Silver lecture 2).

| Piece | Meaning | MountainCar-v0 (from the gymnasium source) |
|---|---|---|
| S | states | (position p, velocity v), p ∈ [−1.2, 0.6], v ∈ [−0.07, 0.07] |
| A | actions | {0 = push left, 1 = no push, 2 = push right} |
| P | transition dynamics | **deterministic**, see below |
| R | reward | −1 on every step, including the last one |
| γ | discount | **your choice**; the environment doesn't define it |

**The real physics, one step** (this is the actual code logic, written as math):

```
v_{t+1} = clip( v_t + (a − 1)·0.001 − 0.0025·cos(3·p_t),  −0.07, 0.07 )
p_{t+1} = clip( p_t + v_{t+1},  −1.2, 0.6 )
if p_{t+1} = −1.2 and v_{t+1} < 0:  v_{t+1} = 0          (hits the left wall)
terminated  ⇔  p_{t+1} ≥ 0.5  (and v ≥ 0)
start: p_0 uniform in [−0.6, −0.4], v_0 = 0
truncated after 200 steps (TimeLimit wrapper)
```

Look at the numbers: the engine adds at most **0.001** per step, while gravity can pull with up to **0.0025**. The engine is **weaker than gravity**. That is *why* the car must swing back and forth. It isn't a detail; it's the whole problem.

### The Markov property
A state is Markov if the future depends only on the current state, not on how you got there. (p, v) **is** Markov: the equations above use nothing else.

**Important consequence for you:** once you discretize, the bin index is **not** Markov anymore. Two different (p, v) in the same cell can have different futures. Keep this in mind for §7.

**Exercise 1.1:** start at p = −0.5, v = 0, and push right (a = 2). Compute v₁ and p₁ by hand (cos(−1.5) ≈ 0.0707). Is the car moving right?
**Exercise 1.2:** same start, but push right forever. Without computing every step, explain why the car can't reach p = 0.5. (Use the 0.001 vs 0.0025 comparison.)

---

## 2. Returns and value functions

**Return** from time t (S&B 3.3):

```
G_t = R_{t+1} + γ R_{t+2} + γ² R_{t+3} + …  =  R_{t+1} + γ G_{t+1}
```

The last form, *return = reward now + γ × return from the next step*, is the single most important line in RL. Everything below comes from it.

**Value functions** (S&B 3.5): for a policy π,

```
v_π(s)    = E_π[ G_t | S_t = s ]                 how good is this state, if I follow π?
q_π(s, a) = E_π[ G_t | S_t = s, A_t = a ]        how good is this action here, then following π?
```

**Optimal** (S&B 3.6): q*(s, a) = max over all policies of q_π(s, a). If you know q*, the optimal policy is simply **π*(s) = argmax_a q*(s, a)**. That's why Q-learning targets q* directly.

### What G looks like in MountainCar
Every reward is −1. If the car reaches the flag after T steps:

```
G_0 = −(1 + γ + γ² + … + γ^{T−1}) = −(1 − γ^T) / (1 − γ)
```

**Exercise 2.1:** with γ = 0.99, compute G₀ for T = 100 and T = 200. (Hint: 0.99¹⁰⁰ ≈ 0.366, 0.99²⁰⁰ ≈ 0.134.) What's the difference between the two?
**Exercise 2.2:** same for γ = 0.9 (0.9¹⁰⁰ ≈ 2.7·10⁻⁵). Now what's the difference? What does this tell you about whether a small γ can tell "fast" apart from "slow"? **This is the core of your H3.**

---

## 3. The Bellman equations, derived

### 3.1 Bellman expectation equation for q_π
Start from q_π(s, a) = E[G_t | s, a] and substitute G_t = R_{t+1} + γ G_{t+1}:

```
q_π(s, a) = E[ R_{t+1} + γ G_{t+1} | S_t = s, A_t = a ]
          = E[ R_{t+1} | s, a ] + γ E[ G_{t+1} | s, a ]                      (linearity)
          = Σ_{s', r} p(s', r | s, a) [ r + γ Σ_{a'} π(a'|s') q_π(s', a') ]  (condition on s', a')
```

Step 3 uses the Markov property: once you know s', what happened before doesn't matter, so E[G_{t+1} | s'] is just the value of s' under π.

In words: **value here = immediate reward + discounted value of where I land.** It's a consistency condition: a true q_π must satisfy it in every state.

### 3.2 Bellman optimality equation for q*
If from s' on you act optimally, you pick the best a' instead of averaging over π:

```
q*(s, a) = Σ_{s', r} p(s', r | s, a) [ r + γ max_{a'} q*(s', a') ]
```

MountainCar is deterministic, so the sum has exactly one term:

```
q*(s, a) = r + γ max_{a'} q*(s', a')       (s' = the next state given by the physics)
```

### 3.3 From the equation to an algorithm
- If you **knew** p(s', r | s, a), you could iterate this equation directly: that's **dynamic programming** (S&B ch. 4; Silver lecture 3).
- You **don't** use the model (the paper treats the env as a black box). So you **sample** it instead: take an action, observe (r, s'), and nudge Q(s, a) toward the sampled right-hand side. That nudge **is** the Q-learning update:

```
Q(s, a) ← Q(s, a) + α [ r + γ max_{a'} Q(s', a')  −  Q(s, a) ]
                        └─── one sample of the Bellman optimality RHS ───┘
```

So Q-learning = **stochastic, sample-based value iteration on the Bellman optimality equation.** If you can say that sentence and explain it, you understand Q-learning.

**Exercise 3.1:** write the Bellman expectation equation for v_π(s) (states only) by following the same 3 steps. Check it against S&B 3.5.
**Exercise 3.2:** why does Q-learning use **max** and not the average over π? What would you be estimating if you used the average?

---

## 4. Monte Carlo vs Temporal Difference

Two ways to estimate values from experience (S&B 5.1, 6.1, 6.2; Silver lecture 4).

| | Monte Carlo (MC) | Temporal Difference (TD) |
|---|---|---|
| Target | the real return G_t, after the episode ends | r + γ V(s'), immediately |
| Waits for episode end? | yes | no, updates every step |
| Bootstraps (uses its own estimates)? | no | **yes** |
| Bias | unbiased target | biased (V(s') may be wrong) |
| Variance | high (sum of many random rewards) | low (one reward + one estimate) |
| Needs Markov states? | less sensitive | more sensitive |

**Bootstrapping** = updating a guess using another guess. TD does it; MC doesn't. Q-learning is a TD method.

### Why this matters a lot in MountainCar
At the start, the car almost never reaches the flag. Every episode is truncated at 200 steps with return ≈ −200.
- MC learns from full returns: almost all returns are identical, so there's almost no signal for which action was better.
- TD can propagate information **backward step by step**: once one episode touches the flag, the cells near the flag get better values, then the cells leading to *those* cells, and so on, over many episodes.

**Exercise 4.1:** in the TD update, the "error" is δ = r + γV(s') − V(s). In MountainCar with Q initialized to 0, what is δ on the very first update, anywhere? What does that do to the value of the visited cell? Connect this to "optimistic initialization" from Course 01.
**Exercise 4.2:** in your own words: why does bootstrapping become *risky* when the state isn't Markov (e.g. after coarse discretization)?

---

## 5. Sarsa vs Q-learning: on-policy vs off-policy

(S&B 6.4–6.7; Silver lecture 5.)

| | Sarsa | Q-learning |
|---|---|---|
| Target | r + γ Q(s', **a'**), with a' the action actually taken next (ε-greedy) | r + γ **max_a'** Q(s', a') |
| Learns the value of… | the policy it's following, exploration included | the greedy (optimal) policy, whatever it's doing |
| Type | **on-policy** | **off-policy** |
| Name comes from | (S, A, R, S', A') | Watkins (1989) *(unverified detail)* |

**Off-policy** means the agent *behaves* with one policy (ε-greedy, to explore) but *learns about* another (greedy). That's why Q-learning is the natural choice when you want q*.

**The classic illustration:** S&B's *Cliff Walking* example in ch. 6 (example number unverified). Q-learning learns the optimal path right along the cliff edge but falls off more often while exploring; Sarsa learns a safer path because it accounts for its own exploration. **Read it.** Your professor may ask "why Q-learning and not Sarsa?".

**Two refinements in the same chapter:**
- **Expected Sarsa** (6.6): uses the *expected* value under π instead of a sampled a'. Lower variance.
- **Maximization bias / Double Q-learning** (6.7): taking a max over *noisy* estimates is biased upward. Double Q-learning keeps two tables to remove that bias. In MountainCar the noise mostly comes from discretization aliasing, so this could be a legitimate extra hypothesis.

**Exercise 5.1:** write the Sarsa update and the Q-learning update one above the other. Circle the single difference.
**Exercise 5.2:** with ε = 0 (pure greedy), are Sarsa and Q-learning the same algorithm? Explain.

---

## 6. Convergence: when does Q-learning actually work?

The theory (Watkins & Dayan, 1992, *"Q-learning"*, Machine Learning journal; summarized in S&B 6.5 *(exact wording unverified)*): **tabular** Q-learning converges to q* with probability 1 **if**:

1. **Every (s, a) pair keeps being updated** (visited infinitely often).
2. **The step sizes satisfy the Robbins–Monro conditions** for each (s, a):
   ```
   Σ_t α_t = ∞        (steps big enough to get anywhere)
   Σ_t α_t² < ∞       (steps shrink enough to settle down)
   ```
   Example: α_t = 1/n(s, a), where n is the visit count, satisfies both. A **constant** α satisfies the first but **not** the second.
3. The problem is a true **finite MDP** (Markov states, bounded rewards).

### How your setup breaks each condition (be honest about this in the report)

| Condition | In your project | Consequence |
|---|---|---|
| 1. all pairs visited | some cells are unreachable (e.g. high speed at the far left), and ε decays toward ~0 | those Q-values never update. That's fine if they're irrelevant, but they show up in your policy heatmap |
| 2. Robbins–Monro | the paper likely uses a constant α *(check)* | no strict convergence; Q keeps fluctuating. This is normal in practice, and it's why your curves stay noisy |
| 3. Markov finite MDP | discretized cells are **not** Markov (§1) | the guarantee doesn't apply at all. Q-learning on bins is a heuristic that usually works, not a proven method |

This table alone is a strong "Discussion / Limitations" paragraph for your report.

**Exercise 6.1:** does α_t = 1/√n satisfy both conditions? What about α_t = 0.1?
**Exercise 6.2:** why does a decaying ε conflict with condition 1? What's the usual practical compromise?

---

## 7. Discretization: the real trade-offs

The paper's method is "tabular discretization". Here is what that costs.

| Fewer bins (coarse) | More bins (fine) |
|---|---|
| fewer cells, so each cell is visited often and learns fast | many cells, each visited rarely, so learning is slow |
| large cells mix very different states (**aliasing**, non-Markov) | small cells are nearly Markov |
| policy can be too crude to swing up efficiently | policy can be precise |
| **generalizes** (one update affects a big region) | **no generalization** between neighbouring cells |

There is a **sweet spot**, and it depends on how many training episodes you allow. That's a testable, quantitative hypothesis, i.e. your **H1**.

**Curse of dimensionality:** N bins per dimension × d dimensions = N^d cells. MountainCar has d = 2, so it's fine. A robot arm with d = 12 at 20 bins = 4·10¹⁵ cells is impossible. That's the honest reason tabular methods don't scale to robotics, and why §8 and function approximation exist (S&B ch. 9–10; Silver lecture 6).

**Uniform vs non-uniform bins:** velocity is tiny (±0.07) and often near 0, while position spans 1.8. Equal bin counts for both may not be ideal. Check what the paper does.

**Exercise 7.1:** with 20 × 20 bins and 3 actions, how many Q-values are there? With 40 × 40? If each training episode visits at most 200 cells, roughly how many episodes would it take for each cell to be visited even once?
**Exercise 7.2:** draw (by hand) the 2D state space and sketch where you think the car spends most of its time early in training. Which cells will be "empty"?

---

## 8. Tile coding: the classic better answer for MountainCar

(S&B 9.5.3 Coarse Coding, 9.5.4 Tile Coding; then 10.1, where the book applies it to Mountain Car.)

**Idea:** instead of **one** grid, use **several** grids (*tilings*), each **shifted** by a fraction of a cell.

```
tiling 1:  |----|----|----|
tiling 2:    |----|----|----|        (shifted)
tiling 3:      |----|----|----|      (shifted more)
             ●
```

- A state activates **one tile per tiling**, so with 8 tilings, 8 active tiles.
- Q(s, a) = **sum** of the weights of the active tiles (for action a).
- Update: every active tile's weight moves toward the target, usually with step size α / (number of tilings).

**Why it's better than one fine grid:**
- **Generalization:** nearby states share some tiles, so learning in one spot helps its neighbours.
- **Resolution:** the overlap of shifted tilings distinguishes states more finely than any single tiling.
- **Cost:** still cheap, just a few lookups and additions.

**Important vocabulary shift:** with tile coding, Q is no longer a table; it's a **linear function** of binary features. The update becomes a **semi-gradient** update (S&B ch. 9–10), and the tabular convergence guarantee of §6 no longer applies as-is.

**For your project:** tile coding is **not** what the paper does, so it's **not** part of the reproduction. It's a strong candidate for an **optional 4th hypothesis**: *"tile coding with K tilings reaches −110 in fewer episodes than the paper's best single grid."* It's much closer to the paper's method than LightGBM, and it fits in the same code with modest changes.

**Exercise 8.1:** with 8 tilings of 8 × 8 tiles each, how many weights are there per action? Compare with a single 64 × 64 grid. Which has more parameters, and which generalizes more?
**Exercise 8.2:** explain in 3 sentences, to your professor, why tile coding is a form of *function approximation* even though it's built from grids.

---

## 9. Exercises tied to the paper (do these before writing `q_learning.py`)

Your target: the paper's Table II (5 configurations, best −108, solved threshold −110).

1. **Extract the spec.** From the paper's method section, fill this table in `notes.md`:

   | Config | pos bins | vel bins | α | γ | ε start → end, decay | episodes | Q init | result |
   |---|---|---|---|---|---|---|---|---|

   Anything the paper doesn't state is a **gap**. List the gaps: they are things you must choose and justify, and a professor values that honesty.

2. **Terminal handling.** Does the paper bootstrap on `truncated`? If it doesn't say, decide, and write one paragraph using §5–6 to justify it.

3. **Predict before you run.** For each of the 5 configs, *predict* the ranking from §6–7 theory, before you have any code. After you reproduce Table II, compare. A correct prediction shows understanding; a wrong one, explained, is an even better discussion section.

4. **Turn the theory into your 3 hypotheses.** Write each in this format:
   > **H1:** "Increasing bins from N₁ to N₂ improves the final average reward over the last 100 episodes, up to N*, after which it decreases within a fixed budget of E episodes." *Test:* vary bins only, 5 seeds each. *Evidence that would reject it:* …

   Suggested sources for H1/H2/H3: §7 (bins), §4 + §6 (ε schedule / exploration), §2 (γ).

5. **Definition of "solved".** What exactly is "−108"? A single episode, or an average over the last 100? Find it in the paper. If you compare different quantities, your reproduction is invalid.

---

## 10. Self-check (closed book, in `notes.md`)

1. Define an MDP and give each element for MountainCar-v0.
2. Derive G_t = R_{t+1} + γ G_{t+1} from the definition of G_t.
3. Derive the Bellman optimality equation for q* in 3 lines.
4. Explain "Q-learning is sample-based value iteration" in 2 sentences.
5. MC vs TD: one advantage of each, and which one MountainCar favours, and why.
6. On-policy vs off-policy: what does Sarsa learn that Q-learning doesn't?
7. State the 3 convergence conditions and which ones your project violates.
8. Why are discretized states non-Markov, and why does that matter for TD?
9. Why does tile coding generalize better than one fine grid?
10. Why does the engine (0.001) vs gravity (0.0025) explain the whole difficulty of MountainCar?

**Next:** Mission 2, `discretize()` alone, tested by hand against your answers to Exercise 7.1 and Course 01 Exercise 3.1.
