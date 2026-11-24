# 01 — MDP, return, Bellman equations (*Processus de Décision Markovien, équations de Bellman*)

> **Goal:** define an RL problem precisely, compute returns, and derive the Bellman equations that every later algorithm is built on.
> **You need:** 00. **Time:** about 2.5 h. **Class:** slides 1-21.
> **Reading:** Sutton & Barto ch. 3 (3.1-3.7); David Silver, lecture 2.

---

## 1. Concepts

### 1.1 Vocabulary (exam definitions)

| Term (FR) | Meaning |
|---|---|
| *Agent* | Decides and learns |
| *Environnement* | Everything outside the agent that reacts to its actions |
| *État* s | Description of the current situation |
| *Action* a | A choice available in s |
| *Récompense* r | Number received after an action: immediate feedback |
| *Politique* π(a\|s) | Probability of taking a in s. *Déterministe* (one action) or *stochastique* (a distribution) |
| *Valeur* V^π(s) | Expected return starting from s and following π |
| *Valeur d'action* Q^π(s,a) | Expected return starting from s, doing a, then following π |

**Reward vs value:** reward = good *now*; value = good *in the long run*. A square with reward 0 can have high value because it leads to the goal.

**Three kinds of learning:** *supervisé* (labelled examples), *non supervisé* (no labels, find structure), *par renforcement* (an environment and a reward signal; choose actions to maximize cumulative reward).

### 1.2 The MDP

MDP = (S, A, P, R, γ)

- S: states. A: actions.
- P(s'|s,a): transition probability (*la dynamique*).
- R(s,a) or R(s,a,s'): reward function.
- γ ∈ [0,1]: *facteur d'actualisation*.

**Markov property:** the future depends only on the current state and action, not on the history. A chess position is Markov: how you got there doesn't matter. A single photo of a moving ball is not: it doesn't show the velocity.

**Why an MDP alone isn't enough** (slide 14): it defines the problem, but gives no way to *evaluate* a policy, no way to *improve* it, no link between now and later rewards, and enumerating policies is exponential. The Bellman equations solve the first three.

### 1.3 The return (*retour*)

```
G_t = r_(t+1) + γ r_(t+2) + γ² r_(t+3) + …  =  r_(t+1) + γ G_(t+1)
```

The recursive form, *return = reward now + γ × return from the next step*, is the most important line in RL. Everything below comes from it.

- γ = 0: myopic agent (*myope*), only the next reward counts.
- γ → 1: far-sighted (*prévoyant*). With γ = 1 and no terminal state, G can be infinite, so continuing tasks need γ < 1.

### 1.4 Value functions

```
V^π(s)   = E_π[ G_t | S_t = s ]               how good is this state, if I follow π?
Q^π(s,a) = E_π[ G_t | S_t = s, A_t = a ]      how good is this action here, then following π?
V^π(s)   = Σ_a π(a|s) Q^π(s,a)
```

**Optimal:** Q*(s,a) = max over all policies of Q^π(s,a), and V*(s) = max_a Q*(s,a). If you know Q*, the optimal policy is simply **π*(s) = argmax_a Q*(s,a)**. That's why Q-learning targets Q* directly.

### 1.5 The Bellman equations, derived

**Expectation equation for Q^π.** Start from Q^π(s,a) = E[G_t | s, a] and substitute G_t = r_(t+1) + γ G_(t+1):

```
Q^π(s,a) = E[ r_(t+1) + γ G_(t+1) | s, a ]
         = E[ r_(t+1) | s, a ] + γ E[ G_(t+1) | s, a ]                       (linearity)
         = Σ_s' P(s'|s,a) [ R(s,a,s') + γ Σ_a' π(a'|s') Q^π(s',a') ]          (condition on s', a')
```

The last step uses the Markov property: once you know s', the past doesn't matter. The same three steps for states give:

```
V^π(s) = Σ_a π(a|s) Σ_s' P(s'|s,a) [ R(s,a,s') + γ V^π(s') ]
```

**Optimality equations.** If you act optimally from s' on, take the best a' instead of averaging over π:

```
V*(s)   = max_a Σ_s' P(s'|s,a) [ R + γ V*(s') ]
Q*(s,a) = Σ_s' P(s'|s,a) [ R + γ max_a' Q*(s',a') ]
```

Read them all as *value = immediate reward + discounted value of what comes next*. The expectation version averages over the policy; the optimality version takes the max.

**From equation to algorithm** (you'll see each in turn):
- Know P and R → iterate the equation directly: **dynamic programming** (02).
- Don't know them → **sample** them: take an action, see (r, s'), nudge Q(s,a) toward the sampled right-hand side. That nudge is **Q-learning** (04).

### 1.6 Running example: MountainCar as an MDP

| Piece | MountainCar-v0 (read from the gymnasium source) |
|---|---|
| S | (position p, velocity v), p ∈ [−1.2, 0.6], v ∈ [−0.07, 0.07] |
| A | {0 = push left, 1 = no push, 2 = push right} |
| P | **deterministic**, see below |
| R | −1 on every step, including the last |
| γ | **your choice**; the environment doesn't define it |

One step of the real physics:

```
v_(t+1) = clip( v_t + (a − 1)·0.001 − 0.0025·cos(3·p_t),  −0.07, 0.07 )
p_(t+1) = clip( p_t + v_(t+1),  −1.2, 0.6 )
if p_(t+1) = −1.2 and v_(t+1) < 0:  v_(t+1) = 0          (hits the left wall)
terminated ⇔ p_(t+1) ≥ 0.5 (and v ≥ 0)
start: p_0 uniform in [−0.6, −0.4], v_0 = 0; truncated after 200 steps
```

The engine adds at most **0.001** per step; gravity can pull with up to **0.0025**. The engine is **weaker than gravity**, so the car must swing back and forth. That is the whole problem.

(p, v) **is** Markov: the equations use nothing else. Once you cut it into bins (module 08), the bin index is **not** Markov anymore.

Since the dynamics are deterministic, Bellman optimality for MountainCar has a single term: Q*(s,a) = r + γ max_a' Q*(s',a').

---

## 2. Worked example: a 3-state chain

Start → Middle → Goal (terminal). Reward +1 on entering Goal, 0 otherwise, γ = 0.9, one action per state, V(Goal) = 0.

- V(Middle) = 1 + 0.9 × V(Goal) = 1 + 0 = **1**
- V(Start) = 0 + 0.9 × V(Middle) = **0.9**

Slide 17 puts the +1 into V(Goal) = 1 instead of on the transition, which shifts every value by one discount step (0.9, 0.81). Same policy either way.

**Return in MountainCar.** Every reward is −1. If the flag is reached after T steps:

```
G_0 = −(1 + γ + … + γ^(T−1)) = −(1 − γ^T) / (1 − γ)
```

With γ = 0.99 and T = 100: 0.99¹⁰⁰ ≈ 0.366, so G_0 = −(0.634 / 0.01) = **−63.4**.

---

## 3. Code: discounted return, two ways

```python
def discounted_return_forward(rewards, gamma):
    return sum(___ * r for t, r in enumerate(rewards))

def discounted_return_backward(rewards, gamma):
    G = 0.0
    for r in reversed(___):
        G = ___ + gamma * ___
    return G
```

Test both on E1 and on the MountainCar formula above (a list of 100 values of −1).

---

## 4. Exercises (by hand first: exam style)

**E1.** Rewards 0, 0, 0, 1, then the episode ends. Compute G_0 for γ = 0.9 and γ = 0.5. Which agent "cares more" about the goal?

**E2.** A state s has two actions. `a1` gives r = 1 and goes to terminal. `a2` gives r = 0 and goes to s2, where V*(s2) = 2. For which γ is `a2` better?

**E3.** π picks *Droite* with 0.7 and *Bas* with 0.3. Q^π(s, Droite) = 0.6, Q^π(s, Bas) = 0.2. Compute V^π(s).

**E4.** Stochastic transition: from s, action a lands in s1 with 0.8 (r = 0, V(s1) = 1) and in s2 with 0.2 (r = −1, V(s2) = 0). γ = 0.9. Compute the one-step Bellman backup for this action.

**E5 (MountainCar physics).** Start at p = −0.5, v = 0, push right (a = 2). Compute v₁ and p₁ (cos(−1.5) ≈ 0.0707). Is the car moving right? Then explain, without computing every step, why pushing right forever can't reach p = 0.5.

**E6 (MountainCar return).** With γ = 0.99, compute G_0 for T = 100 and T = 200 (0.99²⁰⁰ ≈ 0.134). Then the same with γ = 0.9 (0.9¹⁰⁰ ≈ 2.7·10⁻⁵). In each case, how big is the difference between "fast" and "slow"? What does that say about small γ? (This is the core of your project's γ hypothesis.)

**E7.** Derive V^π(s) yourself, following the three steps used for Q^π in section 1.5. Check against S&B 3.5.

---

## 5. Self-check (closed book)

1. State the Markov property in one sentence and give an example where it fails.
2. Write the 5-tuple of an MDP and name each element in French.
3. Why do we need γ < 1 for continuing tasks?
4. Derive G_t = r_(t+1) + γ G_(t+1) from the definition of G_t.
5. Write Bellman optimality for Q* from memory.
6. How do you get π* from Q*? From V*, what extra do you need?
7. Why does Q-learning use **max** and not the average over π?
8. Why does "engine 0.001 vs gravity 0.0025" explain the whole difficulty of MountainCar?

---

## Answers

- **E1:** γ=0.9 → 0.9³ = 0.729. γ=0.5 → 0.5³ = 0.125. The γ=0.9 agent values the distant goal more.
- **E2:** a2 is better when 0 + 2γ > 1, i.e. γ > 0.5.
- **E3:** 0.7×0.6 + 0.3×0.2 = 0.48.
- **E4:** 0.8×(0 + 0.9×1) + 0.2×(−1 + 0.9×0) = 0.72 − 0.2 = 0.52.
- **E5:** v₁ = 0.001 − 0.0025×0.0707 ≈ 0.00082; p₁ ≈ −0.4992. Yes, slightly right. But on the right slope gravity's pull (up to 0.0025) exceeds the engine's 0.001, so the car stalls and rolls back: it must build momentum by swinging.
- **E6:** γ=0.99: T=100 → −63.4, T=200 → −86.6, a gap of about 23. γ=0.9: both ≈ −10.0 (gap ≈ 0.0003). With small γ the agent can barely tell a fast solution from a slow one (or from never arriving).
- **Self-check 1:** see 1.2; failure: a single camera frame of a moving ball.
- **Self-check 3:** otherwise the infinite sum of rewards may diverge.
- **Self-check 4:** G_t = r_(t+1) + γ(r_(t+2) + γ r_(t+3) + …) = r_(t+1) + γ G_(t+1).
- **Self-check 6:** π*(s) = argmax_a Q*(s,a). From V* you also need P and R (a one-step lookahead), which is why model-free methods learn Q, not V.
- **Self-check 7:** max gives the value of acting optimally afterwards (Q*). The average over π would estimate Q^π, the value of the current policy.
- **Self-check 8:** the car can't climb directly; reaching the flag needs a long sequence of swings, so reward information must travel back over ~100+ steps.

**Next:** [02 — Dynamic programming](02-dynamic-programming.md)
