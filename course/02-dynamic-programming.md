# 02 — Dynamic programming (*Programmation dynamique*)

> **Goal:** compute V and the optimal policy exactly when the model is known, and understand policy iteration vs value iteration.
> **You need:** 01. **Time:** about 3 h. **Class:** slides 22-31.
> **Reading:** Sutton & Barto ch. 4; David Silver, lecture 3.

DP assumes you **know the model** P(s'|s,a) and R. FrozenLake exposes its model, so you can code real DP on it.

---

## 1. Concepts

### 1.1 Policy evaluation (*évaluation de politique*)

Given π, compute V^π by repeating the Bellman expectation backup until values move less than a small θ:

```
V_(k+1)(s) ← Σ_a π(a|s) Σ_s' P(s'|s,a) [ R + γ V_k(s') ]
```

Updating V in place (using new values as soon as they're computed) also converges, often faster.

### 1.2 Policy improvement (*amélioration de politique*)

Act greedily with respect to V^π:

```
π'(s) = argmax_a Σ_s' P(s'|s,a) [ R + γ V^π(s') ]
```

*Policy improvement theorem:* π' is at least as good as π in every state. If π' = π, π is optimal.

### 1.3 Policy iteration (*itération de politique*)

Evaluate → improve → evaluate → … until the policy stops changing. Few outer iterations, each one a full evaluation.

### 1.4 Value iteration (*itération de valeur*)

Fold the max into the backup:

```
V_(k+1)(s) ← max_a Σ_s' P(s'|s,a) [ R + γ V_k(s') ]
```

Then read the policy once with argmax at the end. Cheaper sweeps, more of them.

### 1.5 Limits (slide 31)

- Needs the full model.
- Each sweep costs |S|×|A|×(next states): *malédiction de la dimensionnalité*.
- Can need many sweeps.

### 1.6 Generalized Policy Iteration (GPI)

Not in your slides, but it unifies the whole course: any method that alternates "estimate values" and "make the policy greedier", even partially, heads to the optimum. Monte Carlo control (03), Q-learning (04) and SARSA (05) are all GPI without a model.

### 1.7 FrozenLake's model in Gymnasium

```python
import gymnasium as gym
env = gym.make("FrozenLake-v1", is_slippery=True)
P = env.unwrapped.P      # P[s][a] = list of (prob, next_state, reward, terminated)
print(P[0][2])           # what "right" does from square 0
```

Run it first and look. With `is_slippery=True` you'll see three outcomes of probability 1/3 each.

---

## 2. Worked example: policy evaluation by hand

3-state chain Start → Middle → Goal (terminal), reward +1 on entering Goal, γ = 0.9, one action. Start from V = 0 and update in place, Start first:

| Sweep | V(Start) | V(Middle) | Biggest change |
|---|---|---|---|
| 0 | 0 | 0 | |
| 1 | 0 + 0.9×0 = 0 | 1 + 0.9×0 = 1 | 1 |
| 2 | 0 + 0.9×1 = 0.9 | 1 | 0.9 |
| 3 | 0.9 | 1 | 0 → stop |

If you swept Middle first, it would converge in 2 sweeps. Order matters for speed, not for the final answer.

---

## 3. Code

```python
import numpy as np

def policy_evaluation(P, policy, n_states, gamma=0.99, theta=1e-8):
    """policy: array (n_states, n_actions) of probabilities."""
    V = np.zeros(n_states)
    while True:
        delta = 0.0
        for s in range(n_states):
            v_new = 0.0
            for a, pi_sa in enumerate(policy[s]):
                for prob, s2, r, term in P[s][a]:
                    v_new += pi_sa * prob * (r + gamma * ___ * (not term))
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new                     # in-place update
        if delta < ___:
            return V

def q_from_v(P, V, s, n_actions, gamma):
    q = np.zeros(n_actions)
    for a in range(n_actions):
        for prob, s2, r, term in P[s][a]:
            q[a] += ___
    return q

def policy_improvement(P, V, n_states, n_actions, gamma):
    policy = np.zeros((n_states, n_actions))
    for s in range(n_states):
        best = np.argmax(q_from_v(P, V, s, n_actions, gamma))
        policy[s, best] = ___
    return policy

def policy_iteration(P, n_states, n_actions, gamma=0.99):
    policy = np.ones((n_states, n_actions)) / n_actions   # uniform random start
    while True:
        V = policy_evaluation(P, policy, n_states, gamma)
        new_policy = policy_improvement(P, V, n_states, n_actions, gamma)
        if np.array_equal(___, ___):
            return policy, V
        policy = new_policy

def value_iteration(P, n_states, n_actions, gamma=0.99, theta=1e-8):
    V = np.zeros(n_states)
    while True:
        delta = 0.0
        for s in range(n_states):
            v_new = np.max(___)
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < theta:
            break
    return policy_improvement(P, V, n_states, n_actions, gamma), V
```

---

## 4. Exercises

**E1 (by hand).** Redo the class 2×2 Gridworld (slides 24-27) with γ = 0.9 and the uniform *Droite/Bas* policy. Then do one **improvement** step: what is the new policy in S1, S3, S4?

**E2.** Run `policy_iteration` and `value_iteration` on slippery FrozenLake. Print V as a 4×4 grid and the policy as arrows. Same policy? Count the iterations each needed.

**E3.** Evaluate the DP policy over 1,000 episodes with the `evaluate()` function from the class docx. This is the best success rate any method can reach on slippery FrozenLake. (I measured about 0.74 with γ = 0.99 over 20,000 episodes on gymnasium 1.4 with the default 100-step limit: check that you get the same.) Keep the number: modules 04-06 compare against it.

**E4.** Change γ to 0.9 and 0.5. What happens to V near the start square? Does the policy change? Why?

**E5.** Why does the slippery optimal policy sometimes point *away* from the goal near holes?

---

## 5. Self-check (closed book)

1. Policy iteration vs value iteration, one sentence each.
2. What does DP need that Q-learning doesn't?
3. Why is in-place updating of V allowed?
4. Stopping criterion of policy evaluation? Of policy iteration?
5. What is GPI, and which later methods are GPI?

---

## Answers

- **E1:** after evaluation V(S1) = 0.5, V(S3) = 0, V(S4) = 0 (S4 has no path in the class example). Improvement: in S1, *Droite* gives 1 vs *Bas* 0.9×0 = 0, so *Droite*. In S3 and S4 all actions are worth 0: ties, so any choice is "greedy". This is why tie-breaking matters.
- **E2:** same policy (both reach π*). Policy iteration: few outer iterations. Value iteration: many sweeps.
- **E4:** smaller γ makes values shrink fast with distance from the goal; the policy can switch toward riskier, shorter paths.
- **E5:** on slippery ice, pushing "against" a hole means the 1/3 side-slips can't drop you into it. Safety beats directness.
- **Self-check 1:** PI alternates full evaluation and improvement; VI does one max-backup per sweep and extracts the policy at the end.
- **Self-check 2:** the model P and R.
- **Self-check 3:** it still converges, since each backup uses estimates that are at least as recent.
- **Self-check 4:** max change in V below θ; policy unchanged between two improvements.
- **Self-check 5:** alternating evaluation and greedy improvement, even partially. MC control, SARSA, Q-learning.

**Next:** [03 — Monte Carlo](03-monte-carlo.md)
