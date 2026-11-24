# A2 — Programmation dynamique (DP)

Class: slides 22-31. DP assumes you **know the model** P(s'|s,a) and R. FrozenLake exposes it, so you can code real DP.

## 1. The three algorithms

**Évaluation de politique** (policy evaluation): given π, compute V^π by repeating the Bellman expectation backup until values move less than θ.

V_(k+1)(s) ← Σ_a π(a|s) Σ_s' P(s'|s,a) [ R + γ V_k(s') ]

**Amélioration de politique** (policy improvement): act greedily with respect to V^π.

π'(s) = argmax_a Σ_s' P(s'|s,a) [ R + γ V^π(s') ]

*Policy improvement theorem:* π' is at least as good as π everywhere. If π' = π, it's optimal.

**Itération de politique** (policy iteration): evaluate → improve → evaluate → … until the policy stops changing. Converges in few iterations.

**Itération de valeur** (value iteration): fold the max into the backup.

V_(k+1)(s) ← max_a Σ_s' P(s'|s,a) [ R + γ V_k(s') ]

then read the policy once with argmax at the end.

**Limits (slide 31):** needs the full model, cost grows with |S|×|A| per sweep (*malédiction de la dimensionnalité*), can need many sweeps.

**Generalized Policy Iteration (GPI):** not in your slides, but it's the idea that unifies the whole course: any method that alternates "estimate values" and "make the policy greedier" (even partially) converges. MC control, SARSA and Q-learning are all GPI without a model.

## 2. FrozenLake's model in Gymnasium

```python
import gymnasium as gym
env = gym.make('FrozenLake-v1', is_slippery=True)
P = env.unwrapped.P      # dict: P[s][a] = list of (prob, next_state, reward, terminated)
print(P[0][2])           # what "right" does from square 0
```

Run that line first and look at the output. With `is_slippery=True` you'll see three outcomes of probability 1/3 each.

## 3. Skeletons

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
            V[s] = v_new          # in-place update (Gauss-Seidel style)
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
    policy = policy_improvement(P, V, n_states, n_actions, gamma)
    return policy, V
```

## 4. Exercises

**E1 (by hand).** Redo the class 2×2 Gridworld (slides 24-27) with γ = 0.9 and the uniform Droite/Bas policy. Then do one **improvement** step: what is the new policy in S1, S3, S4?

**E2.** Run `policy_iteration` and `value_iteration` on FrozenLake slippery. Print V as a 4×4 grid and the policy as arrows. Do both give the same policy? Count the iterations each needed.

**E3.** Evaluate the DP policy with the `evaluate()` function from the docx over 1000 episodes. This is the best success rate any method can reach on slippery FrozenLake (I measured about 0.74 with γ=0.99 over 20,000 episodes on Gymnasium 1.4, with the default 100-step limit: check you get the same). Keep the number: in A4 you compare Q-learning to it.

**E4.** Change γ to 0.9 and 0.5. What happens to V near the start square, and does the policy change? Explain.

**E5.** Why does the slippery optimal policy sometimes point *away* from the goal near holes?

## 5. Self-check

1. Difference between policy iteration and value iteration in one sentence each.
2. What does DP need that Q-learning doesn't?
3. Why is in-place updating of V allowed?
4. What is the stopping criterion of policy evaluation? Of policy iteration?

---

### Answers

- E1: after evaluation V(S1)=0.5, V(S3)=0, V(S4)=0 (S4 has no path in the class example). Improvement: in S1, Droite gives 1 vs Bas 0.9×0 = 0 → Droite. S3 and S4: all actions are worth 0, so ties; any choice is "greedy". This shows why ties matter.
- E2: same policy (both converge to π*); policy iteration needs few outer iterations, value iteration needs many sweeps.
- E4: smaller γ → values shrink fast with distance from the goal; the policy can change toward riskier, shorter paths.
- E5: on slippery ice, pushing "against" a hole means the 1/3 side-slips can't drop you in it. Safety beats directness.
- Self-check 3: it still converges (often faster), since each backup uses estimates that are at least as recent.
- Self-check 4: max change in V below θ; policy unchanged between two improvements.
