# A1 — MDP, return, Bellman (*Processus de Décision Markovien, équations de Bellman*)

Class: slides 1-21. Deeper derivations: [02-q-learning-deep.md](../02-q-learning-deep.md) sections 1-3.

## 1. The vocabulary (exam definitions)

| Term (FR) | Meaning |
|---|---|
| Agent | Decides and learns |
| Environnement | Everything outside the agent that reacts to its actions |
| État s | Description of the current situation |
| Action a | A choice available in s |
| Récompense r | Number received after an action: immediate feedback |
| Politique π(a\|s) | Probability of taking a in s. *Déterministe* (one action) or *stochastique* (a distribution) |
| Valeur V^π(s) | Expected return starting from s and following π |
| Valeur d'action Q^π(s,a) | Expected return starting from s, doing a, then following π |

**Reward vs value:** reward = good *now*; value = good *in the long run*. A square with reward 0 can have high value because it leads to the goal.

**Three kinds of learning** (*supervisé / non supervisé / par renforcement*): labelled examples / no labels, find structure / an environment plus a reward signal, choose actions to maximize cumulative reward.

## 2. The MDP

MDP = (S, A, P, R, γ)

- P(s'|s,a): transition probability (the *dynamique*)
- R(s,a) or R(s,a,s'): reward function
- γ ∈ [0,1]: *facteur d'actualisation*

**Propriété de Markov:** the future depends only on the current state and action, not on the history. Chess position: how you got there doesn't matter.

**Why MDPs alone aren't enough** (slide 14): they define the problem but give no way to *evaluate* a policy, no way to *improve* it, they need a link between now and later rewards, and enumerating policies is exponential. Bellman solves the first three.

## 3. The return (*retour*)

G_t = r_(t+1) + γ r_(t+2) + γ² r_(t+3) + … = r_(t+1) + γ G_(t+1)

The second form (recursive) is the seed of everything else.

- γ = 0: myopic agent (*myope*), only the next reward counts.
- γ → 1: far-sighted (*prévoyant*). With γ = 1 and no terminal state, G can be infinite.

## 4. Bellman equations

**Expectation (for a given π):**

V^π(s) = Σ_a π(a|s) Σ_s' P(s'|s,a) [ R(s,a,s') + γ V^π(s') ]

Q^π(s,a) = Σ_s' P(s'|s,a) [ R(s,a,s') + γ Σ_a' π(a'|s') Q^π(s',a') ]

**Optimality:**

V*(s) = max_a Σ_s' P(s'|s,a) [ R + γ V*(s') ]

Q*(s,a) = Σ_s' P(s'|s,a) [ R + γ max_a' Q*(s',a') ]

π*(s) = argmax_a Q*(s,a)

Read them as: *value = immediate reward + discounted value of what comes next.* Expectation version averages over the policy; optimality version takes the max.

**Link V and Q:** V^π(s) = Σ_a π(a|s) Q^π(s,a). And V*(s) = max_a Q*(s,a).

## 5. Exercises (by hand, on paper: this is exam style)

**E1.** Rewards 0, 0, 0, 1, then the episode ends. Compute G_0 for γ = 0.9 and γ = 0.5. Which agent "cares more" about the goal?

**E2.** Class 3-state chain Start → Middle → Goal, reward +1 on entering Goal, γ = 0.9, terminal value 0. Compute V(Middle), V(Start). Then compare with slide 17's numbers (which set V(Goal) = 1) and explain the difference in one sentence.

**E3.** A state s has two actions. `a1` gives r = 1 and goes to terminal. `a2` gives r = 0 and goes to s2, where V*(s2) = 2. For which γ is `a2` better than `a1`?

**E4.** π picks *Droite* with 0.7 and *Bas* with 0.3. Q^π(s, Droite) = 0.6, Q^π(s, Bas) = 0.2. Compute V^π(s).

**E5.** Stochastic transition: from s, action a lands in s1 with 0.8 (r = 0, V(s1) = 1) and in s2 with 0.2 (r = −1, V(s2) = 0). γ = 0.9. Compute the one-step Bellman backup for this action.

**E6 (code, 15 min).** Write `discounted_return(rewards, gamma)` two ways: a forward loop with `gamma ** t`, and a backward loop using G = r + γG. Check both on E1.

```python
def discounted_return_backward(rewards, gamma):
    G = 0.0
    for r in reversed(___):
        G = ___ + gamma * ___
    return G
```

## 6. Self-check (closed book)

1. State the Markov property in one sentence and give an example where it fails.
2. Write the 5-tuple of an MDP and name each element in French.
3. Why do we need γ < 1 for continuing (never-ending) tasks?
4. Write Bellman optimality for Q* from memory.
5. How do you get π* from Q*? From V* (what extra do you need)?

---

### Answers

- E1: γ=0.9 → 0.9³ = 0.729. γ=0.5 → 0.125. The γ=0.9 agent values the distant goal more.
- E2: V(Middle) = 1 + 0.9×0 = 1, V(Start) = 0 + 0.9×1 = 0.9. Slide 17 puts the +1 into V(Goal) instead of the transition reward, which shifts every value by one discount step (0.9, 0.81). Same policy either way.
- E3: a2 better when 0 + γ×2 > 1 → γ > 0.5.
- E4: 0.7×0.6 + 0.3×0.2 = 0.48.
- E5: 0.8×(0 + 0.9×1) + 0.2×(−1 + 0.9×0) = 0.72 − 0.2 = 0.52.
- Self-check 3: otherwise the infinite sum of rewards may diverge. Self-check 5: π*(s) = argmax_a Q*(s,a). From V* you also need the model P and R (one-step lookahead), which is why model-free methods learn Q, not V.
- Markov failure example: a single camera frame of a moving ball (no velocity information).
