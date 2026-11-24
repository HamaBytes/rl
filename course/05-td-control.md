# 05 — The TD family: TD(0), SARSA, Expected SARSA, convergence (*Différence temporelle*)

> **Goal:** compare the TD methods, explain on-policy vs off-policy, MC vs TD, and say when Q-learning is guaranteed to work.
> **You need:** 03, 04. **Time:** about 3 h. **Class:** slides 44-57.
> **Reading:** Sutton & Barto 6.1-6.7; David Silver, lectures 4-5.

---

## 1. Concepts

### 1.1 TD = sampling + bootstrapping

- *Échantillonnage* (like MC): learn from real experience.
- *Bootstrapping* (like DP): the target uses your own current estimate of the next state.

**TD(0) prediction:**

```
V(s) ← V(s) + α [ r + γ V(s') − V(s) ]        δ = r + γ V(s') − V(s)   (erreur TD)
```

### 1.2 MC vs TD

| | Monte Carlo (03) | TD (04, 05) |
|---|---|---|
| Target | real return G_t, after the episode | r + γ V(s'), immediately |
| Waits for episode end? | yes | no, updates every step |
| Bootstraps? | no | **yes** |
| Bias | unbiased | biased (V(s') may be wrong) |
| Variance | high (many random rewards) | low (one reward + one estimate) |
| Needs Markov states? | less sensitive | more sensitive |

One sentence that answers half the exam questions comparing them: **TD is biased but low variance; MC is unbiased but high variance.**

Why TD wins in MountainCar: early on almost every episode is truncated at −200, so MC returns are nearly identical and carry no signal. TD propagates information **backward step by step**: once one episode touches the flag, the cells near it improve, then the cells leading to those, and so on.

### 1.3 Three control algorithms: only the target differs

| Algorithm | Target | Type |
|---|---|---|
| SARSA | r + γ Q(s', a'), with a' the action actually chosen next (ε-greedy) | on-policy |
| Q-learning | r + γ max_a' Q(s', a') | off-policy |
| Expected SARSA | r + γ Σ_a' π(a'\|s') Q(s', a') | on-policy when π is the behaviour policy; can be off-policy |

All three: Q(s,a) ← Q(s,a) + α [ target − Q(s,a) ]. If s' is terminal (`terminated`), target = r.

- **On-policy:** learns the value of the policy it actually follows, exploration included.
- **Off-policy:** learns about a different policy (here, greedy) than the one generating the data. That's why Q-learning is the natural choice when you want Q*.
- SARSA's name = its five elements (S_t, A_t, R_(t+1), S_(t+1), A_(t+1)).
- Expected SARSA averages over a' instead of sampling it: lower variance than SARSA.

### 1.4 CliffWalking: seeing on-policy vs off-policy

Not in your slides; it's the standard demo (Sutton & Barto, Example 6.6). A 4×12 grid, start bottom-left, goal bottom-right; the bottom row between them is a cliff (−100, back to start); every step −1.

- **Q-learning** learns the optimal path along the cliff edge, but while it explores with ε it falls often, so its *online* reward during training is worse.
- **SARSA** accounts for its own exploration and learns a safer path one or two rows up: better online reward, slightly longer path.

Robotics link: when exploring on real hardware is dangerous, on-policy caution matters. In gymnasium 1.4 the environment is `CliffWalking-v1` (check: `[k for k in gym.registry if "Cliff" in k]`).

### 1.5 Maximization bias and Double Q-learning (S&B 6.7)

Taking a max over *noisy* estimates is biased upward. Double Q-learning keeps two tables: one picks the best action, the other evaluates it. The same idea returns in Double DQN (11) and SAC's twin critics (13).

### 1.6 Convergence: when is Q-learning guaranteed to work?

Tabular Q-learning converges to Q* with probability 1 (Watkins & Dayan, 1992; summarized in S&B 6.5, *exact wording unverified*) **if**:

1. **Every (s,a) keeps being updated** (visited infinitely often).
2. **Step sizes satisfy the Robbins-Monro conditions** for each (s,a):
   ```
   Σ_t α_t = ∞     (big enough to get anywhere)
   Σ_t α_t² < ∞    (shrink enough to settle)
   ```
   α_t = 1/n(s,a) (n = visit count) satisfies both. A **constant** α satisfies the first, not the second.
3. The problem is a true **finite MDP** (Markov states, bounded rewards).

In practice people use a constant α and a decaying ε anyway: no strict guarantee, Q keeps fluctuating, curves stay noisy. That's normal. How your MountainCar project breaks each condition is in 09.

---

## 2. Worked example: the three targets on one transition

Q(s,a) = 0.4, r = −1, s' non-terminal, Q(s',·) = [0.2, 0.9, 0.5], the ε-greedy agent then picks a' = 0. α = 0.5, γ = 1, ε = 0.3.

- **SARSA:** target = −1 + Q(s', 0) = −0.8 → 0.4 + 0.5 × (−1.2) = **−0.2**
- **Q-learning:** target = −1 + 0.9 = −0.1 → 0.4 + 0.5 × (−0.5) = **0.15**
- **Expected SARSA:** ε-greedy probabilities = [0.1, 0.8, 0.1] (ε/3 each, plus 0.7 on the greedy action). Expected Q = 0.02 + 0.72 + 0.05 = 0.79 → target −0.21 → 0.4 + 0.5 × (−0.61) = **0.095**

---

## 3. Code

Reuse `choose_action` from 04.

```python
import numpy as np

def sarsa(env, n_episodes, alpha=0.1, gamma=0.99, eps_start=1.0,
          eps_min=0.05, eps_decay=0.999, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon, returns = eps_start, []
    for ep in range(n_episodes):
        s, _ = env.reset(seed=seed + ep)
        a = choose_action(Q, s, epsilon, rng)
        done, total = False, 0.0
        while not done:
            s2, r, terminated, truncated, _ = env.step(a)
            a2 = choose_action(Q, ___, epsilon, rng)
            target = r + gamma * ___ * (not terminated)
            Q[s, a] += alpha * (target - Q[s, a])
            s, a = ___, ___
            done = terminated or truncated
            total += r
        epsilon = max(eps_min, epsilon * eps_decay)
        returns.append(total)
    return Q, returns

def expected_sarsa_target(Q, s2, r, gamma, epsilon, terminated):
    nA = Q.shape[1]
    probs = np.full(nA, epsilon / nA)
    best = np.flatnonzero(Q[s2] == Q[s2].max())
    probs[best] += (1 - epsilon) / ___          # split the greedy mass across ties
    return r + gamma * np.dot(___, ___) * (not terminated)
```

---

## 4. Exercises

**E1 (by hand).** Q(s,a) = 0, r = −1, s' non-terminal, Q(s',·) = [−2, −1, −3], the agent then picks a' = 2. α = 0.1, γ = 0.9, ε = 0.3. New Q(s,a) for SARSA, Q-learning and Expected SARSA?

**E2.** Write the SARSA update and the Q-learning update one above the other. Circle the single difference. With ε = 0, are they the same algorithm?

**E3.** CliffWalking: SARSA vs Q-learning, 500 episodes, ε fixed at 0.1, α = 0.5, γ = 1, 10 seeds. Plot reward per episode (moving average 10), then print both greedy paths on the grid. You should reproduce the Sutton & Barto picture.

**E4.** Same, with ε decaying to 0. What happens to the gap? Why?

**E5.** Slippery FrozenLake: Q-learning vs SARSA vs Expected SARSA, 20,000 episodes, 5 seeds, greedy evaluation on 1,000 episodes. Table: mean ± std. Compare with the DP optimum (02 E3).

**E6.** Does α_t = 1/√n satisfy both Robbins-Monro conditions? α_t = 0.1? Why does a decaying ε conflict with condition 1?

**E7.** In your own words: why does bootstrapping become risky when the state isn't Markov (e.g. after coarse discretization)?

**E8.** The docx TP items 1-5 (*prise en main, courbe, hyperparamètres α et γ, baseline, slippery*). If you did them in class, redo them alone, without the docx open.

---

## 5. Self-check (closed book)

1. Why is Q-learning off-policy? Why is SARSA on-policy?
2. In CliffWalking, which algorithm gets the better online reward, and which learns the better greedy path?
3. What is bootstrapping? Name one method that does it and one that doesn't.
4. Write all three targets from memory.
5. TD vs MC: which is biased, which has higher variance?
6. State the three convergence conditions of tabular Q-learning.
7. What is maximization bias and what fixes it?

---

## Answers

- **E1:** SARSA: target −1 + 0.9×(−3) = −3.7 → −0.37. Q-learning: −1 + 0.9×(−1) = −1.9 → −0.19. Expected SARSA: probs [0.1, 0.8, 0.1], expected Q = −0.2 − 0.8 − 0.3 = −1.3, target −2.17 → −0.217.
- **E2:** Q(s',a') vs max Q(s',·). With ε = 0 the next action *is* the greedy one, so yes, they coincide.
- **E4:** as ε → 0 SARSA's policy becomes greedy, both converge to the edge path, the gap closes.
- **E6:** 1/√n: Σ 1/√n = ∞ (first holds), but Σ α² = Σ 1/n = ∞, so it fails the second. 0.1: fails the second. Decaying ε to ~0 means some pairs stop being tried; the practical compromise is a small floor (ε_min) and enough episodes.
- **E7:** the target uses V(s') of a cell that mixes different real states, so errors get copied and spread instead of averaging out.
- **Self-check 1:** Q-learning learns the greedy policy (max) while behaving ε-greedy; SARSA learns the value of the ε-greedy policy it follows.
- **Self-check 2:** SARSA: better online reward. Q-learning: better greedy path.
- **Self-check 3:** using your own estimate in the target. TD and DP do it; MC doesn't.
- **Self-check 5:** TD biased, lower variance; MC unbiased, higher variance.
- **Self-check 7:** max over noisy estimates overestimates; Double Q-learning (two tables) removes it.

**Next:** [06 — Evaluating an agent](06-evaluation.md)
