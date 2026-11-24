# A4 — Différence temporelle: TD(0), SARSA, Q-learning, Expected SARSA

Class: slides 44-57 and the FrozenLake docx. Q-learning intuition and the update rule: [01-q-learning.md](../01-q-learning.md) sections 4-7. On-policy vs off-policy in depth: [02-q-learning-deep.md](../02-q-learning-deep.md) section 5.

## 1. TD = sampling + bootstrapping

- *Échantillonnage* (like MC): learn from real experience.
- *Bootstrapping* (like DP): the target uses your own current estimate of the next state.

**TD(0) prediction:**

V(s) ← V(s) + α [ r + γ V(s') − V(s) ]

δ = r + γ V(s') − V(s) is the **erreur TD**.

**Bias/variance trade-off:** the TD target is biased (it uses an estimate) but low variance (one step of randomness). MC is unbiased but high variance. This sentence answers half of all exam questions comparing MC and TD.

## 2. The three control algorithms (only the target differs)

| Algorithm | Target | Type |
|---|---|---|
| SARSA | r + γ Q(s', a') where a' is the action actually chosen next (ε-greedy) | on-policy |
| Q-learning | r + γ max_a' Q(s', a') | off-policy |
| Expected SARSA | r + γ Σ_a' π(a'\|s') Q(s', a') | either (on-policy when π is the behaviour policy) |

All three: Q(s,a) ← Q(s,a) + α [ target − Q(s,a) ]. If s' is terminal (`terminated`), the target is just r.

**On-policy:** learns the value of the policy it actually follows, exploration included.
**Off-policy:** learns the value of a different policy (here, greedy) than the one generating data.

SARSA's name = the 5 elements it uses: (S_t, A_t, R_(t+1), S_(t+1), A_(t+1)).

## 3. CliffWalking: the classic SARSA vs Q-learning demo

Not in your slides, but it's the standard way to *see* on-policy vs off-policy (Sutton & Barto, Example 6.6).

4×12 grid, start bottom-left, goal bottom-right, the bottom row between them is a cliff (−100, back to start), every step −1.

- **Q-learning** learns the optimal path along the cliff edge. But while it still explores with ε, it falls often, so its *online* reward during training is worse.
- **SARSA** knows it explores, so it learns a safer path one or two rows up. Better online reward, slightly longer path.

Robotics link: when exploration on real hardware is dangerous, on-policy caution matters.

In Gymnasium 1.4 the environment is `CliffWalking-v1` (older versions: `CliffWalking-v0`). Check with `[k for k in gym.registry if 'Cliff' in k]`.

## 4. Skeletons

Reuse your `choose_action` (ε-greedy with random tie-break) from the prep sheet.

```python
def sarsa(env, n_episodes, alpha=0.1, gamma=0.99, eps_start=1.0,
          eps_min=0.05, eps_decay=0.999, seed=0):
    rng = np.random.default_rng(seed)
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    epsilon, returns = eps_start, []
    for _ in range(n_episodes):
        s, _ = env.reset()
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
    probs[best] += (1 - epsilon) / ___          # split greedy mass across ties
    return r + gamma * np.dot(___, ___) * (not terminated)
```

Q-learning: you already have it (docx step 5 and your prep sheet).

## 5. Exercises

**E1 (by hand).** Q(s,a)=0.4, r=−1, s' non-terminal, Q(s',·) = [0.2, 0.9, 0.5], the ε-greedy agent then picks a'=0. α=0.5, γ=1. Compute the new Q(s,a) for SARSA, Q-learning and Expected SARSA with ε=0.3.

**E2.** CliffWalking: train SARSA and Q-learning, 500 episodes, ε fixed at 0.1, α=0.5, γ=1, 10 seeds. Plot reward per episode (moving average 10). Then print both greedy paths on the grid. You should reproduce the Sutton & Barto picture.

**E3.** Same experiment with ε decaying to 0. What happens to the gap between SARSA and Q-learning? Why?

**E4.** FrozenLake slippery: Q-learning vs SARSA vs Expected SARSA, 20,000 episodes, 5 seeds, greedy evaluation on 1,000 episodes. Table: mean ± std. Compare with the DP optimum from A2.

**E5.** The docx TP items 1-5 (prise en main, courbe, hyperparamètres α and γ, baseline, slippery). If you did them in class, redo them alone, without the docx open.

## 6. Self-check

1. Why is Q-learning off-policy? Why is SARSA on-policy?
2. In CliffWalking, which algorithm gets the better online reward and which learns the better greedy path?
3. What is bootstrapping? Name one method that does it and one that doesn't.
4. Write all three targets from memory.
5. TD vs MC: which is biased, which has higher variance?

---

### Answers

- E1: SARSA target −1 + 0.2 = −0.8 → 0.4 + 0.5(−1.2) = −0.2. Q-learning target −1 + 0.9 = −0.1 → 0.4 + 0.5(−0.5) = 0.15. Expected SARSA: probs = [0.1, 0.8, 0.1] → Σ = 0.02 + 0.72 + 0.05 = 0.79, target −0.21 → 0.4 + 0.5(−0.61) = 0.095.
- E3: as ε → 0, SARSA's policy becomes greedy, so both converge to the optimal edge path and the gap closes.
- Self-check 2: SARSA better online reward; Q-learning better greedy path.
- Self-check 3: using your own estimate in the target. TD/DP do it, MC doesn't.
- Self-check 5: TD biased and lower variance; MC unbiased and higher variance.
