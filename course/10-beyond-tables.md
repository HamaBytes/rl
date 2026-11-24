# 10 — Beyond tables: n-step returns, eligibility traces, function approximation

> **Goal:** bridge tabular RL and deep RL: the n-step trade-off, the idea of traces, and replacing the table with a function.
> **You need:** Part 1, and 08 (tile coding, which isn't repeated here). **Time:** 6-10 h. **Class:** not in class (after 16 Jan).
> **Reading:** Sutton & Barto ch. 7, 9, 10, 11.3 (deadly triad), 12; David Silver, lecture 6.

---

## 1. Concepts

### 1.1 n-step returns: between TD(0) and Monte Carlo

```
G_(t:t+n) = r_(t+1) + γ r_(t+2) + … + γ^(n−1) r_(t+n) + γ^n V(s_(t+n))
```

- n = 1 → TD(0). n = ∞ (to the end) → Monte Carlo.
- Larger n: less bias, more variance. Middle values (n = 4 to 8) often learn fastest.

The same trade-off returns as GAE's λ in PPO (13). Understanding it here makes 13 much easier.

### 1.2 Eligibility traces and TD(λ) (know the idea)

Instead of choosing one n, average all n-step returns with weights (1−λ)λ^(n−1). Implemented with a trace e(s) that remembers recently visited states:

```
δ = r + γV(s') − V(s);   e(s) ← e(s) + 1;   for all s: V(s) ← V(s) + α δ e(s);   e ← γλ e
```

λ = 0 → TD(0); λ = 1 → close to MC.

### 1.3 Function approximation

When states are continuous or too many (images, joint angles), replace the table with a parametrized function V̂(s; w) or Q̂(s, a; w).

**Semi-gradient TD update:**

```
w ← w + α [ r + γ Q̂(s', a'; w) − Q̂(s, a; w) ] ∇_w Q̂(s, a; w)
```

"Semi" because we don't differentiate through the target, which also depends on w.

**Linear case:** Q̂(s,a;w) = w_a · φ(s), with features φ (tile coding, polynomials, RBFs). Then ∇ = φ(s).

### 1.4 The deadly triad (S&B ch. 11)

**Function approximation + bootstrapping + off-policy** together can diverge. Q-learning with a neural network has all three. DQN's tricks (11) exist to tame exactly this.

---

## 2. Worked example: 1-step vs 3-step target

Rewards −1, −1, −1 over three steps, γ = 0.9, current estimates V(s₁) = −4, V(s₃) = −5.

- 1-step: −1 + 0.9 × (−4) = **−4.6** (relies heavily on one estimate)
- 3-step: −1 − 0.9 − 0.81 + 0.729 × (−5) = −2.71 − 3.645 = **−6.355** (more real rewards, less weight on an estimate)

---

## 3. Code: linear semi-gradient SARSA on CartPole

```python
import numpy as np, gymnasium as gym

env = gym.make("CartPole-v1")
obs_dim, nA = env.observation_space.shape[0], env.action_space.n

def features(obs):
    # Simple start: bias + raw obs + squares. Later: RBF or tile coding.
    return np.concatenate([[1.0], obs, obs ** 2])

d = len(features(np.zeros(obs_dim)))
W = np.zeros((nA, d))

def q_values(obs):
    return W @ ___

def choose(obs, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(nA))
    return int(np.argmax(q_values(obs)))

# inside the loop:
# delta = r + gamma * q_values(obs2)[a2] * (not terminated) - q_values(obs)[a]
# W[a] += alpha * delta * ___
```

---

## 4. Exercises

**E1.** n-step SARSA on CliffWalking for n = 1, 2, 4, 8. Plot the reward curve for each. Which n learns fastest?

**E2.** Linear SARSA on CartPole with the features above. It will likely plateau low. Then try RBF features (`sklearn.kernel_approximation.RBFSampler` on standardized observations). Report the best average episode length.

**E3.** Tile-coded Q-learning on MountainCar (08, section 1.4): compare with your best single grid from the project.

**E4.** In 5 lines: why can't tabular Q-learning handle a camera image as state?

---

## 5. Self-check (closed book)

1. What do n = 1 and n = ∞ correspond to?
2. What is the deadly triad?
3. Why "semi-gradient"?
4. For a linear Q̂, what is the gradient?

---

## Answers

- **E4:** an 84×84 grayscale image has 256^7056 possible values; you'll never see the same one twice, so a table can't generalize. You need a function that maps similar images to similar values.
- **Self-check 1:** TD(0) and Monte Carlo.
- **Self-check 2:** function approximation + bootstrapping + off-policy learning, which together can diverge.
- **Self-check 3:** the target depends on w too, but it's treated as a constant; only the prediction is differentiated.
- **Self-check 4:** the feature vector φ(s).

**Next:** [11 — Deep Q-Networks](11-dqn.md)
