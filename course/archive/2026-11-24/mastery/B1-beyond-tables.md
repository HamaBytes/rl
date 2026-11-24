# B1 — Beyond tables: n-step returns, eligibility traces, function approximation

Not in class. The bridge from tabular RL (Track A) to deep RL (B2+). Tile coding is already covered in [02-q-learning-deep.md](../02-q-learning-deep.md) section 8, so it isn't repeated here.

## 1. n-step returns: between TD(0) and Monte Carlo

G_(t:t+n) = r_(t+1) + γ r_(t+2) + … + γ^(n−1) r_(t+n) + γ^n V(s_(t+n))

- n = 1 → TD(0). n = ∞ (to the end) → Monte Carlo.
- Larger n: less bias, more variance. Middle values (n = 4 to 8) often learn fastest.

This same trade-off reappears as GAE's λ in PPO (B4). Understanding it here makes B4 much easier.

## 2. Eligibility traces and TD(λ) (know the idea)

Instead of choosing one n, average all n-step returns with weights (1−λ)λ^(n−1). Implemented efficiently with a trace e(s) that remembers recently visited states:

δ = r + γV(s') − V(s);  e(s) ← e(s) + 1;  for all s: V(s) ← V(s) + α δ e(s);  e ← γλ e

λ = 0 → TD(0); λ = 1 → close to MC.

## 3. Function approximation

When states are continuous or too many (camera images, joint angles), replace the table with a parametrized function: V̂(s; w) or Q̂(s, a; w).

**Semi-gradient TD update:**

w ← w + α [ r + γ Q̂(s', a'; w) − Q̂(s, a; w) ] ∇_w Q̂(s, a; w)

"Semi" because we don't differentiate through the target.

**Linear case:** Q̂(s,a;w) = w_a · φ(s), with features φ (tile coding, polynomials, RBFs). Then ∇ = φ(s).

**The deadly triad** (Sutton & Barto ch. 11): **function approximation + bootstrapping + off-policy** together can diverge. Q-learning with a neural network has all three. DQN's tricks (B2) exist to tame exactly this.

## 4. Skeleton: linear semi-gradient SARSA on CartPole

```python
import numpy as np, gymnasium as gym

env = gym.make('CartPole-v1')
obs_dim, nA = env.observation_space.shape[0], env.action_space.n

def features(obs):
    # Simple start: raw obs + bias + squares. Later: RBF or tile coding.
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

## 5. Exercises

**E1.** Implement n-step SARSA on CliffWalking for n = 1, 2, 4, 8. Plot the reward curve for each. Which n learns fastest?

**E2.** Linear SARSA on CartPole with the features above. It will likely plateau low. Then try RBF features (`sklearn.kernel_approximation.RBFSampler` on standardized observations). Report the max average episode length.

**E3.** Write in your own words (5 lines) why tabular Q-learning cannot handle a camera image as state.

## 6. Self-check

1. What do n = 1 and n = ∞ correspond to?
2. What is the deadly triad?
3. Why "semi-gradient"?

---

### Answers

- Self-check 1: TD(0) and Monte Carlo.
- Self-check 2: function approximation + bootstrapping + off-policy learning, which together can diverge.
- Self-check 3: the target also depends on w, but we treat it as a constant and only differentiate the prediction.
- E3 key point: a 84×84 grayscale image has 256^(7056) possible states; you'll never visit the same one twice, so a table can't generalize. You need a function that maps similar images to similar values.
