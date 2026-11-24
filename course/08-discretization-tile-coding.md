# 08 — Continuous states: discretization and tile coding

> **Goal:** turn MountainCar's continuous state into something a table can use, understand what that costs, and know the classic better answer (tile coding).
> **You need:** 00, 01 (section 1.6), 04. **Time:** about 2.5 h. **Class:** not in slides; core of your graded project.
> **Reading:** Sutton & Barto 9.5.3 *Coarse Coding*, 9.5.4 *Tile Coding*, then 10.1 *Episodic Semi-gradient Control* (where the book solves Mountain Car; example number *unverified*).

---

## 1. Concepts

### 1.1 The problem

MountainCar's observation is [position, velocity], with position ∈ [−1.2, 0.6] and velocity ∈ [−0.07, 0.07]. Infinitely many values, so you **can't** have one row per state.

### 1.2 Discretization (the paper's "tabular discretization")

Cut each dimension into a fixed number of **bins**, like squares on a chessboard:

```
velocity
  0.07 ┌───┬───┬───┬───┐
       ├───┼───┼───┼───┤
       │   │ ● │   │   │   ● = the car's (position, velocity)
       ├───┼───┼───┼───┤       falls in cell (column 1, row 2)
 -0.07 └───┴───┴───┴───┘
     -1.2              0.6   position
```

N bins for position and M for velocity give N × M discrete states, so the Q-table has shape **(N, M, 3)**.

For one dimension: find *what fraction of the range* the value is at, multiply by the number of bins, round down. Watch the top edge: the maximum value must give index N − 1, not N.

### 1.3 The trade-offs

| Fewer bins (coarse) | More bins (fine) |
|---|---|
| each cell visited often, learns fast | each cell visited rarely, learns slowly |
| big cells mix very different states (**aliasing**, non-Markov) | small cells are nearly Markov |
| policy may be too crude to swing up well | policy can be precise |
| **generalizes**: one update covers a big region | **no generalization** between neighbouring cells |

There is a **sweet spot**, and it depends on how many training episodes you allow. That's a testable, quantitative hypothesis (09).

**Non-Markov cells.** Two different (p, v) in the same cell can have different futures. TD bootstraps from cell values, so aliasing errors spread (05, E7). This is also why the tabular convergence guarantee (05, section 1.6) no longer strictly applies.

**Curse of dimensionality.** N bins per dimension and d dimensions give N^d cells. MountainCar has d = 2: fine. A robot arm with d = 12 at 20 bins has about 4·10¹⁵ cells: impossible. That's the honest reason tabular methods don't scale to robotics.

**Uniform vs non-uniform bins.** Velocity is tiny (±0.07) and often near 0; position spans 1.8. Equal bin counts for both may not be ideal. Check what the paper does.

### 1.4 Tile coding: the classic better answer

Instead of **one** grid, use **several** grids (*tilings*), each **shifted** by a fraction of a cell:

```
tiling 1:  |----|----|----|
tiling 2:    |----|----|----|        (shifted)
tiling 3:      |----|----|----|      (shifted more)
             ●
```

- A state activates **one tile per tiling**: with 8 tilings, 8 active tiles.
- Q(s,a) = **sum** of the weights of the active tiles for action a.
- Update: every active tile's weight moves toward the target, usually with step size α / (number of tilings).

Why it's better than one fine grid:
- **Generalization:** nearby states share some tiles, so learning in one spot helps its neighbours.
- **Resolution:** the overlapping shifted tilings separate states more finely than any single tiling.
- **Cost:** still a few lookups and additions.

Vocabulary shift: with tile coding Q is no longer a table but a **linear function** of binary features. The update becomes a **semi-gradient** update (10), and the tabular guarantee no longer applies as-is.

---

## 2. Worked example: one bin index

10 bins for position over [−1.2, 0.6].

- Bin width = (0.6 − (−1.2)) / 10 = **0.18**.
- Position −0.5: fraction = (−0.5 + 1.2) / 1.8 = 0.389; × 10 = 3.89; round down → bin **3**.
- Check: bin 3 covers [−1.2 + 3×0.18, −1.2 + 4×0.18) = [−0.66, −0.48). −0.5 is inside. ✓

---

## 3. Code: `discretize()` (Mission 2: write and test it alone, before any Q-learning)

```python
import numpy as np

LOW  = np.array([-1.2, -0.07])
HIGH = np.array([ 0.6,  0.07])

def discretize(obs, n_bins):
    """obs: array [position, velocity]; n_bins: array [N, M]. Returns a tuple (i, j)."""
    ratio = (obs - ___) / (___ - ___)          # fraction of the range, in [0, 1]
    idx = (ratio * ___).astype(int)            # round down
    idx = np.clip(idx, 0, ___ - 1)             # the top edge must not give N
    return tuple(idx)

# hand-checked tests (fill in the expected values from the exercises)
n = np.array([10, 20])
assert discretize(np.array([-0.5, 0.0]), n)[0] == ___
assert discretize(np.array([ 0.6, 0.0]), n)[0] == ___
assert discretize(np.array([-1.2, -0.07]), n) == (___, ___)

# Q-table: Q = np.zeros((N, M, 3)); read with Q[discretize(obs, n)]
```

Using `tuple(idx)` lets you index the 3D table directly: `Q[i, j]` gives the 3 action values.

---

## 4. Exercises

**E1 (by hand).** 10 position bins. Which bin does position 0.6 fall in? And −1.2? Velocity 0.01 with 20 velocity bins?

**E2.** In one sentence each: what happens with 2 bins per dimension? With 1,000?

**E3.** With 20 × 20 bins and 3 actions, how many Q-values are there? With 40 × 40? If an episode visits at most 200 cells, what is the minimum number of episodes before every cell could have been visited once? Why will the real number be much higher?

**E4.** Draw the 2D state space by hand and sketch where the car spends most of its time early in training. Which cells will stay empty?

**E5.** 8 tilings of 8 × 8 tiles: how many weights per action? Compare with one 64 × 64 grid. Which has more parameters, and which generalizes more?

**E6.** Explain in 3 sentences, to your professor, why tile coding is *function approximation* even though it's built from grids.

---

## 5. Self-check (closed book)

1. Why can't you use MountainCar's raw observation as a table row?
2. Write the bin-index formula for one dimension, including the edge case.
3. Coarse vs fine bins: two advantages of each.
4. Why are discretized states non-Markov, and why does that matter for TD?
5. Why does tile coding generalize better than one fine grid?

---

## Answers

- **E1:** 0.6 → 10 × 1.0 = 10, clipped to **9**. −1.2 → **0**. Velocity 0.01: width 0.007, (0.01 + 0.07)/0.007 ≈ 11.43 → **11**.
- **E2:** 2 bins: cells too coarse, aliasing so strong the agent can't tell when to switch direction. 1,000 bins: a million cells, almost never revisited, so nothing generalizes and learning crawls.
- **E3:** 400 cells → 1,200 Q-values; 1,600 cells → 4,800. Minimum 2 and 8 episodes. Far more in reality: visits concentrate near the start (E4) and repeat the same cells.
- **E4:** early on the car oscillates near the valley bottom (position around −0.5, small velocities); high-speed cells and the far right stay empty.
- **E5:** 8 × 64 = 512 per action vs 4,096. The single grid has more parameters; tile coding generalizes more (overlapping tiles).
- **Self-check 4:** one cell mixes states with different futures; TD copies the cell's value into its targets, so errors spread instead of averaging out.

**Next:** [09 — MountainCar project guide](09-mountaincar-project.md)
