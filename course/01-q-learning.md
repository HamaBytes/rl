# Course 01: Q-learning

*Written 23 Nov 2026. Main reference: Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed.), section 6.5. It's free on the authors' site.*

**Goal of this lesson:** understand Q-learning well enough to explain every line of `q_learning.py` before you write it.

**Before you start:** finish `random_agent.py` (Mission 1, Part B). This course builds on that loop.

---

## 1. The intuition

Your random agent always gets -200 because it never learns. Q-learning adds one thing: **a memory of how good each action was in each situation.**

The agent asks one question again and again:

> "If I'm in state **s** and take action **a**, how much total reward will I collect from now until the end, if I play well afterwards?"

That number is **Q(s, a)**, the *quality* of action a in state s.

If you knew Q perfectly, playing well would be easy: in each state, pick the action with the highest Q. Q-learning is a method for **estimating Q from experience**, one step at a time.

---

## 2. The Q-table

When states and actions are countable, you store Q in a table:

| state | action 0 (push left) | action 1 (no push) | action 2 (push right) |
|---|---|---|---|
| s₀ | -12.3 | -14.1 | -9.8 |
| s₁ | -7.0 | -8.2 | -10.5 |
| ... | ... | ... | ... |

- One **row** per state, and one **column** per action.
- Reading the table: in state s₀ the best action is **push right**, because -9.8 is the highest.
- At the start, the table holds a guess (usually all zeros). Learning means fixing those guesses.

**Exercise 2.1:** in state s₁ above, which action would a greedy agent pick? Why are all the values negative in MountainCar?

---

## 3. The problem: MountainCar's state is continuous

The observation is `[position, velocity]`, with position in [-1.2, 0.6] and velocity in [-0.07, 0.07]. There are infinitely many possible values, so you **can't** have one row per state.

**Solution: discretization.** Cut each dimension into a fixed number of **bins**, like the squares on a chessboard:

```
velocity
  0.07 ┌───┬───┬───┬───┐
       │   │   │   │   │
       ├───┼───┼───┼───┤
       │   │ ● │   │   │   ● = the car's current (position, velocity)
       ├───┼───┼───┼───┤        falls in cell (column 1, row 2)
       │   │   │   │   │
 -0.07 └───┴───┴───┴───┘
     -1.2              0.6   position
```

With **N bins** for position and **M bins** for velocity, you get N × M discrete states. So the Q-table has shape **(N, M, 3)**.

The idea for one dimension: find *what fraction of the range* the value is at, multiply by the number of bins, and round down. Be careful with the value exactly at the top edge, because it must not produce index N.

**Exercise 3.1 (by hand):** with 10 bins for position over [-1.2, 0.6]:
- How wide is each bin?
- Which bin index (0–9) does position -0.5 fall in?
- Which bin does position 0.6 fall in? (Careful: it's the edge case.)

**Exercise 3.2:** what happens with 2 bins per dimension? With 1000 bins? Write one sentence each. (This becomes your **H1** later.)

---

## 4. Choosing actions: ε-greedy

If the agent always picks the best-known action, it never tries anything new, so it can get stuck with a bad early guess. If it always acts randomly, it never uses what it learned.

**ε-greedy** balances both:
- with probability **ε**, pick a **random** action (explore)
- otherwise, pick the action with the **highest Q** in this state (exploit)

Usually ε starts high (e.g. 1.0) and **decays** over episodes (e.g. down to 0.01), so the agent explores early and exploits later.

**Exercise 4.1:** with ε = 0.1 and 3 actions, what's the probability the agent picks the greedy action? (Hint: a random choice can also land on the greedy action.)

---

## 5. The update rule: the heart of Q-learning

After every step `(s, a) → r, s'`, update one cell of the table:

```
Q(s, a)  ←  Q(s, a)  +  α · [ r + γ · max_a' Q(s', a')  −  Q(s, a) ]
                              └──────── target ────────┘
                             └──────────── TD error ─────────────┘
```

In words:
1. **Target** = the reward I just got, plus the discounted value of the best action from where I landed.
2. **TD error** = target − my old guess, i.e. how wrong I was.
3. Move my old guess a fraction **α** of the way toward the target.

### Worked example

α = 0.1, γ = 0.99. Old value Q(s, a) = -5. You take the step and get r = -1. In the new state s', the best Q is max Q(s', ·) = -4.

- target = -1 + 0.99 × (-4) = -1 - 3.96 = **-4.96**
- TD error = -4.96 - (-5) = **0.04**
- new Q(s, a) = -5 + 0.1 × 0.04 = **-4.996**

Q went up slightly: this action was a bit better than the agent thought.

### Special case: the episode really ended

If `terminated` is True (the car reached the flag), there is **no future**, so the target is just `r`, with no `γ · max Q(s')` term.

Careful: if `truncated` is True (the 200-step limit), the episode was **cut off**, but the car could have kept going. Many implementations treat both the same way. Note in `notes.md` which choice you make and why. Check what the paper does, because reproducing it means making the same choice.

**Exercise 5.1:** α = 0.5, Q(s, a) = -10, r = -1, and the car just reached the flag (`terminated=True`). What is the new Q(s, a)?

**Exercise 5.2:** same as the worked example, but α = 1.0. What is the new Q? What does α = 1 mean in words? What about α = 0?

---

## 6. The three knobs: α, γ, ε

| Symbol | Name | Range | What it controls | Too low | Too high |
|---|---|---|---|---|---|
| α | learning rate | 0–1 | how fast old guesses change | learns very slowly | unstable, forgets |
| γ | discount factor | 0–1 | how much the future counts | short-sighted | (close to 1 is normal for MountainCar) |
| ε | exploration rate | 0–1 | how often to act randomly | stuck on bad habits | never uses what it learned |

**Why γ matters a lot in MountainCar:** the only way to stop collecting -1 is to reach the flag, which can be ~100+ steps away. With small γ, the agent "doesn't see" that far. (This becomes your **H3**.)

**A hidden trick: initialization.** Every reward in MountainCar is -1, so the true Q-values are all negative. If you start the table at **0**, every untried action looks *better than it really is*, which pushes the agent to try it. This is called **optimistic initialization**, and it helps exploration for free.

**Exercise 6.1:** why might the agent learn faster with Q initialized to 0 than with Q initialized to -100? One or two sentences.

---

## 7. The full algorithm (pseudocode, not Python)

```
create Q-table of shape (N_pos_bins, N_vel_bins, 3), filled with 0
for each episode:
    s = discretize(reset())
    repeat until terminated or truncated:
        choose a from s using ε-greedy
        take a → observe r, s', terminated, truncated
        s' = discretize(s')
        update Q(s, a) with the rule from section 5
        s = s'
    decay ε
    record total reward of this episode
```

Your job in `q_learning.py` is to turn each line into Python yourself. Each line maps to something you've already used: `reset`, `step`, `action_space`, and numpy arrays.

---

## 8. How you'll know it works

- **Learning curve:** plot total reward per episode. Random gives a flat line at -200. A learning agent should start at -200 and, after some hundreds or thousands of episodes, **rise** (toward roughly -150, then higher).
- Curves are **noisy**. Plot a moving average (e.g. over 100 episodes) to see the trend.
- The paper's best score is **-108** (Table II). Don't expect that on your first run.

---

## 9. Self-check (answer in `notes.md` without looking)

1. What does Q(s, a) mean, in one sentence?
2. Why can't you use MountainCar's raw observation as a table row?
3. Write the Q-learning update rule from memory and name each part.
4. What changes in the update when `terminated` is True?
5. What do α, γ and ε each control?
6. Why is initializing Q to 0 "optimistic" in MountainCar?
7. Why does ε usually decay over time?

**Next:** Mission 2, discretization: write and test `discretize()` on its own, before any Q-learning.
