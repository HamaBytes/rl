# TP Apprentissage par renforcement — fiche de préparation (mer. 25 nov, 09:00)

Read time: about 25 minutes. Stop at 22:35 even if you didn't finish. Sections 1, 2 and 5 matter most.

## How I split part 1 and part 2 (my reading)

- **Part 1 (done last class) = the `MDPBellman` deck**, which is exactly slides 1-31 of the big deck: intro, MDP, Markov property, politique, Bellman, programmation dynamique (évaluation / amélioration / itération de politique, itération de valeur).
- **Part 2 (tomorrow) = slides 32-59 of `Reinforcement_Learning_FR`**: méthodes sans modèle, Monte Carlo, TD, **SARSA**, **Q-Learning**, then slide 59 "TP Apprentissage par renforcement" (a Colab link).
- **The docx** is a complete FrozenLake Q-learning TP (Parties 1-3 + "Travaux pratiques guidés"). GUESS: this is the lab script or very close to it.
- **The Gemini .py** is only one line: `!pip install -q python-chess pygame pymunk imageio[ffmpeg]`. GUESS: the TP (or a bonus) uses a game/physics environment (chess, a pymunk physics sim) and saves a video/GIF with imageio. I could not open the Colab links (blocked from here), so I can't confirm.
- I could not read slides 53-57 (Q-learning): they are images. The docx covers the same content.

---

## 1. Rappel partie 1 — 7 idées clés

1. **Boucle agent-environnement**: l'agent observe s_t, choisit a_t, reçoit r_(t+1) et s_(t+1). Objectif: maximiser la **récompense cumulée**, pas la récompense immédiate.
2. **MDP = (S, A, P, R, γ)**. P(s'|s,a) = dynamique, R = récompense, γ ∈ [0,1] = facteur d'actualisation.
3. **Propriété de Markov**: le futur dépend seulement de l'état et de l'action actuels, pas de l'historique.
4. **Politique π(a|s)**: déterministe (une action par état) ou stochastique (une probabilité par action).
5. **Retour**: G_t = r_(t+1) + γ r_(t+2) + γ² r_(t+3) + … Exemple: 0, 0, 1 avec γ=0,9 → G = 0,81.
6. **Bellman**:
   - V^π(s) = Σ_a π(a|s) Σ_s' P(s'|s,a) [ R(s,a,s') + γ V^π(s') ]
   - Q*(s,a) = R(s,a) + γ Σ_s' P(s'|s,a) max_a' Q*(s',a')
   - En clair: valeur = récompense immédiate + valeur actualisée de la suite.
7. **Programmation dynamique (DP)**: évaluation de politique (itérer Bellman jusqu'à convergence), amélioration (π'(s) = argmax_a …), **itération de politique** = évaluer + améliorer en boucle, **itération de valeur** = V_(k+1)(s) = max_a Σ P [R + γ V_k(s')]. **Limite principale: il faut connaître P et R** (le modèle). C'est exactement ce qui motive la partie 2.

Reward vs value: la récompense dit ce qui est bon maintenant, la valeur ce qui est bon à long terme.

---

## 2. Ce que le TP va probablement demander

GUESS, based on the docx's "Travaux pratiques guidés" list and slide 2 ("Implémenter SARSA… Gridworld ou OpenAI Gym"):

1. **Prise en main**: FrozenLake-v1 `is_slippery=False`, table Q, politique affichée en flèches, vérifier à la main.
2. **Courbe d'apprentissage**: moyenne glissante (fenêtre 100), dire à quel épisode l'agent devient bon.
3. **Hyperparamètres**: 3 valeurs de α, 3 valeurs de γ, tableau moyenne ± écart type sur 5 graines.
4. **Baseline**: comparer à un agent aléatoire.
5. **Stochastique**: `is_slippery=True`, expliquer pourquoi c'est plus dur (≈70-80 % max, 20 000+ épisodes, α plus petit).
6. Possible: **SARSA vs Q-learning** (the slides teach SARSA in detail, the docx teaches Q-learning).
7. Bonus: Taxi-v3 (500 états, 6 actions).

### Concepts and formulas you need

| Concept | Formule / idée |
|---|---|
| Sans modèle (model-free) | On ne connaît pas P ni R, on apprend à partir de tuples (s, a, r, s') |
| Monte Carlo | Attend la **fin de l'épisode**, V(s) ≈ moyenne des G observés. First-visit (non biaisé) vs every-visit (plus de données) |
| TD | Met à jour **à chaque pas**. TD = **échantillonnage** (comme MC) + **bootstrapping** (comme DP) |
| SARSA (on-policy) | Q(S,A) ← Q(S,A) + α [ R + γ Q(S',A') − Q(S,A) ], A' = action **réellement choisie** ensuite (ε-greedy) |
| Q-learning (off-policy) | Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ] |
| Erreur TD δ | δ = cible − Q(s,a) |
| ε-greedy | proba ε: action au hasard (exploration), sinon argmax Q (exploitation). ε décroît **par épisode** |
| terminated vs truncated | terminated: plus de futur → cible = r. truncated (limite de temps): on garde γ max Q(s') |

The only code difference between SARSA and Q-learning is the target: `Q[s2, a2]` vs `max(Q[s2])`. Know that cold.

Worked example (from the docx): Q=0,5, r=0, max Q(s')=0,8, α=0,1, γ=0,9 → cible 0,72, erreur 0,22, nouvelle valeur **0,522**.

Your own course covers the same Q-learning core in more depth: [04-q-learning.md](04-q-learning.md). Your graded MountainCar project is a separate thing; tomorrow is a TP.

---

## 3. Code patterns (skeletons, you fill the `___`)

### 3.1 Setup
```python
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', is_slippery=___)
n_states  = env.observation_space.___
n_actions = env.action_space.___
Q = np.zeros((___, ___))
rng = np.random.default_rng(0)
```

### 3.2 ε-greedy with random tie-break
```python
def choose_action(Q, state, epsilon, rng):
    if rng.random() < ___:
        return int(rng.integers(___))
    best = np.flatnonzero(Q[state] == ___)   # all actions tied for best
    return int(rng.choice(best))
```
Why the tie-break: at the start all Q = 0, and plain `argmax` always returns action 0.

### 3.3 Q-learning loop
```python
for episode in range(n_episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action = choose_action(Q, state, epsilon, rng)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = ___ or ___
        target = reward + gamma * ___ * (not ___)
        Q[state, action] += alpha * (___ - Q[state, action])
        state = ___
    epsilon = max(eps_min, epsilon * ___)
```

### 3.4 SARSA loop (the part that differs)
```python
state, _ = env.reset()
action = choose_action(Q, state, epsilon, rng)       # choose A BEFORE the loop
while not done:
    next_state, reward, terminated, truncated, _ = env.step(action)
    next_action = choose_action(Q, ___, epsilon, rng)
    target = reward + gamma * Q[___, ___] * (not terminated)
    Q[state, action] += alpha * (target - Q[state, action])
    state, action = ___, ___
```

### 3.5 Greedy evaluation vs random baseline
```python
def evaluate(env, Q, n_episodes=100, random_policy=False):
    successes = 0
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample() if random_policy else int(np.argmax(___))
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        successes += ___
    return successes / n_episodes
```
Rule: evaluate with ε = 0 and **no Q update**.

### 3.6 Learning curve and seeds
```python
def moving_average(x, window=100):
    return np.convolve(x, np.ones(window) / window, mode=___)

# wrap training in train(env, alpha, gamma, seed) -> (Q, returns)
# run seeds 0..4, plot mean, fill_between(mean - std, mean + std)
```

### 3.7 Policy as arrows
```python
arrows = np.array(['←', '↓', '→', '↑'])    # FrozenLake: 0 left, 1 down, 2 right, 3 up
print(arrows[np.argmax(Q, axis=___)].reshape(4, 4))
```

### 3.8 If they ask for Monte Carlo first-visit (less likely)
```python
# after the episode ends, with episode = [(s0, r1), (s1, r2), ...]
G = 0
for t in reversed(range(len(episode))):
    s, r = episode[t]
    G = ___ * G + r
    if s not in [x[0] for x in episode[:t]]:   # first visit
        returns[s].append(G)
        V[s] = np.mean(returns[s])
```

---

## 4. Questions the teacher may ask (answers at the bottom, try first)

1. Différence entre DP et méthodes sans modèle ?
2. Différence entre Monte Carlo et TD ?
3. Pourquoi SARSA est on-policy et Q-learning off-policy ?
4. Que vaut la cible quand l'épisode est `terminated` ? Et `truncated` ?
5. Pourquoi évaluer avec ε = 0 et pas pendant l'entraînement ?
6. Effet de γ = 0 et de γ = 1 ? Effet d'un α trop grand ?
7. Pourquoi l'agent plafonne à ~75 % en `is_slippery=True` alors qu'il atteint 100 % sans glissade ?
8. Que faire quand le nombre d'états explose (images, positions continues) ?

---

## 5. Setup to check tonight (5 minutes max)

- **Colab is the likely platform** (slides 10 and 59 link Colab notebooks). Open the slide-59 link tonight while logged in to Google so you're not stuck on login at 09:00. In Colab, `!pip install …` lines work as-is.
- **Local fallback** (laptop, Pop!_OS), in a fresh env:
  ```bash
  uv venv && source .venv/bin/activate
  uv pip install gymnasium numpy matplotlib pygame imageio[ffmpeg] python-chess pymunk
  python -c "import gymnasium as gym; e=gym.make('FrozenLake-v1'); print(e.observation_space, e.action_space)"
  ```
  Expected: `Discrete(16) Discrete(4)`. Locally, drop the `!` from `!pip`.
- Package vs import names: `python-chess` imports as `chess`; `imageio[ffmpeg]` is needed to write MP4. `pygame` is needed for `render_mode="human"` on FrozenLake.
- Charge the laptop. Have this sheet and the docx open.

---

## Answers (hidden at the end)

1. DP needs the model (P and R) and computes values by sweeping all states. Model-free methods (MC, TD) learn only from sampled experience (s, a, r, s').
2. MC waits for the end of the episode and averages real returns G (no bootstrapping, high variance, episodic tasks only). TD updates after every step using its own estimate of the next state (bootstrapping, lower variance, works on long or endless tasks).
3. SARSA's target uses Q(S', A'), where A' is the action the ε-greedy policy actually picks, so it learns the value of the policy it follows. Q-learning uses max Q(s', ·), so it learns the greedy (optimal) policy while behaving ε-greedy.
4. Terminated: target = r (no future). Truncated: the state isn't really terminal, so keep r + γ max Q(s').
5. Random actions during training lower the score and hide what the agent actually knows. Evaluate greedy, separately, with no updates, against a random baseline, over several seeds.
6. γ = 0: myopic, only immediate reward. γ = 1: no pressure to take the short path (and sums can diverge on endless tasks). α too large: values oscillate and don't settle.
7. The ice is slippery: an action doesn't always go where intended, so even the optimal policy sometimes falls in a hole. Needs more episodes (20 000+), often a smaller α, and evaluation over ~1000 episodes.
8. Replace the table with a function approximator (a neural network estimating Q): Deep Q-Network (DQN).

Bonus trap to know: slide 17 sets V(Goal) = 1 while slide 19 sets Q(Goal, ·) = 0. It's a convention difference (is the +1 counted on arriving at the goal, or as the goal's value?). Same final logic; if asked, say the terminal state has value 0 and the +1 is the reward of the transition into it.
