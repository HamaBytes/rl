# A5 — Evaluating an agent + exam question bank

Class: docx part 3. This module turns "my code runs" into "I can prove my agent learned", and ends with the question bank to revise for 7-16 Jan.

## 1. Golden rules of evaluation

1. **Never judge on training scores.** Training uses ε > 0, so scores are lowered on purpose. Evaluate separately: greedy (ε = 0), **no Q update**.
2. **Compare to a baseline** (random policy). A score means nothing alone.
3. **Several seeds** (at least 5). Report mean ± standard deviation.
4. **Change one hyperparameter at a time.**
5. **Enough evaluation episodes.** Slippery FrozenLake: 1,000, because success is a noisy 0/1 per episode.

## 2. Metrics

| Métrique | What it says |
|---|---|
| Retour moyen | Average cumulative reward; should rise then plateau |
| Taux de succès | Share of episodes reaching the goal (when reward is 0/1) |
| Longueur d'épisode | Shorter = more efficient |
| Vitesse de convergence | Episodes needed to reach a good level |
| Variance entre graines | Stability; a result is only reliable if reproducible |

Not in the docx but useful: **how many evaluation episodes is enough?** For a success rate p over n episodes, the standard error is about √(p(1−p)/n). With p = 0.74 and n = 100, that's ±0.044, so a 74% vs 70% difference is noise. With n = 1,000 it's ±0.014.

## 3. Hyperparameter table (memorize)

| Paramètre | Trop petit | Trop grand |
|---|---|---|
| α | Very slow learning | Unstable values, oscillation |
| γ | Myopic, ignores the distant goal | Little incentive to shorten the path |
| ε | Freezes on a bad solution | Stays too random |
| Épisodes | Hasn't had time to learn | Wasted compute |

## 4. Diagnosis table (memorize)

| Symptôme | Cause probable | Piste |
|---|---|---|
| Flat curve at 0 | Sparse reward, ε decays too fast, too few episodes | Slower ε decay, more episodes |
| Rises then oscillates or collapses | α too large | Lower α |
| Good in training, bad in evaluation | Too few eval episodes, or ε still high | Greedy eval, more episodes |
| Very different results across seeds | High variance | Average over seeds |
| Path uselessly long | γ = 1 and no step penalty | Slightly lower γ or a small step penalty |

## 5. Exercise: the mini lab report (90 min)

Write a one-page report (French, your exam language) on slippery FrozenLake comparing DP (A2), MC (A3), SARSA and Q-learning (A4):

1. Table: method, episodes, success rate mean ± std over 5 seeds, 1,000 greedy eval episodes each.
2. One figure: learning curves (mean line + std band) for the three model-free methods.
3. Five sentences of analysis: which converges fastest, which is most stable, how far from the DP optimum, and why (use the words *biais*, *variance*, *bootstrapping*, *on-policy*, *off-policy*).

This report is the best exam preparation in this course, because it forces every concept into one argument.

```python
def run_experiment(train_fn, env, seeds=range(5), n_eval=1000, **kwargs):
    curves, scores = [], []
    for s in seeds:
        Q, returns = train_fn(env, seed=s, **kwargs)
        curves.append(moving_average(returns, ___))
        scores.append(evaluate(env, Q, n_episodes=___))   # evaluate returns success rate
    curves = np.array(curves)
    return curves.mean(0), curves.std(0), np.mean(scores), np.std(scores)
```

## 6. Question bank (closed book; answers at the bottom)

**Concepts**
1. Définir agent, environnement, état, action, récompense, politique.
2. Différence entre récompense et valeur.
3. Différence entre apprentissage supervisé, non supervisé et par renforcement.
4. Qu'est-ce que la propriété de Markov ? Donner un exemple.
5. Donner les 5 éléments d'un MDP.
6. Politique déterministe vs stochastique.
7. Que se passe-t-il si γ = 0 ? Si γ = 1 ?
8. Exploration vs exploitation, et la stratégie ε-greedy.

**Bellman and DP**
9. Écrire l'équation de Bellman pour V^π.
10. Écrire l'équation d'optimalité de Bellman pour Q*.
11. Itération de politique vs itération de valeur.
12. Limites de la programmation dynamique.

**Model-free**
13. Pourquoi des méthodes sans modèle ?
14. Monte Carlo first-visit vs every-visit.
15. MC vs TD : quand met-on à jour ? Biais ? Variance ?
16. Que signifie SARSA ? Écrire sa règle de mise à jour.
17. Écrire la règle du Q-learning. Pourquoi est-il off-policy ?
18. Qu'est-ce que l'erreur TD ?
19. terminated vs truncated : que change-t-on dans la cible ?

**Evaluation**
20. Pourquoi évaluer avec ε = 0 ?
21. Pourquoi plusieurs graines ?
22. Pourquoi une baseline aléatoire ?
23. Une courbe monte puis oscille : cause et solution ?
24. Limite du Q-learning tabulaire et solution.

**Calculation (exam style)**
25. Q(s,a)=0.5, r=0, max Q(s',·)=0.8, α=0.1, γ=0.9. Nouvelle valeur ?
26. Récompenses 0, 0, 1, γ=0.9. Retour G_0 ?
27. Class 2×2 Gridworld, uniform policy, γ=0.9: V(S1) after convergence?

---

### Answers

1-8: see A1 section 1 and the docx table. 7: γ=0 → only immediate reward (myope); γ=1 → all future rewards equal, needs episodic tasks. 8: explore = try other actions; exploit = take the best known; ε-greedy: random with probability ε, best otherwise, ε decreasing over time.
9-10: A1 section 4.
11: PI alternates full evaluation and improvement; VI does one max-backup per sweep, policy extracted at the end.
12: needs the model, expensive for big state spaces, many iterations.
13: the model is usually unknown; learn from (s, a, r, s').
14: first-visit counts only the first occurrence per episode (unbiased); every-visit counts all.
15: MC at episode end, unbiased, high variance; TD each step, biased (bootstrapping), lower variance.
16: State, Action, Reward, State, Action. Q(S,A) ← Q(S,A) + α[R + γQ(S',A') − Q(S,A)].
17: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) − Q(s,a)]. Off-policy: learns the greedy policy (max) while behaving ε-greedy.
18: δ = target − current estimate.
19: terminated → target = r. truncated → keep γ max Q(s').
20: random actions lower the score and hide what the agent knows.
21: RL is random; one run can be lucky or unlucky; measure variance.
22: a score only means something compared to a reference.
23: α too large → lower α.
24: one row per state is impossible for huge/continuous spaces → function approximation, DQN.
25: 0.522. 26: 0.81. 27: 0.5.
