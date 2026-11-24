# 07 — Exam question bank (*banque de questions*)

> **Goal:** revise everything in Part 1 closed book, in the language of the exam.
> **You need:** 01-06. **Time:** about 1.5 h per pass. Do it twice: late December, then the week of 4 Jan.
> **How:** answer on paper without notes. Then correct with the answers at the bottom and reread the module named next to each answer you missed.

This is a reference page: no code or worked example.

---

## Concepts
1. Définir agent, environnement, état, action, récompense, politique.
2. Différence entre récompense et valeur.
3. Différence entre apprentissage supervisé, non supervisé et par renforcement.
4. Qu'est-ce que la propriété de Markov ? Donner un exemple et un contre-exemple.
5. Donner les 5 éléments d'un MDP.
6. Politique déterministe vs stochastique.
7. Que se passe-t-il si γ = 0 ? Si γ = 1 ?
8. Exploration vs exploitation, et la stratégie ε-greedy.

## Bellman and DP
9. Écrire l'équation de Bellman pour V^π.
10. Écrire l'équation d'optimalité de Bellman pour Q*.
11. Itération de politique vs itération de valeur.
12. Limites de la programmation dynamique.

## Model-free
13. Pourquoi des méthodes sans modèle ?
14. Monte Carlo first-visit vs every-visit.
15. MC vs TD : quand met-on à jour ? Biais ? Variance ?
16. Que signifie SARSA ? Écrire sa règle de mise à jour.
17. Écrire la règle du Q-learning. Pourquoi est-il off-policy ?
18. Qu'est-ce que l'erreur TD ?
19. terminated vs truncated : que change-t-on dans la cible ?

## Evaluation
20. Pourquoi évaluer avec ε = 0 ?
21. Pourquoi plusieurs graines ?
22. Pourquoi une baseline aléatoire ?
23. Une courbe monte puis oscille : cause et solution ?
24. Limite du Q-learning tabulaire et solution.

## Calculation (exam style)
25. Q(s,a) = 0,5, r = 0, max Q(s',·) = 0,8, α = 0,1, γ = 0,9. Nouvelle valeur ?
26. Récompenses 0, 0, 1, γ = 0,9. Retour G_0 ?
27. Gridworld 2×2 du cours, politique uniforme, γ = 0,9 : V(S1) après convergence ?
28. π choisit Droite avec 0,7 et Bas avec 0,3 ; Q(s, Droite) = 0,6, Q(s, Bas) = 0,2. V^π(s) ?
29. Q(s,a) = 0,4, r = −1, Q(s',·) = [0,2 ; 0,9 ; 0,5], a' = 0, α = 0,5, γ = 1. Nouvelle valeur pour SARSA et pour Q-learning ?
30. Taux de succès 0,74 mesuré sur 100 épisodes. Erreur type ? Et sur 1 000 ?

---

## Answers (with the module to reread)

1-6. See [01](01-mdp-bellman.md), section 1, and the docx table.
7. γ = 0 → only the immediate reward (*myope*). γ = 1 → all future rewards count equally; needs episodic tasks or the sum may diverge. ([01](01-mdp-bellman.md) 1.3)
8. Explore = try other actions; exploit = take the best known; ε-greedy: random with probability ε, best otherwise, ε decreasing over time. ([04](04-q-learning.md) 1.3)
9-10. [01](01-mdp-bellman.md) 1.5.
11. PI alternates full evaluation and improvement; VI does one max-backup per sweep and extracts the policy at the end. ([02](02-dynamic-programming.md))
12. Needs the model, expensive for big state spaces, many iterations. ([02](02-dynamic-programming.md) 1.5)
13. The model is usually unknown; learn from (s, a, r, s'). ([03](03-monte-carlo.md) 1.1)
14. First-visit counts only the first occurrence per episode (unbiased); every-visit counts all. ([03](03-monte-carlo.md) 1.2)
15. MC: at episode end, unbiased, high variance. TD: every step, biased (bootstrapping), lower variance. ([05](05-td-control.md) 1.2)
16. State, Action, Reward, State, Action. Q(S,A) ← Q(S,A) + α[R + γQ(S',A') − Q(S,A)]. ([05](05-td-control.md) 1.3)
17. Q(s,a) ← Q(s,a) + α[r + γ max Q(s',·) − Q(s,a)]. Off-policy: learns the greedy policy (max) while behaving ε-greedy. ([05](05-td-control.md) 1.3)
18. δ = target − current estimate. ([04](04-q-learning.md) 1.4)
19. terminated → target = r. truncated → keep r + γ max Q(s'). ([04](04-q-learning.md) 1.4)
20. Random actions lower the score and hide what the agent knows. ([06](06-evaluation.md))
21. RL is random; one run can be lucky or unlucky; measure the variance. ([06](06-evaluation.md))
22. A score only means something compared to a reference. ([06](06-evaluation.md))
23. α too large → lower α. ([06](06-evaluation.md) 1.5)
24. One row per state is impossible for huge or continuous spaces → discretization ([08](08-discretization-tile-coding.md)), function approximation, DQN ([10](10-beyond-tables.md), [11](11-dqn.md)).
25. 0,522. ([04](04-q-learning.md) worked example B)
26. 0,9² × 1 = 0,81.
27. 0,5. ([02](02-dynamic-programming.md) E1)
28. 0,48. ([01](01-mdp-bellman.md) E3)
29. SARSA −0,2 ; Q-learning 0,15. ([05](05-td-control.md) worked example)
30. ≈ 0,044 ; ≈ 0,014. ([06](06-evaluation.md) worked example)

**Next:** Part 2, [08 — Discretization and tile coding](08-discretization-tile-coding.md), or Part 3 after the exams: [10 — Beyond tables](10-beyond-tables.md).
