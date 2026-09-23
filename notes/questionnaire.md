# Questionnaire de bilan — tout depuis le début

Rappel et vérification, de l'étape 0 à l'étape 5 plus la branche 8. À faire **sans relire
le code ni les notes** : ce que tu ne sais pas réexpliquer, tu ne le sais pas encore.

Il n'y a pas de réponses ici. Tu réponds, je corrige — en nommant ce qui est faux et
pourquoi, comme pour les rubriques du journal.

Les questions marquées 🔴 portent sur du code que tu as écrit. Celles marquées 🟢 sur
l'outillage. Celles marquées 📊 demandent d'interpréter un chiffre.

---

## 1. Fondations — étapes 0 à 2

1. Qu'est-ce qu'une distribution conditionnelle, et qu'est-ce que le bigramme par comptage estime exactement ?
2. Pourquoi la log-vraisemblance négative plutôt que le taux d'erreur ?
3. 📊 Le bigramme donne 2,45 de perte et la baseline uniforme `ln 27 ≈ 3,296`. Qu'est-ce que l'écart mesure, et pourquoi cette baseline précise ?
4. Qu'est-ce que la perplexité, et pourquoi la préfère-t-on parfois à la perte ?
5. Pourquoi faut-il un lissage dans le bigramme ? Que se passe-t-il sans lui ?
6. 🔴 Dans le MLP de l'étape 1, qu'est-ce qu'un embedding, et pourquoi n'est-ce pas un simple encodage un-parmi-N ?
7. 🔴 Écris la règle de la chaîne pour une composition de deux fonctions. Où intervient-elle dans ta backprop ?
8. 🔴 Pourquoi initialise-t-on les poids aléatoirement et pas à zéro ?
9. Que fait exactement `loss.backward()` en PyTorch, et qu'est-ce qu'il ne fait pas ?
10. Pourquoi l'étape 2 (même MLP en PyTorch) valait-elle la peine alors que l'étape 1 marchait déjà ?

## 2. Récurrence — étape 3

11. 🔴 Quelle est la différence de structure entre un MLP à fenêtre fixe et un RNN ?
12. Qu'est-ce que la rétropropagation à travers le temps, et pourquoi coûte-t-elle cher ?
13. Pourquoi le gradient explose ou s'évanouit dans un RNN ? Les deux ont-ils la même cause ?
14. Qu'est-ce qu'une porte dans un LSTM, et quel problème résout-elle ?
15. Qu'est-ce qu'un LSTM ne sait toujours pas faire, et qui a motivé l'attention ?

## 3. Attention et transformer — étape 4

16. 🔴 Que représentent `W_q`, `W_k`, `W_v` ? Pourquoi trois matrices et pas une ?
17. 🔴 Que calcule `Q @ K.T` ? Quelle est la forme du résultat, et que signifie la case (i, j) ?
18. 🔴 Pourquoi diviser par `sqrt(head_dim)` ? Que se passerait-il sans cette division ?
19. 🔴 À quoi sert le masque triangulaire ? Que se passerait-il sans lui, à l'entraînement puis à la génération ?
20. 🔴 Pourquoi `-inf` dans le masque plutôt que zéro ?
21. 🔴 Pourquoi plusieurs têtes ? Qu'apporte le découpage en `num_heads` que `dim` entier n'apporterait pas ?
22. 🔴 Que fait `W_o` après l'attention, et pourquoi est-elle nécessaire ?
23. 🔴 À quoi sert la connexion résiduelle `x = x + out` ?
24. 🔴 Que normalise `layernorm`, sur quel axe, et pourquoi `g` et `b` ?
25. 🔴 Explique la différence entre post-norm et pre-norm. Pourquoi six blocs en post-norm n'apprenaient-ils pas ?
26. 🔴 En pre-norm, pourquoi la normalisation finale avant `W_out` devient-elle obligatoire ?
27. 🔴 Pourquoi le MLP interne passe-t-il par `dim*4` avant de revenir à `dim` ?
28. 🔴 À quoi sert `pos_emb` ? Que produirait le modèle sans encodage de position ?
29. 📊 Le modèle post-norm plafonnait à 6,37 et l'entropie unigramme du corpus vaut 6,3149. Qu'est-ce que ça prouve exactement ?

## 4. Tokenisation BPE — étape 5

30. 🔴 Décris l'algorithme BPE en cinq lignes. Qu'est-ce qu'on compte, qu'est-ce qu'on fusionne, quand s'arrête-t-on ?
31. Pourquoi 2 080 tokens plutôt que 256 caractères ou 50 000 mots ? Quels sont les deux compromis ?
32. 🔴 À quoi sert la pré-tokenisation avant les fusions ?
33. Pourquoi aucune fusion ne peut-elle chevaucher une frontière de mot ? Quelle propriété du vocabulaire le garantit, et comment l'as-tu vérifiée ?
34. Pourquoi la mémoïsation par mot est-elle correcte ? Sur quelle propriété repose-t-elle ?
35. 📊 Un texte de 25 Mo contient 39 229 mots distincts pour 4,99 M d'occurrences. Quel facteur d'accélération la mémoïsation permet-elle en théorie, et pourquoi ?
36. 🔴 Que fait `tokkenisation(mot, bpe, bpeset)` ? Pourquoi un `set` en plus du dictionnaire ?
37. Pourquoi stocker les tokens en `uint16` ? Quelle est la limite, et que se passerait-il si le vocabulaire dépassait 65 536 entrées ?
38. Qu'est-ce qu'un générateur Python, et qu'est-ce que `yield` change par rapport à `return` ?
39. 📊 L'encodeur passe de 12,9 à 0,9 Mo de RAM par Mo de corpus. Quelles sont les trois transformations successives, et laquelle a le plus rapporté ?

## 5. Entraînement : boucle, optimiseur, schedule

40. 🔴 Décris le corps de ta boucle d'entraînement, dans l'ordre, sans code.
41. 🔴 Que fait `tireur_de_lot(tableau, B, T)` ? Que valent `B` et `T`, et que contient chaque élément du lot ?
42. 🔴 Le tirage se fait avec `random.randint`. Est-ce un mélange du corpus ? Quelle conséquence sur la notion d'« époque » ?
43. 🔴 Pourquoi `cross_entropy` attend-elle les classes sur un axe précis ? Qu'est-ce qui plantait quand tu t'es trompé d'axe ?
44. 🔴 Qu'est-ce qu'Adam garde en mémoire en plus des gradients ? À quoi servent `m` et `v` ?
45. 🔴 Pourquoi `m_hat` et `v_hat` plutôt que `m` et `v` directement ?
46. 🔴 À quoi sert `eps` au dénominateur ?
47. 🔴 Si Adam renormalise déjà chaque paramètre, à quoi sert encore l'écrêtage du gradient ?
48. 🔴 `clip_grad_norm_` écrête sur la norme **globale**. Qu'est-ce que ça change par rapport à un écrêtage coordonnée par coordonnée ?
49. 🔴 Décris les trois formes de `config_pas`. Que contiennent `amorce`, `fin`, et le quatrième élément ?
50. Pourquoi un échauffement ? Qu'est-ce qui se passe de mal dans les premiers pas sans lui ?
51. Pourquoi une décroissance en fin d'entraînement ?
52. Pourquoi `1/i²` et `1/ln i` ont-ils été écartés comme schedules ? Quelle condition ne respectent-ils pas ?
53. 🔴 Le checkpoint enregistre `config_pas` et `i`, pas la valeur courante du pas. Pourquoi est-ce mieux ?
54. 🔴 `.to(device)` sur un paramètre faisait planter Adam. Pourquoi ?

## 6. Lois d'échelle et budget

55. Qu'est-ce que la loi de Chinchilla dit exactement ? Que vaut le ratio, et de quoi est-il le ratio ?
56. Quelle est la différence entre un protocole iso-pas et iso-calcul ? Lequel répond à quelle question ?
57. 📊 La courbe en U du 08/09 donnait 1,6194 → 1,5228 → 1,6454. Qu'est-ce qui produit la branche gauche, et qu'est-ce qui produit la branche droite ?
58. 📊 Pourquoi cette courbe s'est-elle révélée en partie fausse ? Qu'est-ce qui était confondu avec le budget de tokens ?
59. Le pas maximal stable varie comme l'inverse de la largeur. Comment l'as-tu vérifié quantitativement le 22/09 ?
60. 📊 L'échauffement et l'écrêtage multiplient le seuil par huit sans changer sa dépendance à la largeur. Qu'est-ce que ça suggère sur ce qui cause l'instabilité ?
61. 📊 `A4` coûte ×1,76 le prix de `A3` sur le portable et ×1,30 sur la 3090. D'où vient la différence, et qu'est-ce que ça implique pour la courbe en U ?
62. Pourquoi augmenter la taille de lot n'a-t-il rapporté que 10 % ? Qu'est-ce que ça dit de l'occupation du GPU ?
63. Pourquoi changer la taille de lot obligerait-il à refaire les mesures de stabilité ?

## 7. Génération et échantillonnage

64. 🔴 Décris la boucle de `genere()`, de l'amorce au texte final.
65. 🔴 Que fait la température, mathématiquement, avant le softmax ?
66. 🔴 Explique top-k et top-p. Lequel garde un nombre fixe de candidats ?
67. 📊 Pourquoi top-p dégénère-t-il à basse température, jusqu'à 95,8 % de redite à `p=0,2` ?
68. 🔴 Le défaut de top-p prenait le token de la frontière au lieu d'échantillonner dans le noyau. Pourquoi était-il invisible à la lecture ?
69. 🔴 Pourquoi la génération ne peut-elle pas dépasser `max_len` tokens ? Quelle ligne plante, et pourquoi ?
70. Qu'est-ce qu'une fenêtre glissante à la génération, et que perd-on en l'utilisant ?
71. Pourquoi le coût de la génération croît-il en `T²` ? Qu'est-ce qui est recalculé inutilement ?

## 8. Mesure et méthode

72. Qu'est-ce qu'un plancher de bruit, et pourquoi aucune comparaison n'a de sens sans lui ?
73. 📊 Deux runs identiques diffèrent de 0,013. D'où vient cet écart, et qu'est-ce qui a été éliminé par la mesure ?
74. 📊 Dix écarts train/val tous positifs, chacun dans le bruit. Pourquoi peut-on conclure quand même ?
75. Qu'est-ce qu'une mesure appariée ? Où l'as-tu utilisée, et qu'est-ce qu'elle annule ?
76. Pourquoi une calibration de 60 pas ne permet-elle pas de comparer deux machines ?
77. 📊 Les trois métriques semblaient saturer sous 1,7. Pourquoi était-ce faux, et qu'est-ce qui l'a révélé ?
78. Pourquoi deux métriques opposées ne suffisent-elles pas si elles partagent un angle mort ? Donne l'exemple précis.
79. Qu'est-ce que le protocole du banc garantit en déposant une copie de `transformer.py` par run ?
80. Pourquoi le banc s'arrête-t-il si une substitution ne correspond pas exactement une fois ?

## 9. Branche 8 — bruit et quantification

81. Quelle est la différence entre bruit multiplicatif et bruit additif ? À quel matériel chacun correspond-il ?
82. 📊 Les deux courbes ont une pente de ~2,05 en log-log. Qu'est-ce que ça dit de la forme de la perte autour du minimum ?
83. Comment passe-t-on d'un niveau de bruit à un nombre de bits ? Écris le raisonnement.
84. 📊 `A1` tolère 3,8 % et `A2` 6,1 %. Le rapport des carrés vaut 2,6. Qu'est-ce que ce 2,6 mesure ?
85. 📊 `W_1` et `W_2` ont la même forme et le même nombre de poids, mais `W_2` est deux à trois fois plus fragile. Qu'est-ce que ça élimine comme explication ?
86. 📊 La précision mixte ne rapporte que 8 %. Pourquoi, alors que l'écart entre familles vaut 2,3 bits ?
87. Pourquoi dupliquer des cellules analogiques gagne-t-il en `√N` et coûte-t-il en `N` ?

## 10. Outillage — à quoi sert chaque script

88. 🟢 `scripts/tokenizer.py` — que produit-il, et dans quel ordre ?
89. 🟢 `scripts/echelle.py` — que fait-il exactement, et pourquoi ne modifie-t-il jamais `transformer.py` ?
90. 🟢 `scripts/sonde_gradient.py` — que mesure-t-il, et comment neutralise-t-il l'écrêtage ?
91. 🟢 `scripts/banc_bruit.py` — que balaie-t-il, et que signifie `--par-famille` ?
92. 🟢 `scripts/echelle_texte.py` — que compare-t-il, et pourquoi fixe-t-il les graines par amorce ?
93. 🟢 `scripts/programme.py` — à quoi sert-il par rapport à `echelle.py` ? Pourquoi existe-t-il ?
94. 🟢 `src/tooling/verifier_decoupage.py` et les autres vérificateurs — que garantissent-ils, et pourquoi sont-ils en zone verte alors qu'ils portent sur le tokenizer ?
95. 🟢 `src/model/Vocabulaire.py` — que produit-il et pour quelle mesure ?
96. 🟢 Pourquoi `runs/` et `data/` sont-ils dans `.gitignore` ?

## 11. Lire un résultat

97. 📊 `val 1,1433`, perplexité 3,14. Explique les deux chiffres à quelqu'un qui n'y connaît rien.
98. 📊 Train 1,0940 et val 1,1433 au pas 120 000. Qu'est-ce que l'écart signifie, et à partir de quand devient-il inquiétant ?
99. 📊 La validation descendait encore à l'arrêt. Pourquoi s'arrêter, alors ?
100. 📊 1 479 M de tokens tirés dans un corpus de 497 M. Combien d'époques, quelle couverture réelle, et pourquoi pas 100 % ?
101. 📊 `A4-grand` diverge à 6·10⁻³ avec 4,1992. Comment reconnaît-on une divergence d'un simple mauvais réglage ?
102. 📊 Le défaut d'attribution de parole disparaît entre 1,36 et 1,24. Qu'est-ce que ça dit de l'ordre dans lequel le modèle apprend ?

## 12. Pièges et erreurs

103. Pourquoi une liste Python ne se copie-t-elle pas avec `=` ? Quel bug ça a produit chez toi ?
104. Pourquoi ne faut-il pas modifier une liste pendant qu'on la parcourt ?
105. Pourquoi `sauvegarde_935` passait-il après `sauvegarde_1496` ? Qu'est-ce que ça a coûté ?
106. Pourquoi `CUDA_VISIBLE_DEVICES=3` ne désigne-t-il pas la carte 3 de `nvidia-smi` ?
107. Pourquoi les deux VM répondaient-elles « No route to host » alors qu'elles étaient allumées ?
108. Qu'est-ce qui rend un run reproductible, et qu'est-ce qui ne suffit pas ?

## 13. Vue d'ensemble

109. Raconte le chemin complet d'un caractère du corpus brut jusqu'à un token d'entrée du modèle.
110. Raconte le chemin complet d'un token de sortie jusqu'à un caractère affiché.
111. Qu'est-ce qui limite ton modèle aujourd'hui : la taille, le corpus, le contexte, ou le calcul ? Justifie.
112. Quelles sont les trois choses que tu referais autrement si tu recommençais le projet ?
