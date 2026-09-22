# Bibliographie annotée

Ordre de lecture aligné sur la feuille de route. Tenu à jour par Claude.

Colonne « quand » : le moment où la lecture est utile. Plusieurs entrées sont marquées
**après**, volontairement — lire l'implémentation d'un tiers avant d'avoir écrit la
sienne remplace la compréhension par de la reconnaissance de forme.

| # | Référence | Étape | Quand |
|---|---|---|---|
| 1 | Karpathy — *makemore* (série vidéo) | 0 → 4 | pendant, comme cours |
| 2 | Karpathy — *micrograd* | 1 | **après** avoir écrit le backward |
| 3 | Bengio et al. 2003 | 1 | avant |
| 4 | Elman 1990 | 3 | avant |
| 5 | Hochreiter & Schmidhuber 1997 | 3 | pendant |
| 6 | Vaswani et al. 2017 | 4 | avant |
| 7 | Karpathy — *nanoGPT* | 4 | **après** avoir écrit son transformer |
| 8 | Sennrich et al. 2016 | 5 | avant |
| 9 | Eldan & Li 2023 (*TinyStories*) | 5 | avant, pour dimensionner |
| 10 | Hoffmann et al. 2022 (*Chinchilla*) | 5 | avant, pour le budget de tokens |
| 11 | Holtzman et al. 2020 (*Degeneration*) | 4 → 5 | avant le banc d'essai |
| 12 | Gu & Dao 2023 (*Mamba*) | après 5 | avant d'écrire un SSM |
| 13 | Hasani et al. 2021 (*Liquid Time-constant*) | après 5 | pour savoir ce qu'on écarte |
| 14 | Wei et al. 2022 (*Chain-of-Thought*) | après 5 | avant, pour cadrer la question |
| 15 | Lee et al. 2023 (*Teaching Arithmetic*) | après 5 | avant, c'est lui qui rend l'essai faisable |
| 16 | Hao et al. 2024 (*Coconut*) | après 5 | avant d'écrire le rebouclage latent |

---

## 1. Karpathy — *makemore* / *Neural Networks: Zero to Hero*

Série vidéo, 2022–2023. Cours : `karpathy.ai/zero-to-hero.html`.
Code : `github.com/karpathy/makemore` (MIT).

| Ép. | Titre | Durée | Étape |
|---|---|---|---|
| 1 | building micrograd | 2h25 | 1 — voir #2 |
| 2 | building makemore (bigrammes) | 1h57 | **0** |
| 3 | makemore Part 2: MLP | 1h15 | 1–2 |
| 4 | makemore Part 3: Activations & Gradients, BatchNorm | 1h55 | 1–2 |
| 5 | makemore Part 4: Becoming a Backprop Ninja | 1h55 | 1 |
| 6 | makemore Part 5: Building a WaveNet | 56 min | hors feuille de route |
| 7 | Let's build GPT: from scratch, spelled out | 1h56 | 4 |
| 8 | Let's build the GPT Tokenizer | 2h13 | 5 |

**Référence centrale du projet.** La progression des vidéos suit presque exactement les
étapes 0 à 4, sur la même tâche (générer des prénoms au niveau caractère).

**Usage.** Comme cours : regarder, mettre en pause, écrire soi-même. Jamais comme source
de copier-coller — le dépôt contient l'intégralité de ce que tu dois produire, ce qui en
fait la façon la plus rapide de rater le projet. L'épisode 5 (backward manuel) est le
plus exigeant et le plus rentable.

**Ne pas commencer par l'épisode 1** : voir [demarrage.md](demarrage.md) §2.

---

## 2. Karpathy — *micrograd*

`github.com/karpathy/micrograd`. Vidéo associée : *The spelled-out intro to neural
networks and backpropagation: building micrograd*.

Moteur d'autograd scalaire en une centaine de lignes, plus un petit MLP au-dessus.
Chaque nombre porte sa valeur, son gradient et le lien vers les opérations qui l'ont
produit ; la rétropropagation est un parcours du graphe en ordre topologique inverse.

**Intérêt.** Il montre que l'autograd n'est pas magique : c'est la règle de la chaîne
appliquée mécaniquement sur un graphe. Après l'étape 1 (backprop écrite à la main sur
des matrices), la lecture prend dix minutes et remet tout en place.

**À lire après l'étape 1, pas avant.** Lu avant, il donne la réponse au principal
exercice du projet.

---

## 3. Bengio, Ducharme, Vincent & Jauvin (2003) — *A Neural Probabilistic Language Model*

*Journal of Machine Learning Research*, 3:1137–1155.

**L'article de l'étape 1.** Premier modèle de langue neuronal qui bat les n-grammes.
Deux idées, toujours vraies aujourd'hui :

1. **La malédiction de la dimension.** Un modèle par comptage sur un contexte de n mots
   doit estimer |V|^n probabilités ; il n'a jamais assez de données et attribue zéro à
   toute séquence non vue. D'où le lissage, qui est un pansement.
2. **L'embedding distribué.** Chaque mot devient un vecteur dense appris. Les mots qui
   apparaissent dans des contextes semblables reçoivent des vecteurs voisins, donc une
   séquence jamais vue à l'entraînement hérite de la probabilité de ses voisines. La
   généralisation vient de la géométrie de l'espace, pas du comptage.

L'architecture est un contexte de taille fixe → concaténation des embeddings → une
couche cachée non linéaire → distribution sur le vocabulaire. C'est la cible de l'étape 1.

Section utile : celle sur le coût du softmax sur grand vocabulaire — le problème qui
motivera plus tard le passage aux sous-mots (voir #8).

---

## 4. Elman (1990) — *Finding Structure in Time*

*Cognitive Science*, 14(2):179–211.

**L'origine des RNN (étape 3).** Question posée : comment représenter le temps dans un
réseau ? La réponse d'Elman est de rendre l'état caché récurrent — le réseau reçoit à
chaque pas l'entrée courante *et* son propre état au pas précédent, qui sert de mémoire.
Le contexte n'est plus une fenêtre de taille fixe, il est accumulé.

Article de sciences cognitives plus que d'ingénierie, court et lisible. L'analyse des
représentations apprises (le réseau découvre seul des catégories nom/verbe à partir de
la seule tâche de prédiction du mot suivant) est ce qui a le mieux vieilli.

---

## 5. Hochreiter & Schmidhuber (1997) — *Long Short-Term Memory*

*Neural Computation*, 9(8):1735–1780.

**Étape 3, second temps.** Diagnostic d'abord : dans un RNN simple, le gradient
rétropropagé sur k pas est un produit de k facteurs. Selon que ces facteurs sont en
moyenne inférieurs ou supérieurs à 1, il s'évanouit ou explose exponentiellement. Un RNN
simple n'apprend donc pas de dépendances longues — non par manque de capacité, mais
parce que le signal d'apprentissage n'arrive pas jusqu'au passé lointain.

Remède : une cellule dont l'état se propage par une voie essentiellement additive, et
des portes apprises qui décident ce qui entre, ce qui sort et ce qui est oublié. Le
gradient traverse cette voie sans être multiplié à chaque pas.

L'article est long et sa notation a vieilli ; les sections 1 à 3 (analyse du problème)
suffisent, et ce sont les plus instructives. La porte d'oubli n'y est d'ailleurs pas —
elle est ajoutée par Gers et al. en 2000.

---

## 6. Vaswani et al. (2017) — *Attention Is All You Need*

NeurIPS 2017. arXiv:1706.03762.

**L'article de l'étape 4.** La récurrence impose un traitement séquentiel : le pas t
attend le pas t−1, donc rien ne se parallélise sur la longueur, et l'information entre
deux positions distantes doit traverser tous les pas intermédiaires. L'attention
remplace ce chemin par un accès direct : chaque position consulte toutes les autres en
une seule opération, et le chemin entre deux positions quelconques devient de longueur
constante.

Points à ne pas survoler :
- la mise à l'échelle du produit scalaire, et l'argument de variance qui la motive ;
- le **masquage causal** — sans lui, le modèle lit le futur et la loss d'entraînement
  devient un mensonge ;
- l'intérêt de plusieurs têtes plutôt qu'une seule plus large ;
- l'encodage de position, rendu nécessaire par le fait que l'attention seule est
  invariante par permutation.

Le projet ne vise que la moitié décodeur, sans l'encodeur ni l'attention croisée.

---

## 7. Karpathy — *nanoGPT*

`github.com/karpathy/nanoGPT`.

Transformer décodeur complet en ~300 lignes lisibles, avec sa boucle d'entraînement.
**Cible architecturale de l'étape 4, à ouvrir seulement après avoir écrit sa propre
version** — la comparaison ligne à ligne vaut alors une relecture de code.

Utile aussi comme point de repère sur les ordres de grandeur : tailles de modèle,
longueur de contexte, budgets d'entraînement.

---

## 8. Sennrich, Haddow & Birch (2016) — *Neural Machine Translation of Rare Words with Subword Units*

ACL 2016. arXiv:1508.07909.

**BPE, étape 5.** Le dilemme : au niveau caractère, les séquences sont longues et le
modèle dépense sa capacité à réapprendre l'orthographe ; au niveau mot, le vocabulaire
est énorme, la queue de distribution mal estimée, et tout mot inconnu devient `<unk>`.

Byte Pair Encoding, détourné de la compression, tranche : on part des caractères et on
fusionne itérativement la paire d'unités adjacentes la plus fréquente du corpus, un
nombre fixé de fois. Les mots fréquents finissent en une seule unité, les mots rares se
décomposent en morceaux réutilisables. Aucun mot n'est hors vocabulaire, et la taille du
vocabulaire est un paramètre que l'on choisit.

Attention en lisant : le vocabulaire s'*apprend* sur le corpus d'entraînement, et
l'encodage d'un texte nouveau réapplique les fusions apprises dans l'ordre. Confondre
les deux phases est l'erreur classique.

---

## 9. Eldan & Li (2023) — *TinyStories*

arXiv:2305.07759.

**Ce qui justifie le dimensionnement du projet.** Sur un corpus synthétique d'histoires
courtes n'employant que le vocabulaire d'un enfant de 3–4 ans, des modèles de 1 à 33 M
de paramètres produisent un anglais grammatical, cohérent et parfois créatif — là où des
modèles bien plus gros entraînés sur du texte généraliste échouent à ce format.

Conclusion opérationnelle : la cohérence dépend autant de l'adéquation corpus/modèle que
de la taille brute. Un corpus restreint et homogène est un choix de conception, pas une
concession. C'est ce qui rend l'objectif du projet atteignable sur 8 Go de VRAM.

---

## 10. Hoffmann et al. (2022) — *Training Compute-Optimal Large Language Models* (Chinchilla)

arXiv:2203.15556.

**Le budget de tokens de l'étape 5.** À budget de calcul fixé, les modèles de l'époque
étaient trop gros et trop peu entraînés. La loi empirique : paramètres et tokens
d'entraînement doivent croître à peu près proportionnellement — de l'ordre de 20 tokens
par paramètre.

Application directe ici : la question n'est pas « quelle taille de modèle tient dans
8 Go ? » mais « quelle taille de modèle mon corpus peut-il nourrir ? ». Un modèle de
30 M de paramètres appelle quelques centaines de millions de tokens ; en dessous, la
capacité supplémentaire ne fait que mémoriser.

---

## 11. Holtzman, Buys, Du, Forbes & Choi (2020) — *The Curious Case of Neural Text Degeneration*

arXiv:1904.09751 (ICLR 2020).

**L'article du décodage, et il est arrivé au bon moment : le 15/08/2026, le banc d'essai
de `generation.py` a reproduit son résultat central sur ce modèle-ci.**

La thèse : maximiser la vraisemblance produit du texte *dégénéré*. Le décodage glouton et
le beam search entrent dans des boucles — mesuré ici sur `p=0.8, T=0.4`, où 12 textes sur
100 partent en cycle (« *and his hat and his hat and his hat…* »). La cause n'est pas un
défaut du modèle mais du critère : le texte humain n'est pas la suite de mots la plus
probable, il est *surprenant par endroits*.

L'article introduit **top-p (nucleus sampling)** : garder le plus petit ensemble de tokens
dont la probabilité cumulée atteint `p`, puis renormaliser. Son avantage sur top-k est que
la taille de l'ensemble **s'adapte** — large quand la distribution est plate, étroite
quand elle est piquée, là où un `k` fixe impose la même largeur dans les deux cas.

À relire pour la suite : l'article insiste sur le fait qu'aucune métrique unique ne
suffit à juger un décodage. Le piège rencontré ici en est l'illustration — `p=0.2, T=0.4`
obtient zéro sur les deux métriques mesurées (répétition, mots inexistants) tout en
produisant cent fois la même histoire, faute d'une mesure de diversité.

---

## 12. Gu & Dao (2023) — *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*

arXiv:2312.00752.

**Candidat pour l'étape d'après.** Le problème posé : un transformer paie l'attention en
O(T²) et traîne un cache qui grossit à la génération ; une récurrence est en O(T) mais
refuse de se paralléliser à l'entraînement. Un modèle d'espace d'états cherche les deux
à la fois.

Le mécanisme tient à une propriété : `dh/dt = A·h + B·x` est **linéaire en l'état**. Une
récurrence linéaire se replie algébriquement — toutes les positions se calculent d'un
coup à l'entraînement, tandis que la génération garde un état de taille fixe. D'où la
formule : il s'entraîne comme une convolution et s'exécute comme une récurrence.

L'apport propre de Mamba est de rendre `A`, `B`, `C` dépendants de l'entrée — la
*sélectivité*. Cela casse la convolution, remplacée par un balayage parallèle.

Prérequis : **S4** (Gu, Goel & Ré, 2022, arXiv:2111.00396), dont il suffit de lire la
motivation ; la théorie HiPPO peut attendre une seconde lecture. Dans Mamba,
l'introduction et la section 3 suffisent à en écrire un — la partie « hardware-aware »
décrit leur noyau CUDA, pas l'algorithme.

**À lire avant d'écrire**, contrairement à nanoGPT : c'est une architecture qu'on n'a
pas, pas une version d'une architecture qu'on a déjà écrite.

---

## 13. Hasani, Lechner, Amini, Rus & Grosu (2021) — *Liquid Time-constant Networks*

AAAI 2021, arXiv:2006.04439. Suite : *Closed-form continuous-time neural networks*,
Nature Machine Intelligence, 2022.

**À lire pour savoir ce qu'on écarte, et pourquoi.** La constante de temps de chaque
neurone dépend de l'entrée et de l'état courant : la vitesse d'oubli varie selon ce qui
est lu. Rien ne se réapprend au lancement — les poids sont figés comme partout ailleurs ;
c'est l'échelle de temps qui est mouvante, et le nom « liquide » prête à confusion sur ce
point.

Le prix de cette expressivité : l'équation est **non linéaire en l'état**, donc les pas
doivent s'enchaîner un par un, à l'entraînement comme à l'inférence. La version en forme
close de 2022 supprime le solveur numérique, pas la séquentialité.

Son terrain est celui des signaux continus à échantillonnage irrégulier — capteurs,
séries médicales, contrôle ; leur résultat le plus connu est un pilotage automobile avec
19 neurones. Sur du texte régulièrement tokenisé, cet avantage n'est jamais exercé.

Le raisonnement complet de la mise à l'écart est dans `JOURNAL.md`, entrée du 28/08/2026.
Sections 1 à 3 suffisantes.

---

## 14. Wei et al. (2022) — *Chain-of-Thought Prompting Elicits Reasoning in LLMs*

NeurIPS 2022, arXiv:2201.11903.

**La référence du mode de raisonnement « avec mots ».** Demander au modèle de produire
les étapes intermédiaires avant sa réponse améliore nettement les tâches de raisonnement.

Le point qui compte ici est la réserve, pas le résultat : le gain **n'apparaît qu'au-delà
d'un seuil d'échelle**, et en dessous la chaîne de pensée dégrade les performances. À
12 M de paramètres, la reproduire directement n'a pas de sens.

Et le blocage n'est pas que la taille : TinyStories ne contient aucune étape
intermédiaire. On ne peut pas comparer deux modes de raisonnement sur des données qui
n'en demandent aucun. La sortie est de changer de tâche, pas d'échelle — voir l'entrée 15.

---

## 15. Lee, Sreenivasan, Lee, Lee & Papailiopoulos (2023) — *Teaching Arithmetic to Small Transformers*

arXiv:2307.03381.

**C'est cette entrée qui rend l'expérience faisable à cette échelle.** De petits
transformers apprennent l'addition à plusieurs chiffres — à condition que le **format des
données** expose les étapes. Le choix de représentation pèse plus que la taille du modèle.

Conséquence directe : l'effet de la chaîne de pensée est reproductible sur un problème
qui se décompose réellement, avec un modèle de la taille du sien. La comparaison devient

| entraînement | ce que le modèle voit |
|---|---|
| sans raisonnement | `entrée → réponse` |
| avec mots | `entrée → étapes → réponse` |
| sans mots | `entrée →` rebouclage de l'état caché, puis réponse (entrée 16) |

Avantage pratique : le corpus se génère, donc il est en zone verte, gratuit, infini, et la
difficulté se règle exactement.

---

## 16. Hao et al. (2024) — *Training Large Language Models to Reason in a Continuous Latent Space* (Coconut)

arXiv:2412.06769.

**Le mode de raisonnement « sans mots ».** Au lieu de décoder un token à chaque étape de
raisonnement, on réinjecte le dernier état caché comme embedding d'entrée du pas suivant :
le raisonnement reste dans l'espace continu et n'est jamais verbalisé.

Pour la comparaison, c'est la bonne forme : même modèle, même tâche, **une seule
modification dans la boucle de génération**. Une variable change.

Réserve honnête : leurs résultats portent sur de gros modèles. À petite échelle, sur une
tâche synthétique, la question est ouverte — c'est précisément ce qui rend l'expérience
intéressante plutôt que confirmatoire.

---

## Compléments (à lire au besoin, quand le sujet se présente)

- **Nielsen (2015), *Neural Networks and Deep Learning*, chapitre 2** —
  `neuralnetworksanddeeplearning.com/chap2.html`. La dérivation complète de la
  rétropropagation en mathématiques, avec ses quatre équations fondamentales. Utile à
  l'étape 1 comme substitut « sans code » à l'épisode 1 de Karpathy. Le chapitre contient
  du Python plus bas dans la page : la partie qui t'intéresse est avant.
- **Kingma & Ba (2014), *Adam*** — arXiv:1412.6980. À ouvrir quand la SGD de l'étape 1
  devient pénible à régler, pas avant : le contraste est l'intérêt.
- **Ba, Kiros & Hinton (2016), *Layer Normalization*** — arXiv:1607.06450. Utile à
  l'étape 4, où la normalisation conditionne la stabilité de l'entraînement.
- **Glorot & Bengio (2010)** et **He et al. (2015)** sur l'initialisation des poids —
  à sortir le jour où une loss reste plate ou part à `NaN` dès les premiers pas.

## 17. Encodages de position qui extrapolent — RoPE et ALiBi *(piste, pas encore lue)*

Ouverte le 22/09/2026, en butant sur le plafond de contexte du run long.

Le problème concret : `pos_emb` est une table apprise de forme `(max_len, dim)`. Il n'existe
pas de ligne 385, donc la génération ne peut pas dépasser 384 tokens, et allonger le contexte
impose de tout réentraîner. La table est une partie des poids, pas un réglage.

- **Su et al. (2021), *RoFormer: Enhanced Transformer with Rotary Position Embedding***
  — arXiv:2104.09864. L'information de position est appliquée comme une **rotation** des
  vecteurs requête et clé, d'un angle proportionnel à la position. Le produit scalaire
  entre deux positions ne dépend alors que de leur **écart**, pas de leurs valeurs absolues.
  Rien n'est appris, donc rien ne borne la longueur. C'est ce qu'utilisent la plupart des
  modèles récents.
- **Press, Smith & Lewis (2021), *Train Short, Test Long: Attention with Linear Biases
  Enables Input Length Extrapolation*** — arXiv:2108.12409. Encore plus simple : aucun
  encodage de position, mais un **biais linéaire en la distance** ajouté aux scores
  d'attention, pénalisant les positions lointaines. Le titre est la promesse : entraîner
  court, généraliser long.

À quelle étape : ni l'une ni l'autre ne résout le vrai mur de l'étape 5, qui est que
**TinyStories ne contient pas d'histoires longues** — un contexte extensible ne sert à rien
sans données longues à y mettre. Elles deviennent pertinentes à l'étape 6, quand la question
sera le coût de l'attention en fonction de la longueur, et qu'on la comparera à un modèle
d'espace d'états dont le coût est linéaire. À lire à ce moment-là, pas avant.

Le contournement sans article et sans réentraînement : **faire glisser la fenêtre** à la
génération, en ne donnant au modèle que les `max_len` derniers tokens. Le plafond disparaît,
la mémoire reste de 384 tokens — ce qui produit un texte long qui oublie son propre début, et
donne à voir la limite au lieu de la contourner.
