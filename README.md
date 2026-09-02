# Completiontexte

Un modèle de langage écrit de zéro, de la table de comptage bigramme au transformer
décodeur entraîné sur 126 millions de tokens — sans `transformers`, sans `tokenizers`,
sans copier-coller.

L'objectif n'est pas d'obtenir le meilleur modèle : c'est de comprendre chaque
mécanisme en l'écrivant. Tout ce qui apprend, prédit ou calcule un gradient a été écrit
à la main.

## La contrainte de méthode

Le projet est mené sous une règle explicite, formalisée dans [CLAUDE.md](CLAUDE.md) :
le dépôt est découpé en **zone rouge** et **zone verte**.

| Zone | Contenu | Qui écrit |
|---|---|---|
| 🔴 `src/model/` | architecture, forward, backward, perte, optimiseur, tokenizer, échantillonnage | moi, exclusivement |
| 🟢 `src/tooling/`, `scripts/`, `tests/` | préparation des corpus, I/O, CLI, logs, courbes, tests de formes | assistance IA autorisée |

L'assistant IA que j'utilise est cantonné à un rôle de professeur et de relecteur sur la
zone rouge : il peut poser des questions, nommer un concept, pointer une erreur et sa
ligne — jamais écrire le code. Un mot-clé `PSEUDOCODE` déclenche du pseudo-code en
français, non exécutable, et **chaque recours est daté et consigné** en fin de
[JOURNAL.md](JOURNAL.md) : c'est la liste de ce que je n'ai pas su écrire seul du
premier coup.

Cette contrainte prime sur la vitesse et sur la performance. Un modèle qui marche mais
que je n'ai pas écrit serait un échec du projet.

## Feuille de route et avancement

| # | Étape | Écrit à la main | État |
|---|---|---|---|
| 0 | Bigramme par comptage | table de fréquences, lissage, échantillonnage | ✅ 26/07/2026 |
| 1 | MLP caractères en **NumPy pur**, backprop à la main | forward, backward, SGD | ✅ 29/07/2026 |
| 2 | Le même MLP en PyTorch | vérification contre les gradients de l'étape 1 | ✅ 29/07/2026 |
| 3 | RNN puis LSTM | cellule récurrente, BPTT | ✅ 29/07/2026 |
| 4 | Transformer décodeur | attention causale, multi-têtes, blocs résiduels | ✅ 17/08/2026 |
| 5 | Tokenizer BPE + passage à l'échelle | BPE, entraînement long, lots | 🔄 en cours |
| 6 | Transformer contre modèle d'espace d'états | récurrence linéaire, balayage parallèle | 🔭 piste |
| 7 | Raisonnement : en mots contre latent, sur tâche synthétique | rebouclage de l'état caché, corpus généré | 🔭 piste |
| 8 | Substrat analogique : quantification, bruit, crossbar | quantification des poids, injection de bruit | 🔭 branche latérale |

Les étapes 6 à 8 sont des **branches, pas une suite** — elles s'ouvrent une fois que la 5
tourne, dans l'ordre qu'on veut.

La **6** est la moins chère : remplacer l'attention par un SSM ne touche qu'une couche,
tout le reste du fichier est identique, donc la comparaison n'a qu'une variable. La **7**
exige un corpus nouveau — TinyStories ne contient aucune étape intermédiaire à raisonner ;
il se génère, donc il est gratuit. La **8** est latérale : elle enseigne le substrat de
calcul plutôt que les modèles de langue, et sa première question se règle entièrement en
logiciel — mes poids survivent-ils à 6 bits bruités ?

## Résultats

**Étape 0 — bigramme caractère** sur 32 032 prénoms : log-vraisemblance négative de
**2,45** par transition (baseline uniforme `ln 27 ≈ 3,296`), soit une perplexité de
**11,6**. Conforme à la référence publiée.

**Étape 5 — transformer sur TinyStories tokenisé en BPE.** Corpus de 500 Mo encodé avec
un vocabulaire de 2080 tokens appris à la main, soit **126 150 131 tokens**
(train 113,6 M · val 6,27 M · test 6,32 M), avec aller-retour encodage/décodage exact
sur les trois splits.

Premier run — 963 000 paramètres (dim 128, 2 blocs, 4 têtes), 24,6 M tokens vus,
2 min 30 sur RTX 5050 :

| pas | perte val | perplexité |
|---|---|---|
| 240 | 4,1255 | 61,9 |
| 1440 | 3,0631 | 21,4 |
| 2880 | 2,7979 | 16,4 |

Validation décroissante à chaque mesure, écart train/val de 0,03 — aucun surapprentissage.

Génération obtenue :

> Once upon a time, there was a little boy named Tim. Tim was eagle to clean things.
> He liked to hope with his friends, and Fluffy was having lots of fun. They played all
> day. The moral of the story is to be brave and be honey.

Grammaire, ponctuation et formules du corpus acquises ; sémantique encore défaillante
(`eagle` pour *eager*, `honey` pour *honest*) — le régime attendu à ce nombre de
paramètres d'après Eldan & Li, *TinyStories*.

### Le mur de la profondeur, et ce qui le levait

Six blocs refusaient d'apprendre. La perte restait à **6,37**, c'est-à-dire au plancher
d'entropie unigramme du corpus (`6,3149`) : le modèle n'avait acquis que la fréquence
des tokens. Deux blocs fonctionnaient. L'initialisation avait été écartée par
l'expérience — 2 000 pas de plus, aucun effet.

La cause était le **placement des normalisations**.

```
post-norm    x = LN( x + f(x) )      la normalisation est SUR le chemin résiduel
pre-norm     x = x + f( LN(x) )      la normalisation est SUR l'entrée de la sous-couche
```

En post-norm, six blocs imposent **douze** renormalisations successives du flux
résiduel : le gradient n'atteint plus les premières couches. En pre-norm le flux
traverse le réseau sans jamais être recalé, et seules les sous-couches voient une
entrée normalisée — ce qui rend obligatoire une normalisation finale avant `W_out`,
sans quoi rien ne borne plus l'échelle des logits.

Mesuré à 250 pas, tout le reste égal :

| 6 blocs | perte |
|---|---|
| post-norm | 6,3664 |
| pre-norm complet | **5,0173** |

L'effet se retrouve à la génération. Taux de mots inexistants en top-k à `T=1,2`, mesuré
sur le banc d'essai du 15/08 (4 amorces × 100 répétitions, protocole antérieur à celui
décrit plus bas) :

| T=1,2 | k=3 | k=10 | k=20 | k=40 | k=80 |
|---|---|---|---|---|---|
| 2 blocs post-norm | 0,243 | 0,782 | 1,247 | 2,045 | 3,058 |
| 6 blocs pre-norm | 0,534 | 0,458 | **0,678** | **1,724** | **2,267** |

Là où le modèle prend des risques, il se trompe environ un tiers de fois moins. À `T=1,6`
le gain disparaît : on échantillonne alors dans la queue de la distribution, et un
meilleur modèle n'y aide pas.

### Run final — 12,39 M de paramètres

dim 384 · 6 blocs · 4 têtes · contexte 384 · lot 32 · Adam écrit à la main ·
30 000 pas · 3 h 30 sur RTX 5050 · 369 M tokens traversés, soit 3,25 passages sur le
split d'entraînement.

| pas | perte val | perplexité |
|---|---|---|
| 1 500 | 2,3155 | 10,1 |
| 15 000 | 1,5814 | 4,86 |
| 30 000 | **1,4635** | **4,32** |

À budget de pas égal, contre le meilleur modèle post-norm (2 blocs, dim 512,
8,63 M de paramètres) :

| pas | 2 blocs post-norm | 6 blocs pre-norm |
|---|---|---|
| 3 000 | 2,1130 | 1,9513 |
| 15 000 | 1,7137 | 1,5361 |
| 30 000 | 1,6132 | **1,4283** |

La profondeur dépasse la version à 2 blocs dès 3 000 pas. Réserve honnête : le modèle
profond a aussi 44 % de paramètres en plus, donc l'expérience ne sépare pas l'effet de
la profondeur de celui de la taille. Ce qu'elle établit sans ambiguïté, c'est que la
profondeur est devenue *utilisable* — elle ne l'était pas du tout.

Écart train/val de 0,035, soit 2,5 % : pas de surapprentissage. Mais la perte a plafonné
sur les 1 000 derniers pas, et 12,39 M de paramètres pour 369 M de tokens traversés
dépasse le ratio de Chinchilla. Le levier suivant est d'agrandir modèle **et** corpus
ensemble, pas le nombre de pas.

Génération obtenue :

> Once upon a time, there was a little boy named Tim. He loved to play with his toys.
> One day, he found a big box in the attic. It was dark and full of old things.

Les défauts ont changé de nature depuis le premier run. Les mots inventés se raréfient ;
restent des phrases grammaticalement correctes et sémantiquement absurdes — *« The water
was going to rain »*, *« The kind farmer promised collected perseverances »*. Aucune
métrique écrite à ce jour ne les détecte.

### Banc d'essai des réglages d'échantillonnage

24 020 textes générés sur une grille de réglages — 6 valeurs de `k`, 6 de `p`, 4
températures, 20 amorces, 25 répétitions — puis **trois** mesures : le taux de mots absents
du corpus (26 107 mots distincts), le taux de 4-grammes répétés à l'intérieur d'un même
texte, et le taux de 4-grammes partagés **entre** les textes d'un même réglage.

Le fichier de génération porte sa provenance — checkpoint, pas, perplexité, graine de base
et règle de dérivation, grilles réellement employées — et le run est rejouable à l'identique.

**Résultat principal : top-p dégénère beaucoup plus vite que top-k.**

Taux de redite entre textes à `T=0,4`, en pourcentage de 4-grammes déjà vus dans un autre
texte du même réglage :

| top-k | k=3 | k=5 | k=10 | k=20 | k=40 | k=80 |
|---|---|---|---|---|---|---|
| redite | 47,3 | 39,6 | 38,0 | 37,8 | 36,3 | 37,0 |

| top-p | p=0,2 | p=0,4 | p=0,5 | p=0,6 | p=0,8 | p=0,9 |
|---|---|---|---|---|---|---|
| redite | **95,8** | 89,6 | 79,9 | 71,0 | 53,2 | 44,4 |

À basse température, le noyau de top-p se referme sur une poignée de tokens et la même
histoire ressort presque à chaque tirage. Top-k, qui garde un nombre fixe de candidats,
ne descend jamais sous 36 %.

Les réglages qui tiennent les trois critères ensemble :

| | inexistants | répétition | redite |
|---|---|---|---|
| `k=5 · T=1,2` | 0,049 % | 0,335 % | 16,2 % |
| `k=80 · T=0,8` | 0,185 % | 0,264 % | 14,0 % |
| `p=0,6 · T=1,2` | 0,078 % | 0,388 % | 13,2 % |
| `p=0,8 · T=0,8` | 0,018 % | 0,556 % | 21,4 % |

`genere()` porte désormais `k=5 · T=1,2` par défaut. L'ancien défaut `k=3` était dominé
sur les trois métriques à la fois.

Trois enseignements de méthode, tous obtenus par la mesure et non par la lecture :

- **un défaut de top-p est resté invisible à la lecture.** L'implémentation prenait le
  token de la frontière au lieu d'échantillonner dans le noyau. Le texte paraissait
  correct ; c'est la mesure qui a révélé le défaut — 0,000 % à bas `p` (donc du greedy)
  et 31,9 % à `p=0,9` ;
- **la moyenne est à queue lourde.** Un ou deux textes sur cent, entrés en boucle
  (*« and his hat and his hat and… »*), décident du chiffre d'une case entière. C'est la
  dégénérescence décrite par Holtzman et al., 2020 — l'article qui a introduit top-p,
  précisément contre ça ;
- **deux métriques opposées ne suffisent pas si elles partagent un angle mort.** Le
  réglage `p=0,2 · T=0,4` affiche 0,417 % de mots inexistants et 0,477 % de répétition,
  des chiffres honorables — pour 95,8 % de redite. La troisième métrique était nécessaire
  pour le voir ; elle montre au passage que le front de Pareto est sain, ce que les deux
  premières ne pouvaient pas établir.

## Structure

```
src/model/      🔴 bigram, rnn, lstm, transformer, encodeur/decodeur BPE, tireur_de_lot,
                   generation (greedy/top-k/top-p), Vocabulaire, analyse_generation
src/tooling/    🟢 tracer.py — tracés, palette, sérialisation des mesures
scripts/        🟢 préparation des corpus, encodage à grande échelle, splits
notes/biblio.md    bibliographie annotée
JOURNAL.md         journal d'apprentissage, un compte rendu par étape
runs/              checkpoints, logs et courbes (non versionnés)
data/              corpus et encodages (non versionnés)
```

## Lancer

Depuis la racine du dépôt, dans cet ordre. Chaque étape lit ce que la précédente a écrit.

```bash
# 1. tokenizer BPE : fusions, vocabulaire, encodage, splits, vérifications
python3 scripts/tokenizer.py --fusions 2000

# 2. liste des mots du corpus, pour l'analyse (26 107 mots, ~60 s)
python3 src/model/Vocabulaire.py

# 3. entraînement — écrit runs/save_runs/<date>/ avec checkpoints et log.txt
python3 src/model/transformer.py

# 4. génération — sélectionne le checkpoint le plus récent, écrit runs/generation/
python3 src/model/generation.py

# 5. analyse — sélectionne la génération la plus récente, écrit mesures et graphiques
python3 src/model/analyse_generation.py
```

Les étapes 3, 4 et 5 ne prennent aucun argument : elles retrouvent seules le dossier le
plus récent. C'est délibéré — un chemin recopié à la main est le moyen le plus sûr
d'analyser l'ancien modèle en croyant analyser le nouveau, sans qu'aucune erreur ne se
lève.

Les hyperparamètres sont en tête de `src/model/transformer.py`. Un run de 30 000 pas à
6 blocs occupe 3,3 Go de VRAM et 3 h 30.

Dépendances : `torch` et `numpy` uniquement. `transformers`, `tokenizers` et `keras`
sont volontairement proscrits — ils contiennent précisément ce que le projet consiste à
écrire.

Environnement de développement : RTX 5050 Laptop 8 Go, PyTorch 2.13 + CUDA 13,
Python 3.14.

## Journal

[JOURNAL.md](JOURNAL.md) contient un compte rendu par étape, rédigé de mémoire sans
relire le code — le test étant que ce que je ne sais pas réexpliquer, je ne l'ai pas
encore appris. On y trouve aussi les blocages réels : la migration vers les lots et le
décalage de tous les axes positifs, `cross_entropy` et son axe des classes, le
`.to(device)` qui transforme un paramètre en tenseur non-feuille et fait planter Adam,
la pré-tokenisation reprise six fois avant que le curseur avance du bon nombre de
caractères, et les quatre versions successives du pre-norm — dont trois déplaçaient la
normalisation au bon endroit du fichier mais sur le mauvais argument.
