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

### L'encodeur : de deux heures à cinq minutes, et de 28 Go à 1,3

Le corpus était le facteur limitant identifié le 23/08 : 450 Mo utilisés sur les 2,2 Go
disponibles, 29,8 tokens vus par paramètre là où l'optimum de Chinchilla en demande 20.
Impossible d'agrandir le modèle sans relire le corpus davantage et surapprendre.

Ce qui bloquait n'était pas le modèle mais la chaîne d'encodage, et pour deux raisons
distinctes — le temps, puis la mémoire.

**Le temps : la mémoïsation.** L'encodeur appliquait les 2 000 règles de fusion à tout le
corpus. Or un texte de 25 Mo ne contient que 39 229 mots distincts pour 4,99 millions
d'occurrences : chaque mot y revient 127 fois en moyenne, et son découpage ne dépend que
de lui-même. Une table mot → tokens réduit le travail réel d'un facteur 127.

Cette équivalence n'est valable que parce qu'aucune fusion ne peut chevaucher une
frontière de mot. La garantie tient à la façon dont les paires sont formées à
l'entraînement : le second membre d'une règle n'est jamais en tête de mot, donc ne porte
jamais l'espace initial. Vérifié sur le vocabulaire — 1 424 règles dont le premier membre
commence par un espace, **zéro** pour le second.

**La mémoire : trois passages en flux.** La mémoïsation seule ne suffisait pas — le corpus
entier restait en mémoire sous forme d'objets Python. Mesuré sur 100 Mo, à chaque étape :

| | RAM par Mo de corpus | pour 2,2 Go |
|---|---|---|
| encodeur mémoïsé | 12,9 | 28 Go |
| découpage en générateur | 3,1 | 6,8 Go |
| lecture du corpus par blocs | 2,2 | 4,8 Go |
| sortie binaire `uint16` | **0,9** | **~1,3 Go** |

La dernière ligne fait disparaître `rencode.py`, l'étape de conversion token → entier qui
réclamait à elle seule 28 Go pour 500 M de tokens. Les indices sont désormais stockés dans
la table de mémoïsation — la conversion se fait une fois par mot distinct, 67 000 fois au
lieu de 560 millions — écrits par tampons dans un binaire brut, puis relus en une passe
pour produire le `.npy`.

**Ce que ça donne** : réencoder les 450 Mo prend 156 s contre ~2 h, et les 2,2 Go tiennent
en ~25 min et 1,3 Go de RAM.

La contrainte tout du long était de ne rien changer à la tokenisation. Trois vérificateurs
écrits en zone verte l'ont garantie : réversibilité du découpage, tokenisation d'un mot
isolé contre la référence (5 397/5 397 mots distincts), et comparaison de l'encodeur entier
à sa version committée. Les 126 150 131 tokens des trois splits sortent **identiques,
entier pour entier**, à ceux produits en août.

### Banc d'échelle : une courbe en U, et une instabilité

Neuf heures de GPU, six configurations, **90 minutes de chronomètre chacune**. Le
protocole est iso-calcul et non iso-pas : à budget de temps égal, le petit modèle fait
beaucoup de pas, le gros en fait peu mais chacun vaut davantage. C'est la question de
Chinchilla posée sur une machine réelle — à calcul fixé, quelle taille minimise la perte ?

Le nombre de pas n'est ni choisi ni estimé : chaque run est arrêté au chronomètre et le
compte réel se lit dans son journal. Deux versions antérieures dérivaient les pas d'une
sonde de 40 pas ; elle mesurait surtout le temps de démarrage et donnait 361 puis 497 ms
pour la même configuration à cinq minutes d'intervalle. Des budgets inégaux à 38 % près
auraient vidé l'iso-calcul de son sens.

| config | dim | blocs | paramètres | pas faits | ms/pas | perte val |
|---|---|---|---|---|---|---|
| A1-petit | 256 | 4 | 4,32 M | 45 750 | 118 | 1,6194 |
| **A2-actuel** | **384** | **6** | **12,39 M** | **17 750** | **304** | **1,5228** |
| A3-moyen | 512 | 7 | 24,38 M | 10 000 | 540 | 1,6454 |
| B2-profond | 448 | 11 | 28,58 M | 7 500 | 720 | 1,7630 |
| B1-large | 640 | 5 | 27,52 M | 9 250 | 584 | *4,4512 — diverge* |
| A4-grand | 640 | 8 | 42,28 M | 5 750 | 939 | *4,9105 — diverge* |

**La courbe en U existe** : 1,6194 → **1,5228** → 1,6454. À 90 minutes de calcul sur cette
machine, l'optimum est autour de 12 M de paramètres. Le petit modèle sature faute de
capacité ; le gros n'a pas le temps de voir assez de tokens.

**Mais deux configurations n'ont pas appris**, et les deux ont `dim=640` :

```
B1-large      pas 1000  4,0203  ->  pas 3000  4,4309  ->  pas 9000  4,4512
A4-grand      pas 1000  4,3652  ->  pas 5750  4,9105   (monotone croissante)
A2-actuel     pas 1000  2,6926  ->  pas 6000  1,7525   (pour comparer)
```

B1 descend puis remonte ; A4 ne descend jamais. Toutes les largeurs de 256 à 512
apprennent proprement. Ce n'est donc pas une limite de capacité mais une **instabilité
d'entraînement** : le pas d'apprentissage est fixé à 0,001 pour toutes les tailles, et il
ne survit pas au passage à `dim=640`.

**Ce que le banc ne peut pas trancher.** Si 0,001 est déjà marginal à `dim=512`, alors la
branche droite du U mesure en partie cette instabilité et non le budget de tokens. Les deux
effets sont confondus, et l'optimum apparent à 12 M pourrait être un artefact. De même, la
comparaison profondeur contre largeur — la question que le run d'août n'avait pas séparée —
reste sans réponse, puisque B1 a divergé.

Le banc du pas d'apprentissage (`--lr`) est écrit pour lever ces deux doutes : trois pas
décroissants à `dim=640`, plus un contrôle à `dim=512`. Deux heures.

### Les deux doutes levés : le pas d'abord, puis le banc refait

**Le balayage du pas** (2 h, 30 min par configuration) tranche la première question. À `dim=640`,
les trois pas plus faibles descendent régulièrement là où 0,001 remontait :

| au pas 2000 | lr 0,001 | 5·10⁻⁴ | 2,5·10⁻⁴ | 1,25·10⁻⁴ |
|---|---|---|---|---|
| dim 640 | 4,51 *(monte)* | **2,13** | 2,22 | 2,49 |

Le seuil d'instabilité se situe donc entre 5·10⁻⁴ et 10⁻³. Et le contrôle à `dim=512` confirme le
soupçon : 10⁻³ **handicapait déjà** cette largeur sans la faire diverger — 2,12 contre 2,39 au pas
2000, 1,97 contre 2,07 au pas 3000.

**Le banc refait à 5·10⁻⁴** (9 h, six configurations, même protocole iso-calcul) donne la seconde
série. En retenant pour chaque taille le meilleur de ses deux pas :

| config | forme | paramètres | val à 10⁻³ | val à 5·10⁻⁴ | retenu |
|---|---|---|---|---|---|
| A1-petit | 256 × 4 | 4,32 M | **1,6194** | 1,6492 | 10⁻³ |
| **A2-actuel** | 384 × 6 | 12,39 M | **1,5228** | 1,5693 | 10⁻³ |
| A3-moyen | 512 × 7 | 24,38 M | 1,6454 | **1,6059** | 5·10⁻⁴ |
| A4-grand | 640 × 8 | 42,28 M | *diverge* | **1,7107** | 5·10⁻⁴ |
| B1-large | 640 × 5 | 27,52 M | *diverge* | **1,6541** | 5·10⁻⁴ |
| B2-profond | 448 × 11 | 28,58 M | 1,7630 | **1,6752** | 5·10⁻⁴ |

Trois conclusions.

**La divergence était bien un problème de pas, pas de capacité.** `A4` passe de 4,91 à 1,7107,
`B1` de 4,45 à 1,6541, sans rien changer d'autre que le pas d'apprentissage.

**La courbe en U tient, et son creux était au bon endroit.** Le handicap existait — `A3` gagne
0,04 en passant à 5·10⁻⁴ — mais pas assez pour dépasser `A2`. À 90 minutes de calcul sur cette
machine, l'optimum reste autour de **12 M de paramètres**.

**Le pas optimal décroît avec la largeur.** Les deux plus petites configurations préfèrent 10⁻³,
les quatre plus larges préfèrent 5·10⁻⁴. C'est le comportement que prédit la théorie : le pas
maximal stable varie comme l'inverse de la largeur, ce que confirme aussi le seuil mesuré —
384 → 640 fait ×1,67, et le seuil passe de ~10⁻³ à ~6·10⁻⁴.

### Profondeur contre largeur

La question que le run d'août ne pouvait pas trancher, faute d'avoir séparé la profondeur de la
taille, et que le 08/09 laissait sans réponse faute d'un run stable :

| | forme | paramètres | pas faits | ms/pas | perte val |
|---|---|---|---|---|---|
| **B1-large** | 640 × 5 | 27,52 M | 8 750 | 617 | **1,6541** |
| B2-profond | 448 × 11 | 28,58 M | 7 250 | 745 | 1,6752 |

**À calcul égal, la largeur l'emporte** — de 0,021, avec 4 % de paramètres en moins. Le mécanisme
est dans la colonne des pas : le modèle profond coûte 745 ms contre 617, donc en 90 minutes il
fait 1 500 pas de moins. Ses onze blocs s'exécutent en série, là où la largeur se parallélise.

**Et à pas égal, elles sont à égalité.** Les deux journaux, comparés au même nombre de pas :

| pas | B1-large (640 × 5) | B2-profond (448 × 11) |
|---|---|---|
| 2 000 | 2,2139 | **2,2117** |
| 5 000 | **1,7660** | 1,7727 |
| 7 000 | **1,6680** | 1,6752 |

Les écarts sont de l'ordre de 0,007, et le classement s'inverse d'un point à l'autre. À nombre de
pas fixé, la profondeur et la largeur apprennent donc **aussi bien**. Tout l'avantage de `B1` à
calcul égal vient de sa **vitesse** — 617 ms par pas contre 745, soit 1 500 pas de plus en
90 minutes. Onze blocs s'exécutent en série ; la largeur, elle, se parallélise.

Le résultat n'est donc pas « la largeur apprend mieux » mais « la largeur coûte moins cher par
pas ». Ce qui reste décisif si le temps de calcul est la ressource rare, et cesse de l'être si
c'est la mémoire ou le nombre de paramètres.

Un mot sur l'outil : `scripts/echelle.py` ne modifie jamais `transformer.py`. Il en dépose
une copie paramétrée dans le dossier de chaque run et exécute celle-là — chaque run porte
donc sa propre source et reste rejouable. Chaque substitution doit correspondre exactement
une fois, sinon le banc s'arrête : sans ce garde-fou, un simple renommage ferait tourner
les six configurations sur les valeurs par défaut en affichant des résultats crédibles.

### Branche 8, première mesure : combien de bits les poids portent-ils ?

Sonde préliminaire, sur des modèles intermédiaires de 90 minutes — pas sur le modèle final.
Deux lois de bruit gaussien appliquées aux sept familles de produits matriciels, sans jamais
toucher aux layernorms ni aux biais, qui resteraient numériques sur un substrat analogique :

```
multiplicatif   σ = alpha × |w|             bruit proportionnel au poids
additif         σ = alpha × max|w|          bruit fixe, en fraction de la plage de la matrice
```

Le bruit est tiré par élément, à chaque mesure. Point de méthode : **les lots de validation sont
identiques d'un alpha à l'autre** — `tireur_de_lot` tire avec le module `random`, le bruit avec le
générateur de torch, donc les deux graines sont fixées séparément. L'écart apparié à la mesure
sans bruit a une dispersion de 0,0009 là où la perte brute varie de 0,0057 : la variance de
l'échantillonnage s'annule, seul l'effet du bruit subsiste.

**Tolérance globale** (`A2-actuel`, 12,4 M, perte de base 1,5319), seuil de +0,01 de perte :

| | multiplicatif | additif |
|---|---|---|
| +0,01 (imperceptible) | 6,1 % | 0,63 % |
| +0,1 (visible) | 18,6 % | 1,89 % |

Les deux courbes ont une pente de **2,05** et **2,09** en échelle log-log : doubler le bruit
quadruple la dégradation — le développement au second ordre autour d'un minimum, où le terme
linéaire s'annule.

**Traduit en bits.** Un bruit d'écart-type σ rend indistinguables deux valeurs plus proches que σ.
Sur une plage utile de 2·max|w|, cela laisse 2/alpha niveaux, soit log2(2/alpha) bits :

```
additif 0,63 % de la plage   ->  8,3 bits en virgule fixe
multiplicatif 6,1 % du poids ->  4,0 bits de mantisse en virgule flottante
```

Les deux lois répondent à deux questions de matériel différentes : un crossbar analogique a un pas
constant, donc c'est l'additif qui le concerne ; un accélérateur numérique peut être l'un ou
l'autre, ce qui explique que `fp8` fonctionne là où `int4` échoue.

**La robustesse croît avec la taille, puis sature.** Seuil de +0,01 :

| | A1 (4,3 M) | A2 (12,4 M) | A3 (24,4 M) | B2 (28,6 M, profond) |
|---|---|---|---|---|
| multiplicatif | 3,8 % | 6,1 % | 6,7 % | 6,8 % |
| additif | 0,32 % | 0,63 % | 0,68 % | 0,88 % |

Le rapport des tolérances donne le rapport des courbures : `(6,1/3,8)² ≈ 2,6`. Le minimum d'`A1`
est 2,6 fois plus étroit que celui d'`A2` — un petit modèle n'est pas seulement moins bon, il est
posé dans une vallée plus resserrée, faute de redondance entre ses poids.

### Où le modèle est fragile

Une famille de matrices bruitée à la fois. Seuil de +0,01 en additif :

| famille | poids | A1 | A2 | A3 | bits (A2) |
|---|---|---|---|---|---|
| `W_2` sortie du MLP | 3,5 M | < 1,0 % | **1,09 %** | **1,09 %** | 7,5 |
| `W_o` sortie d'attention | 0,9 M | < 1,0 % | 1,31 % | 1,65 % | 7,2 |
| `W_out` projection finale | 0,8 M | 1,20 % | 1,34 % | 1,45 % | 7,2 |
| `W_1` entrée du MLP | 3,5 M | < 1,0 % | 1,94 % | 2,59 % | 6,7 |
| `W_v` | 0,9 M | 1,16 % | 2,80 % | 2,74 % | 6,2 |
| `W_k` | 0,9 M | 1,18 % | 2,91 % | 2,81 % | 6,1 |
| `W_q` | 0,9 M | 2,35 % | **5,63 %** | **6,34 %** | 5,2 |

Le même ordre sort des trois modèles, de 4,3 à 24,4 M de paramètres. **Et ce n'est pas une question
de volume** : `W_1` et `W_2` ont exactement la même forme et le même nombre de poids, et `W_2` est
deux à trois fois plus fragile ; `W_q`, `W_k` et `W_v` sont identiques en dimensions et vont de
2,8 % à 5,6 %. La fragilité tient au **rôle** de la matrice dans le calcul.

**Ce que la précision mixte rapporterait — et pourquoi c'est décevant.** L'écart entre la famille la
plus exigeante et la plus tolérante vaut 2,3 bits. Mais les familles fragiles sont aussi les plus
grosses : `W_1` et `W_2` pèsent 7 M des 11,4 M de poids. En numérique, passer de 8 bits uniformes à
une allocation par famille ne fait gagner que **8 %** — 10,5 Mo contre 11,4.

En analogique, on peut réduire le bruit d'une matrice en mettant N cellules en parallèle par poids :
les conductances s'ajoutent, les bruits s'ajoutent en quadrature, le rapport gagne √N. Amener `W_2`
au niveau de `W_q` demanderait donc 27 cellules, et l'ensemble du modèle **14,7 fois la surface**.
Le gain est en racine, le prix est linéaire.

## Structure

```
src/model/      🔴 bigram, rnn, lstm, transformer, encodeur/decodeur BPE, tireur_de_lot,
                   generation (greedy/top-k/top-p), Vocabulaire, analyse_generation
src/tooling/    🟢 tracer.py — tracés et mesures ; quatre vérificateurs de la chaîne
                   d'encodage (découpage, tokenisation d'un mot, encodeur entier,
                   fichiers de tokens entiers)
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

# 1 bis. encodage des trois splits en tokens entiers (~3 min pour 500 Mo)
#        encodeur.encode() écrit directement data/encode_tok_<split>.npy

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
