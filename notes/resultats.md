# Résultats détaillés

Le compte rendu complet de chaque expérience : protocole, chiffres bruts, ce que la
mesure tranche et ce qu'elle ne tranche pas. Le [README](../README.md) n'en garde que la
synthèse ; le [journal](../JOURNAL.md) raconte l'apprentissage, ce fichier garde les
résultats.

## Premiers résultats

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

## Le mur de la profondeur, et ce qui le levait

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

## Run final — 12,39 M de paramètres

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

## Banc d'essai des réglages d'échantillonnage

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

## L'encodeur : de deux heures à cinq minutes, et de 28 Go à 1,3

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

## Banc d'échelle : une courbe en U, et une instabilité

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

## Les deux doutes levés : le pas d'abord, puis le banc refait

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

**La courbe en U tient, et son creux était au bon endroit** — *conclusion révisée le 20/09, voir
plus bas : une fois l'entraînement stabilisé par l'échauffement et l'écrêtage, le modèle de 42 M
égale celui de 12 M à calcul égal — égale, et non dépasse : l'écart de 0,003 est vingt fois
plus petit que le plancher de bruit mesuré le 21/09.* Le handicap existait — `A3` gagne
0,04 en passant à 5·10⁻⁴ — mais pas assez pour dépasser `A2`. À 90 minutes de calcul sur cette
machine, l'optimum reste autour de **12 M de paramètres**.

**Le pas optimal décroît avec la largeur.** Les deux plus petites configurations préfèrent 10⁻³,
les quatre plus larges préfèrent 5·10⁻⁴. C'est le comportement que prédit la théorie : le pas
maximal stable varie comme l'inverse de la largeur, ce que confirme aussi le seuil mesuré —
384 → 640 fait ×1,67, et le seuil passe de ~10⁻³ à ~6·10⁻⁴.

## Profondeur contre largeur

La question que le run d'août ne pouvait pas trancher, faute d'avoir séparé la profondeur de la
taille, et que le 08/09 laissait sans réponse faute d'un run stable :

| | forme | paramètres | pas faits | ms/pas | perte val |
|---|---|---|---|---|---|
| **B1-large** | 640 × 5 | 27,52 M | 8 750 | 617 | **1,6541** |
| B2-profond | 448 × 11 | 28,58 M | 7 250 | 745 | 1,6752 |

**À calcul égal, la largeur l'emporte** — de 0,021, avec 4 % de paramètres en moins. *Réserve
ajoutée le 21/09 : 0,021, c'est à peine plus d'une fois et demie le plancher de bruit. La
conclusion tient par son mécanisme, pas par la marge.* Le mécanisme
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

## Branche 8, première mesure : combien de bits les poids portent-ils ?

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

## Où le modèle est fragile

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

## Le pas d'apprentissage : échauffement, décroissance, écrêtage

Jusqu'ici le pas était **constant** — aucun échauffement, aucune décroissance, aucun écrêtage du
gradient. C'est la cause commune de tout ce qui précède : la divergence à `dim=640`, le handicap
de `dim=512`, et donc le biais de la courbe en U.

Trois formes sont désormais construites au lancement depuis `config_pas ['forme', amorce, fin,
base]` : échauffement linéaire, plateau, puis décroissance en cosinus ou en `1/√i`. Le checkpoint
enregistre **les paramètres**, pas la valeur courante — avec le champ `i`, la valeur appliquée à
n'importe quel pas reste recalculable.

**Choisir le seuil d'écrêtage par la mesure.** Une sonde en zone verte relève la norme globale du
gradient à chaque pas, seuil neutralisé, pour observer la distribution brute. Sur quatre tailles,
chacune à son propre pas d'apprentissage :

| config | médiane | p99 | max | évolution de la médiane (par tranche de 200 pas) |
|---|---|---|---|---|
| dim 256 | 0,570 | 0,718 | 3,49 | 0,44 · 0,49 · 0,56 · 0,59 · 0,60 · 0,61 |
| dim 384 | 0,481 | 0,621 | 4,67 | 0,41 · 0,43 · 0,48 · 0,50 · 0,50 · 0,49 |
| dim 512 | 0,642 | 1,014 | 5,58 | 0,52 · 0,58 · 0,64 · 0,66 · 0,66 · 0,66 |
| dim 640 | 0,571 | 1,359 | 12,59 | 0,52 · 0,54 · 0,55 · 0,59 · 0,59 · 0,59 |

**La norme ne dépend pas de la taille du modèle.** De 4,3 à 42 M de paramètres, toutes les
médianes tiennent entre 0,48 et 0,64, et aucune ne dérive au fil des pas. L'attente d'une
croissance en racine du nombre de paramètres est démentie. Un seuil unique convient donc à toutes
les tailles : **1,0**, au-dessus du p99 des petites configurations et en dessous des pics.

En régime divergent, en revanche, la norme s'emballe — `dim 640` à 10⁻³ sans échauffement passe de
0,49 à 8,39 en 1200 pas, avec un maximum à 179. L'écrêtage y devient un frein à la divergence et
non plus un simple garde-fou.

## Ce que l'échauffement change : la courbe en U révisée

`A4-grand` — 42,3 M de paramètres, 6 000 pas, soit 90 minutes :

| régime | perte val | trajectoire |
|---|---|---|
| lr 10⁻³, sans rien | 4,9105 | 4,37 → 4,51 → 4,69 → 4,86 *(diverge)* |
| lr 5·10⁻⁴, sans rien | 1,7107 | le contournement du 15/09 |
| **lr 10⁻³ + échauffement + écrêtage** | **1,5199** | |

Le grand modèle n'était pas trop gros, **il était mal démarré**. Et une fois stabilisé, il bat de
0,19 la solution prudente consistant à baisser le pas — un écart réel, quatorze fois le bruit.

**À budget de calcul égal** — 90 minutes — il atteint 1,5199 là où `A2-actuel`, l'optimum apparent,
plafonnait à 1,5228. *Correction du 21/09 : cet écart de 0,003 est très en dessous du plancher de
bruit de 0,013 mesuré depuis. Il ne dit pas que le grand modèle est meilleur, il dit qu'il est
**à égalité**.* Ce qui suffit à la conclusion qui compte : la branche droite de la courbe en U
était creusée par l'instabilité et non par le budget de tokens, puisqu'un modèle 3,4 fois plus
gros fait désormais aussi bien dans le même temps. Mais **rien n'établit que l'optimum soit
au-dessus de 12 M** — il faudrait pour ça un écart que la mesure n'a pas produit.

## Les trois schedules comparés

`384 × 6`, 15 000 pas, écrêtage à 1,0, échauffement de 1 000 pas, décroissance sur les 1 000
derniers :

| | lr 10⁻³ | lr 2·10⁻³ |
|---|---|---|
| constant | 1,5901 | 3,6829 *(diverge)* |
| **cosinus** | 1,5270 | **1,4912** |
| racine | 1,5378 | 1,5024 |

- **à 2·10⁻³ le pas constant explose**, tandis que les deux autres tiennent : l'échauffement seul
  suffit à rendre utilisable un pas deux fois plus grand, et c'est lui qui produit le meilleur
  modèle de la série ;
- **le cosinus devance la racine** aux deux pas de base, de 0,011 et 0,013 — mais *correction du
  21/09 : c'est exactement le plancher de bruit. L'écart n'établit rien.* L'argument théorique
  reste (le cosinus descend jusqu'à zéro là où la racine s'arrête à `base/31`), la mesure non ;
- leurs trajectoires sont **identiques jusqu'au pas 14 000**, ce qui est attendu : elles ne
  diffèrent que sur les mille derniers. Ce millier vaut pourtant 0,06 face au pas constant.

Cette dernière observation ouvre la question suivante : la décroissance ne couvre ici que **7 %**
du run, là où les entraînements publiés la font commencer juste après l'échauffement.


## Le plancher de bruit, et ce qu'il invalide

Six runs de 15 000 pas, tout gelé aux meilleures valeurs connues — `384 × 6`, cosinus,
lr 2·10⁻³, écrêtage 1,0, échauffement 1 000 — et une seule chose qui varie : le nombre de pas
sur lesquels s'étale la décroissance.

**La mesure qui commande toutes les autres.** Deux paires de runs strictement identiques, à la
graine près :

| configuration | graine 1337 | graine 4242 | écart |
|---|---|---|---|
| décroissance sur 1 000 pas | 1,4912 | 1,4792 | **0,0120** |
| décroissance sur 14 000 pas | 1,4971 | 1,4820 | **0,0151** |

**Deux entraînements identiques diffèrent de 0,013 en moyenne.** C'est le plancher : en dessous
de cet écart, une comparaison ne distingue pas un effet du tirage des lots et de l'initialisation.

Ce chiffre n'avait jamais été mesuré. Tout le mois de septembre a comparé des configurations sans
lui, en traitant des écarts de 0,02 comme des résultats.

**D'où vient cette dispersion ?** Deux sources possibles : ce que la graine contrôle
(l'initialisation des poids et l'ordre des lots) et ce qu'elle ne contrôle pas (l'ordre des
réductions en virgule flottante sur GPU, qui dépend de l'ordonnancement des threads CUDA).
Un run supplémentaire les sépare : même configuration, **même graine 1337**, relancé 36 heures
après le premier.

    perte finale        1,4912  contre  1,4912
    109 lignes de log   diff : aucune difference
    source deposee      identique, au chemin du dossier de sortie pres

**Les deux runs sont strictement identiques.** Le non-déterminisme du GPU ne contribue rien de
mesurable à cette échelle : la totalité des 0,013 vient de la graine, donc de l'initialisation
et de l'ordre des lots. Deux conséquences pratiques : les runs sont reproductibles à graine
fixée sur cette machine — pas besoin de `torch.use_deterministic_algorithms` ni du ralentissement
qui l'accompagne — et le seul levier pour réduire le bruit est de moyenner sur plusieurs graines.

*Réserve : « identiques » vaut aux quatre décimales du journal. Les poids n'ont pas été comparés
bit à bit. Mais 109 mesures qui coïncident sur 15 000 pas d'un système qui amplifie n'importe
quel écart initial suffisent à trancher.*

**Le balayage lui-même**, à graine fixée :

| décroissance sur | part du run | perte val |
|---|---|---|
| 1 000 pas | 7 % | 1,4912 |
| 3 750 | 25 % | **1,4787** |
| 7 000 | 47 % | **1,4789** |
| 10 500 | 70 % | 1,4867 |
| 14 000 | 93 % | 1,4971 |

L'étendue vaut **0,0184** pour un bruit de **0,0135**. Le rapport est de 1,4 : la tendance
apparente — un creux vers 25-47 %, le pire quand la décroissance démarre juste après
l'échauffement — **n'est pas séparable du hasard** sur un run par point. Il en faudrait trois ou
quatre par configuration, soit une nuit par point.

Conclusion pratique : **le moment où commence la décroissance n'est pas un paramètre critique.**
Ce qui est une réponse utile, puisqu'elle dispense d'y revenir.

**Ce que le plancher invalide, rétroactivement.** Les écarts déjà publiés, relus à cette aune :

| affirmation | écart | verdict |
|---|---|---|
| le cosinus bat le pas constant | 0,099 | **tient** — sept fois le bruit |
| l'échauffement sauve `dim=640` (4,91 → 1,52) | 3,39 | **tient** massivement |
| passer de 10⁻³ à 2·10⁻³ à `384 × 6` | 0,036 | **tient** — presque trois fois le bruit |
| la largeur bat la profondeur à calcul égal | 0,021 | **fragile** — 1,5 fois le bruit |
| le cosinus bat la racine | 0,011 | **sous le bruit** — non établi |
| `A4` (42 M) bat `A2` (12 M) à calcul égal | 0,003 | **sous le bruit** — c'est une égalité |

Les deux dernières lignes étaient écrites ici comme des résultats. Elles ne le sont pas.

La leçon de méthode dépasse ce banc : **une comparaison sans plancher de bruit n'est pas une
mesure, c'est une lecture de chiffres.** Le protocole apparié du banc de bruit (mêmes lots d'un
alpha à l'autre) avait déjà ce souci ; il n'avait jamais été porté sur les comparaisons entre
entraînements, où le coût — un run entier jeté pour ne mesurer que la dispersion — le faisait
paraître du luxe.

## Le pas d'apprentissage une fois l'entraînement stabilisé

Le 15/09, à `dim=512`, le meilleur pas mesuré était **5·10⁻⁴** : 10⁻³ handicapait déjà cette
largeur. C'était sans échauffement ni écrêtage. La question reprise le 21/09, avec les deux :
jusqu'où peut-on monter ?

**Balayage sur 1 500 pas** (`512 × 7 × 8`, cosinus, échauffement 1 000, décroissance 500,
écrêtage 1,0) :

| lr | 1,5·10⁻³ | 2·10⁻³ | 3·10⁻³ | 4,5·10⁻³ | 6·10⁻³ |
|---|---|---|---|---|---|
| perte val | 2,0903 | 2,0379 | 1,9828 | 1,9453 | **1,9340** |
| gain | — | −0,052 | −0,055 | −0,038 | −0,011 |

Monotone du début à la fin, aucune divergence, mais **la courbe s'aplatit** : le dernier
doublement ne rapporte plus que 0,011, soit le plancher de bruit.

**Ce que ce balayage ne peut pas dire.** Sur 1 500 pas dont 1 000 d'échauffement et 500 de
décroissance, le modèle ne passe que 500 pas au pas de base — or les divergences de septembre
mettaient 1 000 à 3 000 pas à se déclarer. Un aplatissement n'est pas une stabilité.

**Deux sondes longues** — 5 500 pas, échauffement 1 000, **décroissance neutralisée**, donc
4 500 pas au pas de base, neuf fois plus que précédemment :

| pas | 1 500 | 2 500 | 3 500 | 4 500 | 5 000 | 5 500 |
|---|---|---|---|---|---|---|
| lr 6·10⁻³ | 2,1154 | 1,8880 | 1,7836 | 1,6991 | 1,7091 | **1,6810** |
| lr 4,5·10⁻³ | 2,1172 | 1,8871 | 1,7758 | 1,6904 | 1,6977 | **1,6676** |

**Aucune des deux ne diverge**, et les deux courbes sont indiscernables — écart final 0,0134,
c'est-à-dire le plancher de bruit, et tous les écarts intermédiaires plus petits encore. Entre
4,5·10⁻³ et 6·10⁻³ il n'y a plus rien à gagner.

**Le résultat qui compte.** Sur la même forme, au même pas 2 000 :

    15/09, sans echauffement ni ecretage, lr 5e-4   ->  2,12
    21/09, avec echauffement et ecretage, lr 4,5e-3 ->  1,945

L'échauffement et l'écrêtage font passer le pas utilisable de **5·10⁻⁴ à au moins 6·10⁻³**, un
facteur **douze**.

*Correction du 22/09 : j'ai d'abord conclu qu'« avec eux le plafond remonte tellement qu'on ne
le trouve plus, et ce qui limite n'est plus la stabilité mais le rendement décroissant ». C'était
prématuré — seul `dim=512` avait été testé. Le plafond existe, il fallait une largeur de plus
pour le voir. Voir la section suivante.*

**Un artefact de mesure, visible parce que les deux courbes le partagent.** Les deux sondes
remontent au pas 5 000 puis redescendent, exactement au même endroit. Les lots de validation
sont tirés par la graine, identique dans les deux runs : les deux mesurent donc sur le *même*
lot difficile. C'est du bruit de mesure, pas d'entraînement — et ça signifie que le plancher de
0,013 en contient une part.

## Le run long : l'étape 5 menée au bout

Le livrable qui manquait. Tout ce qui avait tourné jusqu'ici était soit sur l'ancien corpus de
500 Mo (16/08), soit des bancs de 78 à 90 minutes servant à régler les hyperparamètres d'un
entraînement qu'on n'avait pas encore fait.

    A3-moyen   512 x 7 x 8   24,38 M parametres
    40 453 pas, 491 M tokens traverses = le VOLUME d'une epoque du corpus
    20,4 tokens par parametre
    cos, lr 4,5e-3, echauffement 1000, decroissance sur les 10 000 derniers, ecretage 1,0
    5 h 58 sur RTX 5050 portable, 534 ms/pas

**Perte de validation : 1,2382. Perplexité : 3,45.**

Contre **1,4635**, l'ancien record (12,4 M, 30 000 pas, 16/08) : un gain de **0,225**, soit
**dix-sept fois le plancher de bruit**. Pour une fois, aucune précaution de lecture n'est
nécessaire.

| pas | 2 000 | 6 000 | 10 000 | 14 000 | 20 000 | 26 000 | 32 000 | 36 000 | 40 000 |
|---|---|---|---|---|---|---|---|---|---|
| val | 1,9530 | 1,6381 | 1,5763 | 1,4810 | 1,4060 | 1,4122 | 1,3296 | 1,3017 | **1,2382** |

**Aucun plateau à la fin.** Les 4 000 derniers pas valent 0,064, l'un des plus gros gains du
run — c'est la décroissance du cosinus qui agit sur le dernier quart. Deux remontées, à 26 000
et 34 000, toutes deux sous le bruit de mesure de la validation. Le run d'août, lui, plafonnait
sur ses 1 000 derniers pas.

Ce qui a changé depuis août, dans l'ordre d'importance : le corpus complet au lieu de 500 Mo,
un pas d'apprentissage quatre fois et demie plus grand rendu utilisable par l'échauffement et
l'écrêtage, et une taille de modèle doublée à budget de tokens correspondant.

### Ce que la loss vaut comme mesure, revisité sur une plage double

Vingt checkpoints, de 1,95 à 1,24 — là où le run du 16/08 n'offrait que 2,32 à 1,46. Mêmes
15 amorces, mêmes graines à chaque palier.

**Les trois métriques d'échantillonnage restent plates sur toute la plage**, comme la fois
précédente : mots inexistants entre 0 et 0,5 %, répétition entre 0 et 1,5 %, redite entre 0 et
1 %, sans aucune tendance.

*Correction du 23/09 : faux, et pour une raison instructive — les textes ne faisaient que
120 caractères, parce que la génération s'arrêtait au premier saut de ligne. Sur des textes de
340 tokens, les trois métriques corrèlent bel et bien avec la perte. Voir la section suivante.*

**Mais à la lecture, cette fois, le progrès est visible.** Sur la même amorce et la même graine :

    val 1,4810   "I am scared," she said, "Don't worry, Max. We can be friends!"
    val 1,3631   "I am scared," she said, "It's okay, Tim. We can be brave together."
    val 1,2382   "I am scared," she said, "What do we do?"

Aux deux premiers paliers, le modèle **confond qui parle** : le personnage effrayé prononce la
réplique rassurante. C'est le défaut qui subsistait à 1,4635 dans le run d'août et qu'aucune
métrique ne détectait. À 1,2382 il a disparu — la réplique est cohérente avec l'état du
personnage qui la prononce.

Cela tranche entre les deux hypothèses posées le 21/09 sur l'absence de progrès visible entre
1,72 et 1,46 : **le progrès était réel et hors de portée de l'observation**, et non pas confiné
à des tokens sans effet sur le texte. Il fallait descendre plus bas pour le rendre lisible.

Réserve : trois exemples ne font pas une mesure. Le protocole qui trancherait — une lecture en
aveugle où l'on classe des paires sans savoir de quel palier elles viennent — reste à faire.

**Et la limite reste la même** : 109 caractères en moyenne par texte, parce que la génération
s'arrête au premier saut de ligne (`generation.py:151`). La cohérence à longue portée n'est
toujours pas testable.

**Une précision sur « une époque ».** `tireur_de_lot` tire des positions de départ au hasard
dans le corpus (`random.randint`), indépendamment, à chaque pas. Ce n'est donc pas un balayage :
c'est un échantillonnage **avec remise**. Sur 40 453 pas, le volume consommé égale bien la
taille du corpus — 491 M tokens pour 497 M disponibles — mais la couverture réelle vaut
`1 − e⁻¹ ≈ 63 %`. Environ **37 % du corpus n'a jamais été vu**, et une part du reste l'a été
plusieurs fois.

Conséquence : le ratio de 20,4 tokens par paramètre est un ratio de *volume*, pas de matière
distincte. Le modèle a vu l'équivalent d'une époque, pas une époque. Et il reste un tiers du
corpus inexploité — ce qui est une réponse partielle à la question « faut-il agrandir le modèle
ou repasser sur les tokens ».

## Le plafond retrouvé, et la loi de largeur qui tient

Mesuré sur une RTX 3090 (`gpu01`, machine de l'école), `A4-grand` 640 × 8 × 8, 42,28 M de
paramètres, 1 500 pas, cosinus, échauffement 1 000, décroissance 500, écrêtage 1,0 :

| lr à `dim=640` | perte val |
|---|---|
| 3·10⁻³ | **1,9191** |
| 4,5·10⁻³ | 1,9269 |
| 6·10⁻³ | **4,1992 — diverge** |

Les deux premiers sont à égalité — 0,0078, sous le plancher de bruit. Le troisième casse.

**Le seuil d'instabilité à `dim=640` se situe donc entre 4,5 et 6·10⁻³**, là où `dim=512`
supportait encore 6·10⁻³ sans broncher. Le plafond existe bien : il fallait une largeur de plus
pour l'atteindre.

**Et la loi du 15/09 se vérifie quantitativement.** Si le pas maximal stable varie comme
l'inverse de la largeur, passer de 512 à 640 — un facteur 1,25 — doit faire descendre le seuil
de 6·10⁻³ à **4,8·10⁻³**. C'est exactement l'intervalle où la divergence se produit.

| | seuil à `dim=512` | seuil à `dim=640` |
|---|---|---|
| sans échauffement ni écrêtage (15/09) | ~10⁻³ | ~6·10⁻⁴ |
| avec échauffement et écrêtage (21-22/09) | > 6·10⁻³ | 4,5 – 6·10⁻³ |

L'échauffement et l'écrêtage **ne suppriment pas la loi, ils la translatent** d'un facteur
huit à dix. Sa forme — inversement proportionnelle à la largeur — est intacte. C'est un
résultat plus fort que « le plafond a disparu » : il dit que le mécanisme sous-jacent est le
même, et que les garde-fous déplacent le point de rupture sans changer sa dépendance à la
largeur.

**Réglage retenu pour un run long à `A4-grand` : 3·10⁻³.** Même perte que 4,5·10⁻³ et une
marge de deux contre le seuil — ce qui compte sur 68 000 pas, là où les sondes n'en font que
1 500.

### Ce que vaut la machine de l'école

Mesures en régime établi, 1 500 pas, même configuration et même graine que sur le portable :

| | RTX 5050 portable | RTX 3090 `gpu01` | rapport |
|---|---|---|---|
| `A3-moyen` 24,4 M | 547 ms/pas | **253** | ×2,16 |
| `A4-grand` 42,3 M | 939 ms/pas | **329** | ×2,85 |

**Le gain croît avec la taille du modèle** : un petit modèle n'occupe pas une carte de 24 Gio.
Conséquence directe, le coût relatif du gros modèle s'effondre — `A4` coûte ×1,76 le prix de
`A3` sur le portable, mais seulement **×1,30** sur la 3090. À calcul égal, l'optimum de la
courbe en U se déplace donc vers la droite sur cette machine : la conclusion « l'optimum est
autour de 12 M » était une propriété du portable autant que du corpus.

Le run Chinchilla complet de `A4-grand` — 846 M de tokens, 68 815 pas — passe de **17 h 55 à
6 h 18**.

**Et la reproductibilité tient d'une machine à l'autre.** `A3-moyen` à 4,5·10⁻³ donne 1,9453 sur
le portable et **1,9456** sur la 3090, alors que tout diffère : Python 3.11 contre 3.14, torch
2.14 contre 2.13, Ampere contre Blackwell. Trois dix-millièmes d'écart, soit quarante fois sous
le plancher de bruit. Le résultat du 21/09 sur le déterminisme ne valait que pour une même
machine ; il s'étend à des matériels et des versions différents.

**La taille de lot n'est pas un levier ici.** `lot=32` avait été choisi pour tenir dans les
8 Gio du portable ; la 3090 en offre 23,6, donc la question se posait. Mesuré sur `A4-grand`,
500 pas :

| lot | ms/pas | tokens/s |
|---|---|---|
| 32 | 346 | **35 500** |
| 64 | 626 | 39 300 *(+10 %)* |
| 128 | — | manque de mémoire |

Doubler le lot multiplie le temps par pas par 1,81 pour deux fois plus de travail : le débit ne
gagne que 10 %. **La carte était déjà saturée à 32** — il n'y avait pas de parallélisme
inutilisé à récupérer. Ce qui explique aussi, rétrospectivement, pourquoi le rapport contre le
portable plafonne à ×2,85 : la 3090 n'est pas sous-employée, elle fait le même travail plus vite.

À budget de tokens fixé, passer à `lot=64` ferait gagner 38 minutes sur un run de 6 h 30.
Réglage conservé à **32** : changer le lot change le bruit du gradient, donc le pas
d'apprentissage optimal — et tous les seuils de stabilité mesurés aujourd'hui l'ont été à 32.
Rejouer ça pour 10 % serait un mauvais échange.

*Un piège technique, noté parce qu'il aurait coûté des heures : `CUDA_VISIBLE_DEVICES=3` ne
désigne pas la carte 3 de `nvidia-smi`. Le runtime CUDA trie les cartes de la plus rapide à la
plus lente par défaut, `nvidia-smi` les trie par position sur le bus PCI. Il faut joindre
`CUDA_DEVICE_ORDER=PCI_BUS_ID` pour que les deux numérotations coïncident — sans quoi le calcul
part sur une GTX 1660 sans que rien ne le signale.*

## Les métriques ne saturaient pas : les textes étaient trop courts

Quarante checkpoints du run de 120 000 pas, de **1,77 à 1,10** — la plus large plage jamais
mesurée — avec les mêmes 15 amorces et les mêmes graines à chaque palier. Une seule chose change
par rapport aux deux mesures précédentes : les textes font **340 tokens au lieu de ~40**, parce
que la génération ne s'arrête plus au premier saut de ligne.

**Sur l'ensemble de la plage, les trois métriques corrèlent avec la perte :**

| métrique | corrélation avec la perte | p |
|---|---|---|
| mots inexistants | **+0,584** | < 0,0001 |
| répétition | **+0,455** | 0,0017 |
| redite | **+0,400** | 0,0072 |

Les trois sont positives et significatives à n = 40 : quand la perte descend, les trois défauts
reculent. La conclusion des 21 et 22/09 — « elles saturent sous 1,7 et ne mesurent plus rien » —
était donc **un artefact de la longueur des textes**. Sur 120 caractères, il n'y a qu'une
vingtaine de 4-grammes : presque rien à répéter, presque rien à partager. Les métriques tombaient
à zéro par manque de matière, pas par excellence du modèle.

**Mais restreint aux paliers sous 1,30, le tableau change complètement :**

| métrique | corrélation (n = 29) | p |
|---|---|---|
| mots inexistants | +0,386 | 0,030 |
| répétition | −0,058 | 0,76 |
| redite | −0,191 | 0,31 |

**Seule la métrique lexicale garde un signal**, et encore, faible. Les deux métriques
structurelles n'en ont plus aucun : leurs corrélations sont négatives, non significatives,
c'est-à-dire indistinguables de zéro.

**La lecture juste est donc intermédiaire entre les deux erreurs commises.** Les métriques ne
sont pas inutiles — elles suivent la perte tant qu'il reste des défauts grossiers à corriger,
c'est-à-dire au-dessus de 1,3. En dessous, elles cessent de discriminer, et ce n'est plus un
problème de longueur de texte mais de nature : elles comptent des symptômes de surface qui ont
disparu, pendant que ce qui progresse encore — la cohérence, le sens — n'est mesuré par aucune
des trois.

*Deux erreurs de méthode dans cette seule mesure, et elles s'annulent presque. La première :
conclure « elles saturent » sans remarquer que les textes étaient trop courts pour qu'elles aient
quoi que ce soit à mesurer. La seconde : annoncer une tendance nette à mi-parcours, sur les onze
premiers points d'une série bruitée, alors que les vingt-neuf suivants la démentent. Deux façons
opposées de lire une courbe partielle.*
