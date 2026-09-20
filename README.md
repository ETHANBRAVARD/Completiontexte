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

## Résultats en un coup d'œil

État actuel : **transformer 12,4 M de paramètres**, 6 blocs pre-norm, contexte 384,
entraîné sur un corpus TinyStories de 2,2 Go tokenisé maison — perte de validation
**1,4635**, perplexité **4,32**.

> Once upon a time, there was a little boy named Tim. He loved to play with his toys.
> One day, he found a big box in the attic. It was dark and full of old things.

Chaque ligne ci-dessous est une expérience menée pour trancher une question, jamais pour
illustrer une intuition. Le détail — protocole, chiffres bruts, réserves — est dans
**[notes/resultats.md](notes/resultats.md)**.

| Question posée | Ce que la mesure répond |
|---|---|
| Pourquoi 6 blocs n'apprennent-ils pas ? | Le placement des normalisations : post-norm **6,37** → pre-norm **5,02** |
| Quels réglages d'échantillonnage ? | 24 020 textes, 3 métriques : top-p dégénère (95,8 % de redite à `p=0,2`), top-k jamais sous 36 % → défaut `k=5 · T=1,2` |
| Comment encoder 2,2 Go ? | Mémoïsation + flux : 2 h → 5 min, 28 Go → 1,3 Go de RAM, 126 150 131 tokens **identiques entier pour entier** à la version d'août |
| Plus gros = meilleur, à calcul égal ? | Courbe en U, creux apparent à 12 M — **mais** deux configurations à `dim=640` divergeaient |
| Était-ce la capacité ou le pas ? | Le pas : `A4` (42 M) passe de **4,91** à **1,71** en baissant `lr` de 10⁻³ à 5·10⁻⁴ |
| Largeur ou profondeur ? | À pas égal : **égalité** (écarts de 0,007). À calcul égal la largeur gagne, parce qu'elle coûte 617 ms/pas contre 745 |
| Où écrêter le gradient ? | Sa norme **ne dépend pas de la taille** (médianes 0,48–0,64 de 4 à 42 M) → seuil unique **1,0** |
| Ce que l'échauffement change | `A4` : 4,91 → 1,71 → **1,5199**, soit mieux que l'optimum apparent `A2` (1,5228) **à calcul égal** — la courbe en U était creusée par l'instabilité |
| Quel schedule ? | Cosinus à 2·10⁻³ → **1,4912** ; le pas constant explose à ce pas de base |
| Combien de bits les poids portent-ils ? | **8,3 bits** en virgule fixe, **4,0 bits** de mantisse. La précision mixte ne rapporterait que 8 % |
| Où le modèle est-il fragile ? | Par rôle, pas par volume : `W_2` tolère 1,09 % de bruit, `W_q` 5,63 % — à forme identique |

Question ouverte à ce jour : la décroissance du pas ne couvre que **7 %** du run, là où
les entraînements publiés la font commencer juste après l'échauffement.

## Structure

```
src/model/      🔴 bigram, rnn, lstm, transformer, encodeur/decodeur BPE, tireur_de_lot,
                   generation (greedy/top-k/top-p), Vocabulaire, analyse_generation
src/tooling/    🟢 tracer.py — tracés et mesures ; quatre vérificateurs de la chaîne
                   d'encodage (découpage, tokenisation d'un mot, encodeur entier,
                   fichiers de tokens entiers)
scripts/        🟢 préparation des corpus, encodage à grande échelle, splits
notes/resultats.md le compte rendu détaillé de chaque expérience
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
