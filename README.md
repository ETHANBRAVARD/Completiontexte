# Completiontexte

> *README temporaire, coécrit avec Claude.*

Un modèle de langage écrit de zéro, de la table de comptage bigramme au transformer
décodeur de 42 millions de paramètres entraîné sur 1,5 milliard de tokens — sans
`transformers`, sans `tokenizers`, sans copier-coller.

L'objectif n'est pas d'obtenir le meilleur modèle : c'est de comprendre chaque
mécanisme en l'écrivant. Tout ce qui apprend, prédit ou calcule un gradient a été écrit
à la main.

## La contrainte de méthode

Le projet est mené sous une règle explicite, formalisée dans [CLAUDE.md](CLAUDE.md) :
le dépôt est découpé en **zone rouge** et **zone verte**.

| Zone | Contenu | Qui écrit |
|---|---|---|
| 🔴 `src/model/` | architecture, forward, backward, perte, optimiseur, tokenizer, échantillonnage | moi, exclusivement |
| 🟢 `src/tooling/`, `scripts/` | préparation des corpus, I/O, CLI, logs, courbes, vérificateurs | assistance IA autorisée |

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
| 5 | Tokenizer BPE + passage à l'échelle | BPE, entraînement long, lots | ✅ 23/09/2026 |
| 6 | Transformer contre modèle d'espace d'états | récurrence linéaire, balayage parallèle | 🔭 piste — avec RoPE / ALiBi, les encodages de position qui extrapolent |
| 7 | Raisonnement : en mots contre latent, sur tâche synthétique | rebouclage de l'état caché, corpus généré | 🔭 piste |
| 8 | Substrat analogique : quantification, bruit, crossbar | quantification des poids, injection de bruit | 🔄 ouverte le 14/09 — bruit mesuré, quantification à écrire |

Les étapes 6 à 8 sont des **branches, pas une suite** — elles s'ouvrent une fois que la 5
tourne, dans l'ordre qu'on veut.

La **6** est la moins chère : remplacer l'attention par un SSM ne touche qu'une couche,
tout le reste du fichier est identique, donc la comparaison n'a qu'une variable. La **7**
exige un corpus nouveau — TinyStories ne contient aucune étape intermédiaire à raisonner ;
il se génère, donc il est gratuit. La **8** est latérale : elle enseigne le substrat de
calcul plutôt que les modèles de langue, et sa première question se règle entièrement en
logiciel — mes poids survivent-ils à 6 bits bruités ? C'est la branche ouverte aujourd'hui.

## Résultats en un coup d'œil

État actuel : **transformer 42,3 M de paramètres**, 8 blocs pre-norm de largeur 640,
8 têtes, contexte 384, vocabulaire BPE de 2 080 tokens, entraîné sur un corpus TinyStories
de 2,2 Go tokenisé maison (497 M tokens). 120 000 pas, 1,47 milliard de tokens traversés,
soit **trois passages sur le corpus**. Perte de validation **1,1433**, perplexité **3,14**,
sans surapprentissage visible. Dix heures sur une RTX 3090 de l'école ; le modèle
précédent, 24,4 M sur une seule époque, atteignait 1,2382 en six heures sur mon portable.

> *Tim and his dog were playing in the garden when* they heard a loud noise. It was a big
> truck! Tim and the dog were scared. They ran to Tim's house to hide. When they were
> inside, Tim's mom said, "Don't worry, the truck is just doing its job. We'll get some
> water to wash away the dirt." Tim was happy his mom was not mad.

*Amorce en italique, échantillonnage top-k 5 à température 1,2. La grammaire et le
dialogue tiennent ; la causalité reste fragile.*

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
| Ce que l'échauffement change | `A4` : 4,91 → 1,71 → **1,5199** — à calcul égal il **égale** l'optimum apparent `A2` (1,5228). La branche droite de la courbe en U était creusée par l'instabilité, pas par le budget de tokens |
| Quel schedule ? | Cosinus à 2·10⁻³ → **1,4912** ; le pas constant explose à ce pas de base. Cosinus contre racine : écart sous le bruit, non tranché |
| Que donne l'étape 5 menée au bout ? | 24,4 M de paramètres, le volume d'une époque du corpus complet, 6 h : **1,2382** contre 1,4635 pour l'ancien record. Un écart de 0,225, soit **dix-sept fois le plancher de bruit** |
| Et 42 M sur trois passages du corpus ? | **1,1433**, en 10 h sur la 3090 de l'école. La validation ne remonte pas durablement : à trois passages, la répétition ne produit pas de surapprentissage visible |
| Quand démarrer la décroissance ? | **Ça ne change rien de mesurable** : étendue 0,018 pour un bruit de 0,013 |
| Deux runs identiques, à quel point diffèrent-ils ? | **0,013** de perte de validation. C'est le plancher sous lequel une comparaison ne veut rien dire — et il invalide deux conclusions écrites ici en septembre |
| Jusqu'où monter le pas d'apprentissage ? | L'échauffement et l'écrêtage le font passer de **5·10⁻⁴ à ~5·10⁻³**, un facteur dix. Le plafond existe toujours — il se mesure à `dim=640`, entre 4,5 et 6·10⁻³ |
| La loi « seuil ∝ 1/largeur » survit-elle à l'échauffement ? | **Oui, translatée.** 512 → 640 fait ×1,25 ; le seuil prédit passe de >6·10⁻³ à 4,8·10⁻³, et la divergence tombe exactement dans cet intervalle |
| Les métriques de texte suivent-elles la perte ? | **Oui au-dessus de 1,3** (r = 0,40 à 0,58 sur 40 paliers), **non en dessous** : seule la métrique lexicale garde un signal. Les avoir crues saturées était un artefact — les textes ne faisaient que 120 caractères |
| Un lot plus grand irait-il plus vite ? | **Non** : 32 → 64 ne gagne que 10 % de débit, 128 manque de mémoire. La carte est déjà saturée à 32 |
| D'où vient ce bruit ? | **Entièrement de la graine.** À graine fixée, deux runs relancés à 36 h d'intervalle donnent des journaux au diff vide : le non-déterminisme du GPU n'y contribue rien de mesurable |
| Combien de bits les poids portent-ils ? | Sur un modèle de 12 M : **8,3 bits** en virgule fixe, **4,0 bits** de mantisse. La précision mixte ne rapporterait que 8 % |
| Où le modèle est-il fragile ? | *(16/09, remis en question le 03/10)* Par rôle, pas par volume : `W_2` tolère 1,09 % de bruit, `W_q` 5,63 % |
| Le modèle final est-il plus robuste ? | **Selon la loi de bruit.** Au bruit multiplicatif, ni plus ni moins qu'à 24 M (seuil 6,6 %). Au bruit additif, nettement plus fragile : +0,48 de perte à 2 % de bruit contre +0,09, soit 9,1 bits au lieu de 8,2. Ses trois familles les plus fragiles sont exactement ses trois familles à **queues lourdes** (max/σ de 17 à 27) : sensibilité propre, ou plage gonflée par quelques poids extrêmes ? Question ouverte, que la quantification doit trancher |

Le résultat le plus utile n'est pas une ligne du tableau, c'est son étalon : **deux entraînements
strictement identiques, à la graine près, diffèrent de 0,013**. Tout le mois de septembre a
comparé des configurations sans ce chiffre, en traitant des écarts de 0,02 comme des résultats.
Deux d'entre eux n'en étaient pas.

## Structure

```
src/model/        🔴 bigram, réseau NumPy à la main (poid), même réseau en PyTorch
                     (Reseau_torch), rnn, lstm, transformer, BPE (bpe_stories, bpe_liste,
                     encodeur, decodeur), tireur_de_lot, generation (greedy/top-k/top-p),
                     Vocabulaire, analyse_generation
  analogique/     🔴 étape 8 — bruit multiplicatif et additif sur les poids
  espace_etats/   🔴 étape 6 — pas commencée
  raisonnement/   🔴 étape 7 — pas commencée
  archives/          code abandonné, gardé pour mémoire
src/tooling/      🟢 tracés ; vérificateurs de la chaîne d'encodage (découpage, tokenisation
                     d'un mot, encodeur entier, fichiers de tokens) ; inspection, allègement
                     et vérification des checkpoints ; répartition des poids
scripts/          🟢 préparation des corpus, tokenizer, encodage à grande échelle, bancs
                     d'échelle et de texte, sonde de gradient
  analogique/     🟢 banc de bruit (étape 8)
  nuits/          🟢 programmes de nuit, archivés tels qu'ils ont tourné
notes/resultats.md   le compte rendu détaillé de chaque expérience
notes/biblio.md      bibliographie annotée
notes/questionnaire-reponses.md   bilan en 112 questions, et mes réponses
JOURNAL.md           journal d'apprentissage, un compte rendu par étape
runs/, data/         checkpoints, logs, courbes et corpus (non versionnés)
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

# 6. étape 8 — bruit sur un checkpoint : global, puis une famille de matrices à la fois
python3 scripts/analogique/banc_bruit.py --checkpoint runs/<run>/<checkpoint>.pt
python3 scripts/analogique/banc_bruit.py --checkpoint runs/<run>/<checkpoint>.pt --par-famille
python3 src/tooling/repartition_poids.py --checkpoint runs/<run>/<checkpoint>.pt
```

Les étapes 3, 4 et 5 ne prennent aucun argument : elles retrouvent seules le dossier le
plus récent. C'est délibéré — un chemin recopié à la main est le moyen le plus sûr
d'analyser l'ancien modèle en croyant analyser le nouveau, sans qu'aucune erreur ne se
lève.

Les hyperparamètres sont en tête de `src/model/transformer.py`. Ordres de grandeur : un
run de 30 000 pas à 6 blocs occupe 3,3 Go de VRAM et 3 h 30 sur le portable ; le run final
(42 M, 120 000 pas) a pris 10 h sur une RTX 3090.

Dépendances : `torch` et `numpy` uniquement. `transformers`, `tokenizers` et `keras`
sont volontairement proscrits — ils contiennent précisément ce que le projet consiste à
écrire.

Environnement de développement : RTX 5050 Laptop 8 Go, PyTorch 2.13 + CUDA 13,
Python 3.14. Runs longs sur une RTX 3090 24 Go de l'école (PyTorch 2.14, Python 3.11) :
à configuration et graine égales, les deux machines donnent la même perte à 3·10⁻⁴ près.

## Journal

[JOURNAL.md](JOURNAL.md) contient un compte rendu par étape, rédigé de mémoire sans
relire le code — le test étant que ce que je ne sais pas réexpliquer, je ne l'ai pas
encore appris. On y trouve aussi les blocages réels : la migration vers les lots et le
décalage de tous les axes positifs, `cross_entropy` et son axe des classes, le
`.to(device)` qui transforme un paramètre en tenseur non-feuille et fait planter Adam,
la pré-tokenisation reprise six fois avant que le curseur avance du bon nombre de
caractères, et les quatre versions successives du pre-norm — dont trois déplaçaient la
normalisation au bon endroit du fichier mais sur le mauvais argument.
