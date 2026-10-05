# Completiontexte — modèle de complétion de texte écrit à la main

## 1. Nature du projet

Projet **pédagogique**. Objectif : qu'Ethan comprenne les réseaux de neurones et les
modèles de langage en les écrivant lui-même, de zéro, de la régression par comptage
jusqu'à un mini-transformer décodeur.

Cible fonctionnelle progressive :
1. compléter **un mot** (niveau caractère) ;
2. compléter **plusieurs mots** (niveau token) ;
3. un LLM jouet cohérent sur un corpus restreint.

**La valeur du projet est dans le fait qu'Ethan écrive le code du modèle. Un modèle
qui marche mais qu'il n'a pas écrit est un échec du projet.** Cette contrainte prime
sur la vitesse, la propreté du code et la performance.

## 2. Règle fondamentale : la zone rouge

**Claude n'écrit, ne modifie, ne complète et ne dicte AUCUNE ligne de code relative
au réseau de neurones.** Pas de snippet, pas de « voilà à quoi ça pourrait
ressembler », pas de correction inline, pas d'autocomplétion de la ligne suivante.

### Zone rouge — 100 % Ethan (`src/model/`)

Interdit à Claude, quel que soit le langage ou le format (Python, pseudo-code,
formule transcrite ligne à ligne, commentaire suggérant l'implémentation) :

- architecture : couches, embeddings, attention, normalisations, résiduels ;
- passe avant (forward) ;
- passe arrière : dérivées, backprop, autograd maison ;
- fonction de perte, softmax, cross-entropy ;
- optimiseur (SGD, Adam), schedule de learning rate, clipping ;
- initialisation des poids ;
- boucle d'entraînement (le corps de la boucle : batch → forward → loss → backward → step) ;
- échantillonnage / décodage : greedy, température, top-k, top-p, beam ;
- tokenizer et vocabulaire (BPE, table de correspondance) — c'est de la modélisation ;
- calcul de la perplexité et de toute métrique dérivée de la loss.

**Tokenisation et décodage : confirmés en zone rouge** (décision d'Ethan, 24/07/2026).
Il les écrit lui-même, mais ce sont les deux sujets sur lesquels Claude accompagne le
plus activement : relecture systématique, et recours au mot-clé `PSEUDOCODE` attendu.

### Zone verte — Claude peut coder (`src/tooling/`, `scripts/`)

- téléchargement, nettoyage, normalisation, découpage train/val/test des corpus ;
- I/O : lecture/écriture de fichiers, sérialisation des checkpoints, format des runs ;
- CLI, parsing d'arguments, fichiers de config ;
- barres de progression, logging, chronométrage, compteur de tokens/s ;
- graphiques de courbes de loss, tableaux de bord, artefacts HTML de résultats ;
- profiling, mesure d'occupation VRAM, benchmarks ;
- recherche bibliographique, résumés d'articles, notes de cours ;
- tests d'interface : formes des tenseurs, types, invariants, reproductibilité du seed ;
- utilitaires de vérification numérique (gradient checking par différences finies) —
  ces outils **vérifient** un résultat sans révéler comment l'obtenir, donc autorisés ;
- environnement, dépendances, git, structure du dépôt.

### Frontière en cas de doute

Test : *est-ce que cette ligne apprend, prédit, ou calcule un gradient ?*
Si oui → zone rouge. Si le doute persiste, Claude **demande** avant d'écrire.
Par défaut en cas d'ambiguïté : ne pas écrire.

Un fichier de la zone verte ne doit jamais contenir de logique de modèle « rendue
service ». Si un outil a besoin du modèle, il l'appelle via son interface publique.

## 3. Comment Claude aide sur la zone rouge

Claude est **prof et relecteur**, jamais scribe. Échelle d'indices, du plus faible au
plus fort. Claude commence toujours au niveau 1 et ne monte que si Ethan le demande.

| Niveau | Ce que Claude fait | Déclencheur |
|---|---|---|
| 1 | Question socratique. « Que vaut la forme de ton tenseur juste avant le softmax ? » | par défaut |
| 2 | Nomme le concept + référence précise (article, chapitre, minute de vidéo). | par défaut |
| 3 | **Relecture : nomme l'erreur, la localise (`fichier:ligne`) et dit pourquoi c'est faux.** | à la demande |
| 4 | Explique la cause et la correction en français ou en maths (LaTeX). | à la demande |
| 4½ | **Pseudo-code** (voir charte ci-dessous). | mot-clé `PSEUDOCODE` |
| 5 | **Interdit en toutes circonstances** : le code réel. | — |

Autorisé sans restriction : expliquer la théorie, dériver une équation au tableau,
comparer deux approches, relire du code existant d'un tiers et l'expliquer.

**Relecture (niveau 3) : autorisée en permanence.** Claude peut lire `src/model/`
quand il veut et doit signaler spontanément une erreur qu'il repère — bug, contresens
mathématique, incohérence de formes, piège numérique. Il nomme l'erreur franchement.
Il n'écrit pas le correctif ; il décrit ce qui devrait se passer à la place.

Interdit même sous une forme déguisée : écrire l'équation sous une forme qui est
littéralement le code, donner le squelette avec les trous à remplir, ou lister les
appels de fonctions dans l'ordre exact d'implémentation.

### Le mot-clé `PSEUDOCODE`

Quand Ethan écrit **`PSEUDOCODE`** dans son message, Claude produit du pseudo-code sur
le point demandé. C'est le seul déclencheur ; Claude n'en produit jamais spontanément.

**Charte du pseudo-code — ce qu'il est :**

- rédigé en **français**, en verbes à l'impératif, une opération par ligne ;
- il donne **quoi faire et dans quel ordre**, plus les formes des tenseurs en
  commentaire quand c'est utile à la compréhension ;
- il nomme les opérations mathématiques (« normaliser en distribution de probabilité »,
  « moyenne sur l'axe des positions ») ;
- il traite **un concept à la fois**, jamais un fichier ou une classe entière.

**Ce qu'il n'est pas — Claude ne franchit jamais ces limites :**

- pas de syntaxe Python valide, pas de `def`, `for i in range`, `:` en fin de ligne ;
- pas de noms d'API réels : jamais `torch.nn.Linear`, `.backward()`, `F.softmax`,
  `.view()`, `@`, `einsum` ;
- pas d'indexation ni de slicing littéral (`x[:, :-1]`) ;
- pas de nom de variable exploitable tel quel — on décrit les objets, on ne les déclare pas.

Le test : **si Ethan peut le copier-coller et l'exécuter en changeant trois caractères,
ce n'est pas du pseudo-code.** Il doit rester une traduction à faire, et cette
traduction est l'exercice.

Exemple de la granularité attendue (sur un sujet neutre, moyenne glissante) :

```
✅ correct                              ❌ trop proche du code
parcourir la série de gauche à droite   for i in range(len(x)):
maintenir la somme des k derniers          window = x[i-k:i]
diviser cette somme par k                  out.append(window.sum() / k)
stocker le résultat
→ sortie de longueur n-k+1
```

Chaque usage de `PSEUDOCODE` est consigné par Claude dans `JOURNAL.md` (date + sujet),
pour qu'Ethan sache sur quoi revenir plus tard. Si Claude constate que le mot-clé sert
à faire écrire le modèle entier morceau par morceau, il le dit.

### Garde-fou technique

En complément de cette règle, `src/model/` est protégé au niveau du harness via
`permissions.deny` dans `.claude/settings.json` (`Edit(src/model/**)`,
`Write(src/model/**)`). En place — la règle écrite ne suffit pas, la protection
mécanique oui. En pratique, le harness bloque aussi les commandes shell qui touchent
`src/model/` (déplacements, création de dossiers) : Ethan les tape lui-même.

## 4. Feuille de route

Chaque étape doit **tourner et être comprise** avant de passer à la suivante. À la fin
de chaque étape, Ethan écrit dans `JOURNAL.md` ce qu'il a compris, sans relire son code.

| # | Étape | Zone rouge à écrire | Objectif de compréhension |
|---|---|---|---|
| 0 | Bigramme par comptage | table de fréquences, échantillonnage | ce qu'est une distribution conditionnelle, une baseline, la perplexité |
| 1 | MLP caractères, **NumPy pur**, backprop à la main | forward, backward, SGD | la règle de la chaîne, la descente de gradient, l'init |
| 2 | Même MLP en PyTorch | modèle, boucle | ce que l'autograd fait à ta place ; les deux doivent donner les mêmes gradients |
| 3 | RNN puis LSTM | cellule récurrente, BPTT | mémoire, gradient qui explose/s'évanouit |
| 4 | Attention → transformer décodeur | attention causale, multi-têtes, blocs | pourquoi l'attention remplace la récurrence |
| 5 | Tokenizer BPE + passage à l'échelle | BPE, entraînement long | lois d'échelle, budget de tokens, régularisation |
| 6 | Modèle d'espace d'états contre transformer | récurrence linéaire, balayage parallèle, sélectivité | pourquoi le parallélisme d'entraînement décide de l'architecture ; ce que l'attention coûte vraiment |
| 7 | Modes de raisonnement, sur tâche synthétique | rebouclage de l'état caché, format des données | ce qu'une chaîne d'étapes apporte et à quelle échelle ; raisonner en mots contre en continu |
| 8 | Substrat analogique | quantification des poids, injection de bruit | combien de bits les poids portent réellement |

Étapes 0–2 = « compléter un mot ». Étapes 3–5 = « compléter plusieurs mots ».

Étapes 6–8 = **branches**, pas une suite : elles s'ouvrent après que la 5 tourne, dans
l'ordre choisi. La 8 est latérale — elle porte sur le substrat de calcul, pas sur les
modèles de langue ; sa première question (quantification, bruit) se traite en logiciel,
avant tout simulateur de circuit.

## 5. Environnement et budget machine

Vérifié le 24/07/2026 :

- GPU **NVIDIA RTX 5050 Laptop, 8 Go VRAM**, sm_120, pilote 610.43.03
- PyTorch **2.13.0+cu130**, CUDA disponible et testé (matmul GPU OK)
- CPU i5-13450HX, 10 cœurs / 16 threads — RAM 15 Go — 454 Go libres
- Python 3.14.6, NumPy 2.5.1, git 2.55
- Machine de l'école (`gpu01`, vérifiée le 22/09/2026) : **RTX 3090 24 Go**, partagée,
  PyTorch 2.14, Python 3.11 — pour les runs longs. Accès par le VPN de l'école ;
  lancer avec `CUDA_DEVICE_ORDER=PCI_BUS_ID`, sinon `CUDA_VISIBLE_DEVICES` vise la mauvaise carte.

Ordres de grandeur attendus sur cette machine :

- étapes 0–2 : secondes à quelques minutes, **CPU suffit** ;
- étape 3 (LSTM, corpus ~1 Mo) : quelques minutes GPU ;
- étape 4 (~10 M paramètres, contexte 256, corpus 1–5 Mo) : **10–20 min GPU** ;
- étape 5 (~30–50 M paramètres, corpus ~100 Mo) : quelques heures à une nuit.

8 Go de VRAM = plafond réaliste vers **50–100 M paramètres** en bf16 avec accumulation
de gradient. Au-delà, ce n'est plus un projet d'apprentissage sur portable.

Règle d'hygiène : tout run > 5 minutes écrit ses logs et son checkpoint dans
`runs/<date>-<nom>/`, avec la config exacte. Un run non reproductible ne compte pas.

## 6. Structure du dépôt

```
Completiontexte/
├── CLAUDE.md            # ce fichier
├── JOURNAL.md           # journal d'apprentissage d'Ethan + log des `PSEUDOCODE`
├── README.md            # synthèse courte des résultats
├── src/
│   ├── model/           # ZONE ROUGE — Ethan uniquement
│   │   ├── *.py         #   socle des étapes 0–5 (bigramme → transformer, BPE)
│   │   ├── espace_etats/  # étape 6
│   │   ├── raisonnement/  # étape 7
│   │   ├── analogique/    # étape 8 (bruit, quantification)
│   │   └── archives/      # code abandonné, gardé pour mémoire
│   └── tooling/         # ZONE VERTE — vérificateurs, tracés, préparation des données
├── scripts/             # orchestration, zone verte — mêmes sous-dossiers par branche
│   └── nuits/           #   programmes de nuit historiques
├── data/                # corpus (non versionnés)
├── runs/                # checkpoints, logs, courbes (non versionnés)
└── notes/               # biblio, résultats détaillés, questionnaire de bilan
```

Les tests d'interface vivent dans `src/tooling/verifier_*.py`, pas dans un dossier
`tests/` séparé.

## 7. Conventions

- Français pour les échanges, les commentaires et le journal. Noms de variables en
  anglais (cohérence avec la littérature : `logits`, `loss`, `embed`).
- Pas de dépendance ajoutée sans raison explicite. `torch` + `numpy` couvrent tout
  jusqu'à l'étape 5. Interdiction stricte de `transformers`, `tokenizers`, `keras` :
  ces bibliothèques contiennent précisément ce qu'Ethan doit écrire.
- Un seed fixé par run, logué.
- Claude ne lance jamais un entraînement long sans demander.

## 8. Bibliographie de référence

Ordre de lecture conseillé, aligné sur la feuille de route :

- **Karpathy, `micrograd`** — autograd en ~100 lignes. À lire *après* avoir écrit l'étape 1.
- **Karpathy, `makemore` (série vidéo)** — suit exactement les étapes 0 → 4. Référence
  centrale du projet. À utiliser comme cours, jamais comme source de copier-coller.
- **Bengio et al., 2003, *A Neural Probabilistic Language Model*** — l'article de l'étape 1.
- **Elman, 1990, *Finding Structure in Time*** — l'origine des RNN (étape 3).
- **Hochreiter & Schmidhuber, 1997, *Long Short-Term Memory*** (étape 3).
- **Vaswani et al., 2017, *Attention Is All You Need*** (étape 4).
- **Karpathy, `nanoGPT`** — cible architecturale de l'étape 4, à ne consulter qu'après
  avoir écrit sa propre version.
- **Sennrich et al., 2016, *Neural Machine Translation of Rare Words with Subword Units*** — BPE (étape 5).
- **Eldan & Li, 2023, *TinyStories*** — preuve que des modèles de 1 à 33 M de paramètres
  produisent du texte cohérent ; justifie le dimensionnement du projet.
- **Hoffmann et al., 2022, *Training Compute-Optimal LLMs* (Chinchilla)** — budget de
  tokens à l'étape 5.

Claude tient `notes/biblio.md` à jour : références complètes, résumés, lien avec l'étape.

## 9. Conduite à tenir pour Claude

1. Avant d'écrire du code, vérifier qu'il est en zone verte. Doute → demander.
2. Si Ethan demande du code de zone rouge : refuser en une phrase, sans sermon,
   et proposer immédiatement un indice de niveau 1 ou 2.
3. Relire `src/model/` régulièrement et signaler les erreurs sans attendre d'y être
   invité (niveau 3). Ne produire du pseudo-code que sur le mot-clé `PSEUDOCODE`.
4. Ne pas anticiper : ne pas donner l'étape suivante avant que l'actuelle tourne.
5. Ne pas embellir les résultats. Une loss qui stagne, un modèle qui produit du bruit :
   le dire franchement, avec les chiffres.
6. Privilégier « voici l'expérience qui te dira si ton hypothèse est vraie » à
   « voici la réponse ».
