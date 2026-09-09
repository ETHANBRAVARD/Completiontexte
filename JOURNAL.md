# Journal d'apprentissage

Deux usages, deux sections.

**Comptes rendus d'étape** — à la fin de chaque étape de la feuille de route, tu écris
ce que tu as compris **sans relire ton code**. C'est le test : ce que tu ne sais pas
réexpliquer de mémoire, tu ne l'as pas encore appris. Un compte rendu court et honnête
vaut mieux qu'un résumé complet recopié.

**Log des `PSEUDOCODE`** — tenu par Claude. Chaque recours au mot-clé y est daté, avec
le sujet. C'est la liste des points que tu n'as pas su écrire seul du premier coup :
ce sont exactement ceux à réattaquer plus tard, à froid, sans indice.

---

## Comptes rendus d'étape

Gabarit à recopier pour chaque étape :

```
### Étape N — <titre>            (JJ/MM/AAAA)

**Ce qui tourne**
<en une phrase : ce que le code fait, et le chiffre qui le prouve — loss finale,
perplexité, exemple de sortie>

**Ce que j'ai compris**
<de mémoire, code fermé>

**Ce qui m'a bloqué**
<le bug ou le contresens, et ce qui l'a débloqué>

**Ce que je ne comprends pas encore**
<à assumer par écrit — c'est la partie la plus utile du journal>
```

---

### Étape 0 — Bigramme par comptage            (terminée le 26/07/2026)

> Log factuel tenu par Claude, à la demande d'Ethan. Les deux rubriques de
> compréhension sont laissées volontairement vides : elles sont à Ethan, de mémoire.

**Ce qui tourne**

Bigramme au niveau caractère sur `data/names.txt` (32 032 prénoms), écrit en Python
pur, sans dépendance. Trois briques, toutes fonctionnelles :

- `frequence` — table de comptage 27×27 (26 lettres + un symbole de frontière `' '`),
  lissage add-α (α = 0,001), normalisation par ligne en distributions conditionnelles ;
- `lossmodel` — log-vraisemblance négative moyenne par transition sur tout le corpus :
  **2,45** (baseline uniforme ln 27 ≈ 3,296 ; perplexité $e^{2,45} \approx 11{,}6$).
  Conforme à la référence publiée (makemore, ~2,45) ;
- `inventeprenom` — échantillonnage pondéré via `random.choices`, boucle de génération
  qui part de la frontière et s'arrête sur la frontière. Sorties typiques :
  `orayrin`, `leryxa`, `jarrin` — plausibles par morceaux, incohérentes sur la longueur.

**Ce qui m'a bloqué** *(observé pendant la session)*

- la gestion du **symbole de frontière** : le compter en fin de prénom *et* le
  réinitialiser entre deux prénoms (plusieurs itérations avant que le comptage soit juste) ;
- `math.log(0)` sur un bigramme jamais vu → a rendu le **lissage** obligatoire, pas optionnel ;
- confusion loss / probabilité moyenne (log oublié un instant), et moyenne des moyennes
  vs moyenne pondérée par transition ;
- `random.choices` appelé sans `weights` → tirage uniforme silencieux, tout le modèle ignoré.

**Ce que je ne comprends pas encore**

**Ce que j'ai compris**

- **La table de comptage** : au départ, c'est juste un comptage — combien de fois la
  lettre a suit la lettre b. Une fois normalisée, chaque ligne devient la probabilité
  d'une lettre sachant la lettre précédente.
- **Distribution conditionnelle** : conditionnée à la lettre précédente. Un modèle qui
  ignore la lettre précédente perd toute information de contexte : il ne se base plus
  que sur la fréquence globale d'une lettre. Par exemple, après deux e (`ee`), il
  mettrait encore e en priorité, alors que `eee` n'est jamais pertinent.
- **La baseline** sert de comparatif : elle permet de voir si les systèmes suivants sont
  plus ou moins cohérents, et si les réseaux sont utiles ou non dans cet exercice.
- **La perplexité** = exp(loss) : le nombre de lettres entre lesquelles le modèle hésite
  en moyenne. Dans un cas d'aléatoire pur, chaque lettre a une chance sur 27, ce qui
  donne exp(−log(1/27)) = 27. Trajectoire du projet : 27 (hasard) → 11,6 (bigramme)
  → ~7,8 (transformer).


**Fichier** : `Completion_de_texte.py` (racine). À renommer et déplacer dans `src/model/`
avant l'étape 1.

---

### Étape 1 — MLP caractères, NumPy pur, backprop à la main   (terminée le 29/07/2026)

> Log factuel tenu par Claude. Rubriques de compréhension laissées à Ethan.

**Ce qui tourne** (`src/model/poid.py`)

Deux modèles construits de zéro, sans autograd :

- **Bigramme neuronal** (une couche linéaire, one-hot → W → softmax) : backprop à la
  main, SGD, loss **2,45** — égale le bigramme par comptage, mais appris. Gradient
  vérifié contre les différences finies (écart ~1e-11).
- **MLP** (Bengio) : contexte K=3, embeddings (dim 2), couche cachée + `tanh`, sortie
  softmax. Backprop **manuelle à travers les deux couches** — produit extérieur pour les
  gradients de poids, transposée pour propager, terme-à-terme pour le `tanh`, découpe de
  `∂E` vers les lignes de l'embedding. **Les cinq gradients (`∂C, ∂W1, ∂b1, ∂W2, ∂b2`)
  vérifiés contre le numérique à ~1e-11.** Entraîné sur `names.train.txt`, loss de
  validation **~2,47** (pas=0,01). Baseline aléatoire ln 27 ≈ 3,30, unigramme ~2,83.


**Ce qui m'a bloqué** *(observé pendant la session)*

- distinction **contexte / cible** (récurrente, du bigramme jusqu'au `∂C`) ;
- trois opérations à ne pas confondre : produit **extérieur** (gradient de poids),
  produit **matriciel** avec transposée (propager), **terme-à-terme** (traverser tanh) ;
- paramètres **appris** vs **recréés** : embedding régénéré à chaque appel, mesure de
  loss qui reconstruisait des poids aléatoires (bugs longs à voir) ;
- pièges Python/NumPy : `liste += tableau`, curseur de fichier épuisé (`seek(0)`),
  fenêtre de contexte non glissée à l'entraînement (→ modèle figé sur la marginale) ;
- réglage du pas : 0,5 diverge, 0,05 oscille, 0,01 descend proprement.

**Ce que je ne comprends pas encore**

**Ce que j'ai compris**

- **Règle de la chaîne** : on remonte de la fin vers le début, car l'information que
  l'on veut ajuster — la loss — se trouve à la fin ; c'est donc d'elle qu'on part.
  En partant du début, on n'aurait aucun regard sur le résultat final. En remontant,
  chaque gradient calculé se réutilise pour toutes les couches en amont.
- **Gradient par rapport aux logits** : c'est la sortie du softmax avec −1 sur la case
  cible. Ça donne la direction de correction : le logit de la bonne lettre monte,
  les autres descendent à hauteur de la probabilité qu'elles avaient prise.
- **Initialisation à zéro** : impossible. Si les poids sont identiques, les neurones
  subissent la même rétroaction, on ne pourra jamais les distinguer, ils resteront
  toujours égaux — aucun apprentissage possible. L'init aléatoire sert à casser
  cette symétrie.
- **La table d'embedding C** donne une direction à chaque lettre : elle oriente le
  vecteur de chaque lettre dans l'espace en fonction de son « sens » — elle pourrait
  par exemple dire que e est une voyelle. Contrairement au one-hot, où toutes les
  lettres sont à égale distance et où cette géométrie est figée, celle de C s'apprend.


**À explorer plus tard** : dim d'embedding plus grande (2 bride la capacité — le MLP
n'exploite pas encore vraiment ses 3 lettres de contexte), plus d'époques, corpus complet.

---

### Étape 2 — même MLP en PyTorch (autograd)   (terminée le 29/07/2026)

> Log factuel tenu par Claude. Rubriques de compréhension laissées à Ethan.

**Ce qui tourne** (`src/model/Reseau_torch.py`)

Le même MLP qu'à l'étape 1, réécrit avec des **tenseurs torch** (`requires_grad=True`).
Passe avant écrite à la main ; passe arrière déléguée à **`loss.backward()`**.

- **Résultat central de l'étape** : sur mêmes paramètres et même exemple, les gradients
  de l'autograd (`.grad`) et ceux dérivés à la main à l'étape 1 coïncident à **~1e-17**
  (bruit d'arrondi) — bien plus serré que les différences finies (~1e-11), parce que les
  deux calculent *la même formule analytique*. L'autograd = la règle de la chaîne d'Ethan,
  exécutée en suivant le graphe des opérations.
- Boucle d'entraînement torch : loss d'entraînement **~2,37**, **sous le bigramme (2,45)**
  — le contexte de 3 lettres paie enfin. (À confirmer sur validation.)

**Ce que j'ai compris**

- **Ce que fait l'autograd** : PyTorch enregistre toutes les étapes de calcul pendant
  le forward ; au `.backward()`, il les remonte pour calculer les gradients. Il
  remplace la partie « dérivation » que j'avais écrite à la main à l'étape 1 — mais
  il n'ajuste rien : la mise à jour des poids reste mon code (jusqu'à mon Adam maison).
- **Remise à zéro des gradients** : dans torch, les gradients s'accumulent à chaque
  itération. Sans remise à zéro, les corrections se referaient à chaque fois, en
  s'additionnant aux anciennes.

**Ce que je ne comprends pas encore / à réviser**

- Le **résultat central de l'étape 2** : je ne l'avais plus en tête au contrôle du
  30/07/2026. Il s'agissait de comparer les gradients de ma backprop NumPy et ceux
  de l'autograd, sur mêmes poids et même exemple — écart ~1e-17, donc ma dérivation
  à la main était exacte. (Je l'avais confondu avec la recherche du bon pas.)
- La **perplexité** : réflexe inversé au contrôle (j'ai répondu 1/27 au lieu de 27).
  À refaire de tête : perplexité = exp(loss).



- API torch : `nn.Softmax(x)` (construit un module) vs `torch.softmax(x, dim=0)` ;
  `torch.tensor(2)` (un scalaire) vs `torch.randn(forme)` ;
- **placement de `torch.no_grad()`** : forward + `backward()` DEHORS (pour construire et
  dériver le graphe), update + remise à zéro DEDANS ;
- **torch accumule les `.grad`** → remise à zéro obligatoire à chaque pas ;
- direction/pas de la descente (`-= pas*grad`, pas `+=`) ;
- mesure de loss : diviser par le nombre de **transitions**, pas de prénoms (×6 sinon) ;
  accumuler `loss.item()`, pas le tenseur (sinon tout le graphe reste en mémoire).


---

### Étape 3 — RNN puis LSTM   (terminée le 29/07/2026)

> Log factuel tenu par Claude. Rubriques de compréhension laissées à Ethan.

**Ce qui tourne**

- **RNN** (`src/model/rnn.py`) : cellule `h = tanh(W_xh·x + W_hh·h + b)`, BPTT via
  autograd (un `backward()` par mot, loss sommée sur la séquence), état remis à zéro
  entre les mots. **Explosion du gradient rencontrée en direct** (loss 2,52 → 4,84),
  matée par **gradient clipping**. Loss finale ~**2,19**.
- **LSTM** (`src/model/lstm.py`) : quatre portes écrites à la main (i, f, o, candidat),
  `cell = f⊙cell + i⊙c̃`, `h = o⊙tanh(cell)`. Réécrit **de mémoire** après suppression
  d'une première version. Loss **2,09** en 10 époques — meilleur modèle du projet
  (bigramme 2,45 → MLP 2,37 → RNN 2,19 → LSTM 2,09).
- **Génération** : `aleena`, `myla`, `alexa`, `shaydi`, `dalynn`, `caileegh` — mots
  cohérents sur toute leur longueur, orthographe plausible ; contre `orayrin`/`leryxa`
  du bigramme. Ratés résiduels : `liecz`, `vellabne`.

**Théorie validée en dialogue** : le long du chemin `cell_t → cell_{t-1}`, le gradient
n'est multiplié que par `f_gate` (élément par élément, dans [0,1], apprise à chaque pas)
— contre la même matrice pleine `W_hh` répétée k fois dans le RNN. L'oubli devient une
décision apprise. (Première réponse « o_gate », corrigée par Ethan en « f_gate ».)

**Ce que j'ai compris**

- **Ce que représente h** : la mémoire du réseau, recalculée à chaque pas à partir de la
  lettre courante et de sa valeur précédente. Contrairement au MLP et à sa fenêtre fixe
  de 3 lettres, le RNN a accès en principe à un contexte de longueur illimitée.
- **Remise à zéro de h entre deux prénoms** : sinon il y aurait des biais et de la
  mémorisation liés au prénom précédent, ce qui n'est pas le but du RNN.
- **Explosion du gradient** : si les valeurs ne sont pas renormalisées, les valeurs
  extrêmes prennent le dessus, car les nombres plus grands que 1 élevés à la puissance k
  tendent très vite vers l'infini. Versant symétrique : en dessous de 1, le gradient
  s'évanouit (0,7⁹ ≈ 0,04) et les premières lettres ne reçoivent plus de signal.

**Ce que je ne comprends pas encore / à réviser** *(contrôle à froid du 30/07/2026)*

- **Le dépliage dans le temps** : un RNN déroulé sur 9 lettres équivaut à un réseau de
  9 couches partageant la même matrice W_hh ; le gradient la traverse 9 fois et son
  gradient est la somme des contributions de tous les pas. C'est là qu'est la profondeur.
- **Un seul backward() par mot** : la vraie raison est mécanique — le graphe de calcul
  est libéré après le premier appel. (J'avais répondu « pour garder le contexte ».)
- **Le clipping** : rééchelonne le vecteur gradient sans changer sa direction. Il traite
  l'explosion, il ne peut rien contre l'évanouissement.
- **La loss sommée** : rend le gradient proportionnel à la longueur du mot — pas
  irréguliers, et mots longs qui pèsent plus lourd sans que je l'aie décidé.
- **Les quatre portes** : c̃ propose le contenu nouveau, i décide ce qu'on écrit,
  f ce qu'on garde de la mémoire, o ce qu'on expose en sortie. (Je les avais mélangées.)
- **Le point central de l'étape** : le long de cell_t → cell_{t-1}, le gradient n'est
  multiplié que par **f_gate**, élément par élément, apprise à chaque pas — pas par une
  matrice répétée. Je l'avais trouvé seul le 29/07 (et corrigé Claude qui disait o_gate),
  je ne l'avais plus le 30/07 : à revoir en priorité.
- **cell vs h** : cell est la mémoire longue (autoroute du gradient), h ce que le réseau
  expose à ce pas. Sans la séparation, impossible de garder une information en réserve
  sans l'utiliser immédiatement.


**Ce qui m'a bloqué** *(observé pendant la session)*

- prédire la lettre **suivante** et non la lettre courante (décalage entrée/cible) ;
- un `backward()` par pas de temps → graphe libéré → erreur ; la BPTT exige un seul
  `backward()` par séquence ;
- divergence sans clipping, aggravée par la loss **sommée** (gradient ∝ longueur du mot) ;
- à vérifier plus tard : loss de **validation** du LSTM (`alexa` généré = vrai prénom,
  possible mémorisation).

---

### Étape 4 — Transformer décodeur   (en cours, commencée le 29/07/2026)

> Log factuel tenu par Claude. Rubriques de compréhension laissées à Ethan.

**Ce qui tourne**

- **Transformer 1 bloc** (`src/model/transformer.py`) : embeddings caractère + position,
  attention causale **multi-têtes** (4 têtes, dim 64), masque triangulaire, résiduels,
  LayerNorm **post-norm**, MLP ×4, écrit d'un jet — aucun bug bloquant à la relecture.
- **Épisode d'optimisation instructif** (29–30/07) : en SGD pur (pas 0,001),
  loss 2,26 en 10 époques puis 2,21 en 30 — pente qui s'effondre. Instrumentation des
  normes de gradient (valeur de retour de `clip_grad_norm_`) : normes réelles **4–8**
  contre un seuil de clipping à **1,0** hérité du LSTM → clipping déclenché à chaque
  mot, pas effectif divisé par ~5. Seuil relevé → **2,187 en 9 époques**, stable,
  encore en descente. Le transformer rejoint le RNN (2,19), le LSTM (2,09) reste devant.
- **Génération** : `katian`, `josey`, `tisha`, `vinnie` ; ratés `fphmthita`, `egckoi`
  — plus de variance que le LSTM, cohérent avec un entraînement inachevé.
- **Adam écrit à la main** (30/07/2026) : deux moyennes mobiles (β₁=0,9, β₂=0,999),
  correction de biais avec `t` global, pas 0,001. Un bug d'indice à l'écriture
  (`m[i]` — compteur de mots — au lieu de l'indice de paramètre), localisé en
  relecture, corrigé par Ethan. Résultat : **2,0522 en 10 époques**, encore en
  descente — **meilleur modèle du projet, LSTM battu dès l'époque 5**.
- **Ablation d'optimisation** (même architecture, 9-10 époques) :
  SGD clippé à 1 → 2,26 ; SGD clippé à 10 → 2,19 ; **Adam → 2,05**.
- Génération sous Adam : `hayceigh`, `dariana`, `cortalee`, `layna`, `estrey` ;
  `triston` et `aiden` sortis tels quels (vrais prénoms → soupçon de mémorisation
  persistant) ; raté notable : un prénom d'une seule lettre (`a`).
- **Théorie validée en dialogue** : en régime établi, le pas effectif d'Adam vaut
  `pas × g/√(g²) = pas × signe(g)` — l'amplitude du gradient est normalisée,
  seule la direction compte ; d'où l'insensibilité aux échelles hétérogènes
  entre couches. (Réponses d'Ethan : « 0 ? » puis « 1 ? » — la seconde est la bonne.)

**Observations ouvertes**

- La norme du gradient **croît** au fil des époques (6 → 7,5) — à expliquer
  (piste : quelles couches sont hors du périmètre des LayerNorm ?).
- `daniel` généré tel quel : même soupçon de mémorisation qu'`alexa` (LSTM).
  Loss de **validation** toujours à faire, pour les deux modèles.
- ~~Prochain chantier : Adam~~ — fait le 30/07/2026, objectif atteint (voir ci-dessus).
- **Loss de validation ajoutée par Ethan** (30/07/2026, sur `names.val.txt`, jamais vu
  à l'entraînement) : **val 2,0869 / train 2,0520** à l'époque 9, val en descente
  continue, jamais de remontée → **pas de surapprentissage**, le soupçon de
  mémorisation (`alexa`, `daniel`, `triston`) est levé pour le transformer.
  Artefact noté : val < train aux époques 0-2 (train = moyenne pendant l'époque,
  val = mesure en fin d'époque).
- **Loss de validation du LSTM ajoutée par Ethan** (30/07/2026) : **val 2,1104 /
  train 2,0898** à l'époque 9, val en descente continue → pas de surapprentissage
  non plus. **Duel val contre val : transformer 2,0869 > LSTM 2,1104** — le
  transformer gagne d'~0,024, les deux encore en descente. Soupçon de mémorisation
  levé pour les deux modèles.
- Reste ouvert : norme de gradient croissante sous SGD, prénoms trop courts
  générés (`rj`, `a`).
- **Empilement de N blocs écrit par Ethan** (30/07/2026) : `num_blocs` en seul
  bouton, listes de tenseurs par type indexées par bloc (init indépendante par
  compréhension de liste), liste plate pour Adam inchangé, vue groupée pour le
  forward. Relecture : aucun bug. **Résultat de l'ablation profondeur : 2 blocs
  val 2,0894 contre 1 bloc val 2,0869 en 10 époques — égalité**, pour 2× les
  paramètres de bloc. Sur des prénoms (~7 caractères), un bloc d'attention suffit ;
  la profondeur répond à un besoin de la tâche, ce n'est pas un gain gratuit.
  Les blocs empilés paieront sur de longues dépendances (étape 5).
- **Zone rouge de l'étape 4 complète** : attention causale multi-têtes, blocs
  empilés, Adam à la main, validation. Restent : compte rendu d'Ethan (code fermé) ;
  dette technique signalée : forward copié 3× (train/val/génération) — à factoriser.

**Ce que j'ai compris**

- **Attention vs récurrence** : dans le LSTM, l'information passe de la lettre en cours
  au contexte, se fond avec toutes les lettres du contexte puis disparaît ; dans le
  transformer, elle reste une information accessible jusqu'au bout.
- **Masque causal** : on met −inf car on veut qu'après softmax le poids soit 0 :
  exp(−inf) = 0, alors que exp(0) = 1. Sans lui, le modèle connaîtrait les prochaines
  lettres et pourrait les ressortir telles quelles pour avoir juste ; les poids les
  plus importants seraient sur les lettres futures — on aurait fabriqué une tête de
  lecture plutôt qu'un apprentissage.
- **Têtes multiples** : ça permet 4 critères de pertinence en simultané, pour le même
  nombre de poids qu'une seule grosse tête.
- **Adam** : m sert à caractériser l'inertie et v à renormaliser. En régime établi,
  le pas effectif vaut une constante (≈ le learning rate) : l'amplitude du gradient
  ne compte plus, seule la direction reste.
- **Validation** : un réseau très fonctionnel et un réseau qui a surappris sont
  indiscernables sur la loss de train, car elle dit seulement s'il a raison, pas
  comment il y arrive. Le surapprentissage se voit quand la val remonte pendant que
  le train continue de baisser : ce que le train gagne encore n'est plus que de la
  mémorisation. Je ne l'ai pas observé : mes deux vals descendaient jusqu'au bout.

**Ce qui m'a bloqué**

- L'erreur que j'ai le plus de chances de refaire : oublier comment fonctionne la
  structure et m'emmêler dans les formules.

**Ce qui m'a bloqué** *(observé pendant la session)*

- seuil de clipping recopié d'un contexte (loss sommée du LSTM) vers un autre
  (loss moyennée) sans re-vérifier s'il se déclenchait — mesuré, puis corrigé.

---

### Étape 5 — Tokenizer BPE + passage à l'échelle   (commencée le 31/07/2026)

> Log factuel tenu par Claude. Rubriques de compréhension laissées à Ethan.

**Corpus** (zone verte, préparé par Claude — `src/tooling/prepare_tinystories.py`)

- **TinyStories V2 GPT-4** choisi contre un corpus français : vocabulaire volontairement
  simple, conçu pour les petits modèles, et résultats comparables à la littérature.
  Le français est réservé à un second run, une fois le premier acquis.
- Source : fichier de validation HuggingFace (22 Mo) — assez pour écrire et déboguer
  le BPE sans attendre les 2 Go du corpus complet.
- Nettoyage : reliquats cp1252 et ponctuation typographique ramenés en ASCII, accents
  résiduels dépouillés, lignes vides supprimées. **76 caractères distincts.**
- Découpage **par histoire** (jamais au milieu d'une), seed 1337 :
  train **19,92 Mo** / val 1,11 Mo / test 1,09 Mo, 27 629 histoires.
- **Fragment initial tronqué repéré par Ethan** : le fichier brut de HuggingFace commence
  au milieu d'une histoire. Écarté à la main, puis garde-fou ajouté au script pour qu'un
  retéléchargement ne le réintroduise pas en silence.

**Théorie vue en dialogue avant écriture**

- Pourquoi le BPE : au niveau caractère les séquences sont longues (coût d'attention en
  T²) ; au niveau mot le vocabulaire explose et tout mot inconnu est irrécupérable.
- L'espace collé en tête de token : distingue le mot entier du fragment interne
  (` the` vs `the` dans *theater*) et rend le décodage exactement réversible.
  (Réponse d'Ethan : marque la position début/fin de mot — juste, affinée ensuite.)
- Entraînement des fusions et encodage sont **deux programmes distincts** ; l'ordre des
  fusions est l'algorithme.

**Ce qui tourne** *(chaîne complète validée le 11/08/2026)*

Quatre programmes écrits par Ethan, en Python pur, sans dépendance :

- **`bpe_stories.py`** — entraînement des fusions. Comptage des mots (34 224 distincts
  pour 3 970 994 occurrences), espace collé en tête, puis 1000 fusions : à chaque tour,
  comptage de toutes les paires adjacentes pondéré par la fréquence du mot, sélection de
  la plus fréquente, réécriture du corpus. Sortie `bpe.json` : **alphabet de 76
  caractères + 1000 règles ordonnées**. Run : **2 min**.
- **`bpe_liste.py`** — dérive le vocabulaire des règles : alphabet (indices 0–75) puis
  produit de chaque fusion (76–1075). **1076 tokens, aucun doublon** — garanti par
  construction : les produits font ≥ 2 caractères donc ne peuvent pas percuter
  l'alphabet, et une paire déjà fusionnée n'existe plus nulle part, donc ne peut pas
  être choisie deux fois.
- **`encodeur.py`** — texte → tokens. Délimiteur `[ ]` vérifié hors alphabet au
  démarrage, application des 1000 règles **dans l'ordre du fichier**, garde-fou sur
  caractère inconnu. Corpus complet encodé : **5 846 939 tokens pour 19 918 766
  caractères, ratio 3,41 car/token**. Run : **2 min 13**.
- **`decodeur.py`** — tokens → texte, par simple recollage.

**Les trois contrôles passent :**

- **Aller-retour exact sur les 20 Mo** : `''.join(tokens)[1:] == corpus`, caractère pour
  caractère, retours à la ligne et ponctuation compris.
- **Déterminisme** : réentraînement du BPE après ajout de `'\n'` → les 1000 fusions
  ressortent **identiques** à celles du 2 août.
- **Couverture** : 0 token du corpus encodé absent du vocabulaire.

Premiers tokens du corpus : `' Once', ' upon', ' a', ' time,', ' in', ' a', ' small',
' hous', 'e,', ' there'`.

**Décisions de modélisation prises**

- **`'\n'` conservé** et ajouté à l'alphabet (indice 75) : il marque la frontière entre
  histoires — même rôle que le symbole de frontière du bigramme de l'étape 0. Sans lui
  le corpus serait un flot continu de 20 Mo sans début ni fin, et le modèle
  n'apprendrait jamais à terminer une histoire. Aucune fusion ne le concerne (le
  `split()` de l'entraînement l'avait éliminé), il reste donc un token isolé.
- **Vocabulaire gelé avant l'entraînement.** Un caractère inconnu fait planter
  l'encodeur bruyamment plutôt que d'être ajouté à la volée : un vocabulaire qui
  dépend du texte encodé rendrait tout checkpoint illisible dès que les indices bougent.
- **Ponctuation collée aux mots** (`' time'`, `' time,'`, `' time.'` = 3 tokens
  distincts) — perte de vocabulaire assumée pour l'instant ; les tokenizers de
  production détachent la ponctuation avant le BPE. À reprendre si le vocabulaire
  devient contraignant.

**Ce qui m'a bloqué** *(observé pendant la session)*

- **Stocker la concaténation au lieu du couple** : `' the'` au lieu de `(' th', 'e')`.
  Une fusion est une **règle de réécriture**, pas un token — l'information de découpe
  est détruite par la concaténation, et l'encodeur devient impossible à écrire.
- **Confondre alphabet et vocabulaire** : recherche des tokens dans les 75 caractères
  alors que ` Once` n'y est pas et ne peut pas y être. L'artefact contient des règles ;
  le vocabulaire en est *dérivé*, il n'est écrit nulle part.
- **Format d'échange entre encodeur et décodeur** : la chaîne à crochets `"[ Once][ upon]"`,
  échafaudage interne de l'encodeur, écrite sur disque pendant plusieurs itérations —
  au point de devoir transporter sa propre clé `separateur` pour être relisible. Symptôme
  reconnu tardivement ; le passage à une liste de tokens a fait disparaître d'un coup le
  bloc de re-découpage (qui avait produit une **boucle infinie**) et simplifié le décodeur.
- **Trois `json.dump` successifs dans un fichier** → JSON invalide (`Extra data`).
  Un fichier JSON contient exactement une valeur.
- **`encode.json` écrasé** en lançant un nouveau script sans changer son chemin de sortie.
- **`s += x` dans une boucle sur 20 M caractères** : quadratique, car une chaîne Python
  est immuable — chaque ajout réalloue et recopie tout. Mesuré : **1710 s sur 2 Mo**
  contre **0,21 s** pour l'accumulation en liste puis un seul assemblage
  (×8014, et l'écart croît avec la taille). Extrapolé : ~16 h contre ~2 s.
  Même famille : `x in une_liste` (parcours linéaire, 75 comparaisons × 20 M) remplacé
  par un ensemble ; la liste est conservée en parallèle car elle seule porte l'ordre,
  donc les indices.
- Bugs Python de parcours : `for car in encode2` puis `car[i]` (le caractère courant
  n'est pas la chaîne) → boucle infinie ; `pass` pris pour une instruction de saut ;
  `if ensemble is not None` (toujours vrai) au lieu du test de vacuité ; `retour2=''`
  puis `+=` pour construire ce qui devait être une liste ; clé renommée dans le
  dictionnaire mais pas dans les écritures (`KeyError`) ; `breakpoint()` laissé dans un
  script ; perte des crochets et de l'espace initial en réécrivant la boucle
  d'accumulation.

**Dette technique / à faire**

- `encode.json` fait **43 Mo** et vit dans `src/model/` : sa place est dans `data/` ou
  `runs/`, et il ne doit pas être versionné. Le format naturel à cette taille est un
  tableau NumPy binaire, pas du JSON.
- Encodeur et décodeur sont des **scripts** à chemins codés en dur, pas des fonctions —
  d'où l'impossibilité de les tester directement (les vérifications d'aller-retour ont
  été faites en réécrivant les chemins à la volée depuis l'extérieur).
- La conversion **token → entier** n'est pas encore écrite : `bpe_liste.json` fournit la
  table, `encode.json` contient encore des chaînes.

---

#### Reprise du 11/08/2026 (après-midi) — la pré-tokenisation refaite

> Log factuel tenu par Claude. **Remplace les chiffres du bloc « Ce qui tourne » ci-dessus.**

**Le défaut trouvé**

`bpe_stories.py` collait un espace en tête de **chaque** mot au moment d'apprendre les
fusions (`mot = ' ' + mot`, après un `ligne.split()`). L'encodeur, lui, applique les
règles sur le flux brut. Un mot précédé d'un `'\n'` n'a pas d'espace devant : **aucune
règle de début de mot ne pouvait se déclencher**, et le mot tombait en caractères isolés.

```
[' much', '.', '\n', 'O', 'n', 'e', ' day,']      au lieu de     [..., '\n', 'One', ' day,']
```

Mesuré sur les 20 Mo : 132 580 lignes, **285 886 tokens** pour leurs premiers mots
(2,65 par mot contre 1,40 ailleurs), soit **4,9 % du corpus** gaspillé — et précisément
à la position où le modèle doit apprendre comment une histoire commence.

**Cause profonde.** `split()` détruit l'information de *quel* blanc précédait chaque mot ;
`' ' + mot` **affirme** ensuite que c'était un espace, ce qui est faux 132 580 fois. Le
corpus sur lequel les fusions étaient apprises n'était pas celui qui était encodé.

**Deux schémas essayés, tous deux mesurés sur le corpus complet**

| schéma | total tokens | début de ligne | ailleurs | ratio car/token |
|---|---|---|---|---|
| origine — espace forcé en tête | 5 846 939 | 2,654 tok/mot | 1,402 | 3,4067 |
| blanc rattaché **en queue** (`'One '`) | 5 877 686 | 1,379 | 1,446 | 3,3889 |
| blanc rattaché **en tête**, réel (retenu) | **5 749 964** | 1,629 | 1,405 | **3,4642** |

Le schéma en queue répare parfaitement les débuts de ligne (−137 403 tokens) mais
dégrade tout le reste (+167 576) : ses premières fusions sont des **terminaisons
génériques** (`'e '`, `'d '`, `'t '`, `'. '`) qui ne disent rien de l'identité du mot,
là où l'espace en tête produit des **ancres** qui se prolongent en mots entiers
(`' t'` → `' the'` → `' to'`, `' and'` dès le rang 11). Il dédouble en outre les
9 470 mots distincts de **fin** de ligne, contre 1 760 de **début** pour l'autre.

**Schéma retenu** — celui de GPT-2 : le blanc qui part avec le mot est celui qui le
**précède réellement** (donc rien en début de ligne), et `'\n'` est une unité isolée.

```
"Once upon a time\n\nOne day"  →  'Once' | ' upon' | ' a' | ' time' | '\n' | '\n' | 'One' | ' day'
```

**Ce que ça change par rapport aux décisions notées le matin**

- `'\n'` n'est plus ajouté à la main à l'alphabet : il **entre par l'apprentissage**
  comme unité à part entière. La phrase « aucune fusion ne le concerne » du bloc
  *Décisions de modélisation* ne vaut plus.
- Chaque mot a maintenant **deux formes possibles** : `' mot'` (cas courant) et `'mot'`
  (début de ligne, 1 760 distincts, dont les 100 premiers couvrent 90 %). Dédoublement
  assumé — c'est celui de GPT-2, qui a bien `' One'` *et* `'One'`.

**Chiffres finaux** *(run `runs/20260811-1418-tokenizer/`)*

- 19 918 766 caractères → **5 749 964 tokens**, ratio **3,4642** (+1,69 %, soit
  **96 975 tokens économisés**)
- début de ligne **1,63** tok/mot (était 2,65) · ailleurs **1,41** · rapport **1,16**
- aller-retour exact sur les 20 Mo, `decodeur.py` conforme, 1076 tokens sans doublon,
  0 token hors vocabulaire, 0 token contenant un `'\n'` autre que `'\n'` lui-même
- apprentissage 1 min 8 · encodage 2 min 6

**Ce qui reste, et qui n'est plus un bug**

Le rapport de 1,16 est le coût résiduel des formes nues. Le levier n'est plus la
pré-tokenisation mais le **budget de fusions** : 1 000 règles pour 34 224 mots distincts.
`'Once'` et `'One'` ont leur forme nue, `'While'` sort encore en `'W','h','ile'`.
Monter le vocabulaire est un réglage à mesurer, pas une correction.

**Erreurs de parcours rencontrées** *(Python, pas modélisation — six itérations)*

- `for mot in ligne` itère les **caractères**, pas les mots.
- `i += 1` dans un `for i in range(...)` : **sans effet**, le `for` réaffecte `i` à chaque
  tour depuis son itérateur. Un `for` sur `range` avance de 1, toujours ; un découpage en
  unités doit avancer de **la longueur de l'unité produite** → boucle `while` à curseur
  explicite, la même forme que celle déjà écrite pour appliquer les fusions.
- Consommer un caractère dans `mot` sans avancer le curseur → première lettre doublée
  (`' uupon'`), puis boucle infinie sur le `'\n'` (9,9 M incréments en 3 s).
- Deux `i += 1` pour un caractère consommé → un caractère sur deux sauté (`'Oneuo'`).
- Relire `ligne[i]` après avoir consommé le dernier caractère de la ligne → `IndexError`.
- `mot = ' ' + mot` **après** avoir lu le mot : colle au mot l'espace qui le **suit**.
  Tous les blancs décalés d'un mot — `" Once upon atime"`. « Blanc en tête » n'est pas
  une direction de concaténation, c'est le choix de **quel** blanc ; le seul moment où
  l'on tient celui qui précède, c'est quand le curseur est **dessus**.

**L'invariant qui règle tous ces cas d'un coup**

> Le curseur avance exactement du nombre de caractères consommés, à chaque tour,
> sans exception. Et une unité ne consomme jamais le blanc qui la suit.

**Ce que la vérification a appris**

- Le test du **multiensemble de caractères** (mêmes caractères, mêmes comptes) **ne voit
  pas** un décalage de position. Il laisse passer le `' Once upon atime'`. Seul
  `''.join(tokens) == texte` l'attrape. Un contrôle qui ignore l'ordre ne prouve pas
  la reconstruction.
- Le **nombre de fusions apprises** est une alarme gratuite : s'il sort à 38 ou 418 au
  lieu de 1000, c'est que la pré-tokenisation produit des unités trop courtes et que les
  paires se sont épuisées.
- Deux prédictions chiffrées de Claude ont été démenties par la mesure (le ratio devait
  monter avec le schéma en queue : il a baissé ; les débuts de ligne devaient tomber à
  1,38 avec le schéma en tête : 1,63). Le banc de contrôle a tranché les deux fois.

**Outillage ajouté** *(zone verte, écrit par Claude)*

`scripts/tokenizer.py` — enchaîne `bpe_stories` → `bpe_liste` → `encodeur`, chronomètre,
archive dans `runs/<date>-tokenizer/` (log, config, copies des JSON, sauvegarde des
artefacts écrasés), et passe sept contrôles : nombre de fusions, doublons du vocabulaire,
couverture, aller-retour (plus conformité de `decodeur.py`), ratio de compression,
coût des débuts de ligne, échantillon autour des sauts de ligne.

```
python scripts/tokenizer.py                               # corpus complet, ~3 min 30
python scripts/tokenizer.py --corpus data/stories.val.txt # essai rapide
python scripts/tokenizer.py --etapes verif                # re-vérifier sans recalculer
```

**Dette technique — mise à jour**

- ✅ `encode.json` est désormais écrit dans `data/`, plus dans `src/model/`.
- ✅ Encodeur et décodeur sont bien des **fonctions** paramétrées par leurs chemins
  d'entrée ; c'est ce qui permet au script de les orchestrer. Restent faux : leurs
  chemins par défaut au niveau module (`src/model/bpe.json`, inexistant) et l'absence
  de bloc `__main__` — ils ne se lancent donc pas seuls.
- ⬜ Chemins de **sortie** toujours codés en dur dans les trois modules.
- ⬜ Conversion **token → entier** toujours pas écrite : `encode.json` contient encore
  5 749 964 chaînes (43 Mo de JSON là où un tableau d'entiers suffirait).
- ⬜ `bpe_stories.py` ligne 15 : `ligne[i]` déborde si une ligne se termine par un espace
  sans saut de ligne derrière. Latent — les trois fichiers `stories.*.txt` finissent
  tous par `'\n'`.
- ⬜ Le token de paragraphe `'\n\n'` (24 866 occurrences) n'existe pas : la lecture ligne
  par ligne livre deux `'\n'` séparés. Lire le texte d'un seul tenant le rendrait
  fusionnable.

**Aucun `PSEUDOCODE` demandé pendant cette session.**

---

#### Reprise du 13-14/08/2026 — le transformer sur tokens BPE

> Log factuel tenu par Claude. Rubriques de compréhension laissées à Ethan.

**Ce qui tourne**

Le transformer de l'étape 4, écrit pour des prénoms au niveau caractère, adapté au
corpus TinyStories tokenisé. Quatre changements de fond :

- **vocabulaire 27 → 2080**, sur la table d'embedding et la couche de sortie ;
- **contexte 30 → 256** positions ;
- **dimension de lot ajoutée** à toute la passe avant — c'était le vrai chantier ;
- **exécution sur GPU** (RTX 5050), là où les étapes 0 à 4 tournaient sur CPU sans
  qu'Ethan s'en soit aperçu.

**`tireur_de_lot.py`** — programme neuf. Reçoit le tableau d'entiers, une taille de lot
et une longueur de contexte ; rend deux tenseurs `(B, T)` : les entrées et les cibles,
décalées d'un token. Tire `B` décalages au hasard, prend `T+1` tokens à partir de
chacun, et découpe chaque fenêtre en deux. Contrat vérifié :
`cible[b,i] == entrée[b,i+1]` sur tout le lot.

**Premier run** *(13/08, `runs/save_runs/20260813-2326/`)*

```
dim 128 · 2 blocs · 4 têtes · vocabulaire 2080   ->  963 000 paramètres
3000 pas × 32 × 256 = 24,6 M tokens              ->  ~2 min 30 sur GPU

pas    240   val 4,1255   perplexité 61,9
pas   1440   val 3,0631   perplexité 21,4
pas   2880   val 2,7979   perplexité 16,4
pas   3000   train 2,7681   val 2,7979
```

Validation **décroissante à chaque mesure**, sans une seule remontée. Écart train/val de
**0,03** : aucun surapprentissage.

**Première génération**

```
Once upon a time, there was a little boy named Tim. Tim was eagle to clean things.
He liked to hope with his friends, and Fluffy was having lots of fun. They played all day.
The moral of the story is to be brave and be honey.
```

Grammaire correcte, ponctuation et guillemets appariés, formules du corpus acquises
(`Once upon a time`, `The moral of the story is`), frontières d'histoire respectées.
Sémantique défaillante : `eagle` pour *eager*, `honey` pour *honest*, et des mots
inexistants assemblés de fragments plausibles — `strengue`, `trayake`, `kindning`.
C'est le régime décrit par Eldan & Li pour ~1 M de paramètres.

**Passage à l'échelle** *(14/08)*

Corpus porté de 90 Mo à **500 Mo** (sur les 2,2 Go téléchargés). L'encodeur ne peut pas
traiter plus d'une centaine de mégaoctets d'un coup — il construit une chaîne Python de
3 caractères par caractère du corpus, soit ~60 octets de RAM par caractère. Encodage
donc **par morceaux de 85 Mo**, découpés sur des frontières d'histoire, avec le
vocabulaire gelé de 2080 tokens ; les sorties sont ensuite concaténées.

Résultat : **126 150 131 tokens** (train 113,6 M · val 6,27 M · test 6,32 M), ratio 3,96,
aller-retour exact sur les trois splits.

Modèle porté à **dim 256 · 6 blocs → 5,86 M paramètres**, 30 000 pas.

**Ce qui m'a bloqué** *(observé pendant la session)*

- **Boucler sur le lot.** Premier réflexe : tirer 1000 fenêtres puis les traiter une par
  une. Ça annule l'objet même du lot et ramène au régime de l'étape 4.
- **Les axes.** Ajouter une dimension en tête décale tous les indices positifs
  (`transpose(0,1)` → `transpose(1,2)`) et invalide toutes les formes complètes
  (`view(T, h, hd)` → `view(B, T, h, hd)`). Les indices **négatifs** (`transpose(-2,-1)`,
  `softmax(dim=-1)`, `layernorm(dim=-1)`) y sont immunisés — ils ont traversé la
  migration sans une correction.
- **`cross_entropy`** attend l'axe des classes en deuxième position. Avec des logits
  `(B, T, V)` et des cibles `(B, T)`, il faut aplatir les deux premiers axes.
- **Le type des cibles** : `int32` suffit pour indexer un embedding, `cross_entropy`
  exige `int64`.
- **`.to(device)` sur un paramètre** en fait un tenseur non-feuille : `.grad` reste
  `None` et la boucle Adam plante sur `float * NoneType`. L'appareil se donne **à la
  création**.
- **Un même nombre écrit à deux endroits — cinq fois dans la même journée** : `T` défini
  à 256 puis recalculé en `len(tok)` (masque de 2 pétaoctets), la taille de lot, le
  diviseur de la loss d'entraînement, celui de la validation, l'intervalle d'affichage
  passé de 50 à 250 sans son diviseur (loss affichée 5× trop grande). Chaque fois, le
  correctif était le même : un nom, un seul endroit.
- **Un remplacement global de `T` par `max_len`** a emporté les 13 `.T` de transposition
  de tenseur, sans rapport avec la variable.
- **Environnement** : Triton, le compilateur de noyaux GPU de PyTorch, échouait à
  compiler faute des en-têtes `python3-devel`. Visible seulement à la génération, où
  les formes petites empruntent un autre chemin que l'entraînement.

**Outillage ajouté** *(zone verte, écrit par Claude)*

- `scripts/prepare_stories.py` — préparation incrémentale du corpus brut (1,1 Go de
  pointe au lieu de 6-7), avec filtre sur l'alphabet gelé : une histoire contenant un
  caractère inconnu est écartée ici plutôt que de faire échouer l'encodeur une heure
  plus tard. Sur 500 Mo : 64 occurrences fautives, 36 histoires sur 558 715.
- `scripts/encode_gros_corpus.py` — encodage par morceaux et concaténation.
- `scripts/nuit.sh` — enchaînement corpus → tokens → entiers → entraînement.

**Incident** : la première chaîne de nuit a perdu l'encodage de train (94 min).
`encodeur.py` écrit toujours dans `data/encode.json`, chemin qui servait aussi de
destination finale à train ; le premier morceau de val l'a écrasé. Corrigé en assemblant
chaque split à l'écart et en ne déplaçant qu'à la toute fin. Val et test, écrits après,
étaient intacts.

---

#### Reprise du 14-15/08/2026 — lois d'échelle et initialisation

> Log factuel tenu par Claude. Rubriques de compréhension laissées à Ethan.

**Le contexte porté à 384**

Première expérience : `max_len` de 256 à 384, tout le reste identique. Gain net —
perplexité 6,85 → 6,37 pour 33 000 paramètres de plus (seuls les embeddings de position
changent). L'explication tient à la couverture des histoires : à 256 tokens, 87 % des
histoires tiennent entières ; à 384, environ 95 %. Le modèle voit presque toujours
l'histoire complète, début compris.

**Réserve méthodologique** : deux choses ont bougé en même temps. Le lot portant
12 288 tokens au lieu de 8 192, à nombre de pas égal ce run a aussi vu 50 % de tokens de
plus. L'attribution n'est pas propre — leçon retenue pour les runs suivants, tous menés à
budget de tokens identique.

**Le goulot n'était pas le GPU**

Chronométrage à 100 pas : `dim=256` et `dim=384` donnent **exactement le même temps**
(0,1184 s contre 0,1185 s). Élargir le modèle de 45 % ne coûtait rien — signe que le GPU
attendait. Décomposition du pas :

```
tireur_de_lot           21,0 ms   18 %   (double boucle Python, 12 288 tokens recopiés)
boucle Adam à la main   14,4 ms   12 %   (~180 lancements de noyau par pas)
reste                     83 ms   70 %
```

Le calcul matriciel ne dominait pas ; l'interpréteur Python, si. Conséquence pratique :
de la capacité disponible gratuitement. *(Après correction de l'initialisation, le
chronométrage a changé — 0,072 s à `dim=256` contre 0,169 s à 512 — le calcul est
redevenu dominant.)*

**Loi d'échelle sur la largeur** *(ancienne initialisation, `0.1` constant)*

Trois largeurs, 2 blocs, contexte 384, 30 000 pas, 369 M tokens vus, seed 1337. Seul
`dim` change :

| `dim` | paramètres | val | perplexité | gain |
|---|---|---|---|---|
| 256 | 2,74 M | 1,8513 | 6,37 | — |
| 384 | 5,29 M | 1,7292 | 5,64 | −11,5 % |
| 512 | 8,60 M | 1,6622 | 5,27 | −6,6 % |

Rendement décroissant : le second doublement rapporte environ deux fois moins que le
premier par paramètre ajouté.

**L'initialisation en fonction du fan-in**

Une couche calcule `y = w1·x1 + ... + wn·xn`. Si les poids ont une variance σ² et les
entrées une variance 1, alors `Var(y) = n × σ²`. Pour que la sortie garde l'amplitude de
l'entrée, il faut **σ = 1/√n**, où `n` est le nombre d'entrées de la couche.

L'ancien `0,1` constant était donc trop grand d'un facteur qui **croît avec la largeur** :

```
             sigma correct    0,1 était...
dim 256         0,0625        1,6× trop grand
dim 384         0,0510        2,0×
dim 512         0,0442        2,3×

W_2 (fan-in 4·dim)            3,2× à dim=256,  4,5× à dim=512
```

Sept lignes modifiées (`W_q`, `W_k`, `W_v`, `W_o`, `W_1`, `W_2`, `W_out`), les biais et
les embeddings laissés — ils ne somment rien. Piège évité : `W_1` et `W_2` ont des formes
inversées, donc des fan-in différents (`dim` et `dim*4`).

**Résultat : six points appariés**

| `dim` | ancien init | | init 1/√n | | gain |
|---|---|---|---|---|---|
| | loss | perplexité | loss | perplexité | |
| 256 | 1,8513 | 6,37 | 1,8265 | 6,21 | −2,4 % |
| 384 | 1,7292 | 5,64 | 1,6946 | 5,44 | −3,4 % |
| 512 | 1,6622 | 5,27 | 1,6132 | 5,02 | −4,8 % |

**L'écart se creuse avec la largeur**, exactement comme la théorie le prédisait — c'était
le point qui pouvait la démentir. Une initialisation constante ne se contente pas d'être
approximative : elle se dégrade à mesure qu'on grandit. Le gain reste modeste ici (2 à
5 %), mais il continuerait de croître à `dim=1024` ou 2048.

**Une hypothèse réfutée**

Le plateau des 6 blocs avait été attribué à l'atténuation du chemin résiduel, causée par
l'initialisation. Calcul refait en comptant les **deux** normalisations par bloc (j'en
avais oublié une) :

```
                  survie par bloc   après 2 blocs   après 6 blocs
ancien init 0,1        9,7 %            0,94 %        0,0001 %
init en 1/√n          57,7 %             33 %           3,7 %
```

L'initialisation remonte la survie d'un facteur trente mille à 6 blocs. Test mené :
`dim=256`, 6 blocs, nouvelle initialisation, 2000 pas → **toujours 6,32**, soit
l'entropie unigramme. Aucun changement.

**L'hypothèse est réfutée.** L'initialisation était bien un problème — sur la largeur, où
son effet est mesuré ci-dessus — mais elle n'est pas la cause de l'échec en profondeur.
Reste comme explication le **placement** de la normalisation et non son échelle : en
post-norm, le chemin résiduel est renormalisé à chaque bloc, donc jamais libre. Les deux
réponses connues agissent ailleurs — le **pré-norm** (`x + f(LayerNorm(x))`, qui laisse le
résiduel intact) et l'**échauffement du pas d'apprentissage**. Ni l'une ni l'autre n'a
encore été testée.

**État du meilleur modèle**

```
dim 512 · 2 blocs · 4 têtes · contexte 384 · vocabulaire 2080
8,6 M paramètres · 369 M tokens vus · perplexité 5,02

train 1,6132   val 1,6132   écart nul à quatre décimales
```

Toujours aucun surapprentissage, les deux courbes descendaient encore à l'arrêt.

**Ce qui m'a bloqué** *(observé pendant la session)*

- **Le sélecteur d'échantillonnage placé dans la boucle d'entraînement**, autour de la
  loss et du `backward`. L'échantillonnage n'existe qu'à la génération : à l'entraînement
  la cible est connue, la cross-entropy compare la distribution entière à la vérité, il
  n'y a rien à choisir.
- **Confusion entre initialisation et pas d'apprentissage.** L'initialisation fixe les
  valeurs de départ, une fois ; le pas fixe la distance parcourue à chaque gradient. Deux
  leviers distincts contre le même symptôme — l'échauffement agit sur le second.
- **`dim` codé en dur** rend impossible d'enchaîner plusieurs configurations sans
  bricoler le fichier. En faire un argument de ligne de commande est le premier pas vers
  une série d'expériences.

**Ce qui manque encore et se fait sentir**

La **reprise sur checkpoint** : les fichiers contiennent poids, moyennes d'Adam et numéro
de pas, mais rien ne les relit. Toute prolongation refait le calcul déjà fait.

La **génération dans un programme séparé** : soixante checkpoints en réserve, et aucun
moyen de les interroger sans relancer une heure d'entraînement. C'est ce qui a empêché de
tester l'échantillonnage.

---

#### 15/08/2026 — `generation.py` : interroger un modèle au lieu d'en fabriquer un

> Log factuel tenu par Claude. Rubriques de compréhension laissées à Ethan.

**Ce qui tourne**

Premier programme du projet qui **interroge** un modèle plutôt que d'en produire un.
Il charge un checkpoint, reconstruit le modèle depuis la configuration qui y est
enregistrée, encode une amorce, et génère du texte — en trois secondes au lieu d'une
heure d'entraînement.

```
genere(chemin_du_checkpoint, longueur, mode, température, amorce)
```

Trois modes d'échantillonnage écrits : `greedy`, `topk`, `topp`. Plus les deux fonctions
`encode` et `rencode`, reprises des programmes du tokenizer et adaptées pour travailler en
mémoire sur une amorce courte.

**Le principe de l'échantillonnage**

À l'entraînement, il n'y a rien à choisir : la cible est connue, la cross-entropy compare
la distribution entière à la vérité. L'échantillonnage n'existe qu'à la **génération**,
où le modèle produit une distribution et où il faut en sortir un token.

L'ordre des opérations est ce qui compte :

```
logits  --(1) température : diviser par T--> softmax --> probabilités
        --(2) troncature : mettre à zéro--> (3) renormaliser --> tirage
```

La température agit **avant** le softmax : diviser les logits par `T` puis exponentier
revient à élever les probabilités à la puissance `1/T`. Les troncatures agissent
**après**, sur les probabilités, et exigent de renormaliser — en divisant par la nouvelle
somme, pas en repassant un softmax.

**Les quatre stratégies**

| | ce qu'elle fait | réglage |
|---|---|---|
| greedy | prend le token le plus probable | aucun, déterministe |
| température | déforme la distribution sans rien supprimer | `T` continu |
| top-k | garde les `k` plus probables, annule le reste | `k` absolu |
| top-p | garde les plus probables jusqu'à cumuler `p` | `p`, nombre de candidats variable |

Le top-p adapte le nombre de candidats à la forme de la distribution — deux quand le
modèle est sûr, trois cents quand il hésite. C'est la réponse au défaut du top-k, dont le
`k` fixe déforme d'autant plus que la distribution est concentrée.

**Résultats sur le meilleur modèle** *(dim 512, 2 blocs, perplexité 5,02, amorce
« Once upon a time »)*

```
greedy    Once upon a time there was a little girl named Lucy. She was three years old
          and loved to play with her toys. One day, she was playing with her toy bear
          when she heard a loud noise. She looked around and saw a big truck with a big

top-k 3   Once upon a time there was a little girl called Daisy. Daisy liked to explore
          the world around her. She had a very long, thin tail and she liked to explore
          the world around her.

top-p 0,9 Once upon a time there were two good friends, Amy and Sarah. Whenever Mimi
          listened in enough. Not mummy spoke. Daisy decided she didn't turn it.
```

Le compromis est lisible directement : greedy est cohérent mais plat, top-k à 3 boucle
(« liked to explore the world around her » deux fois), top-p varie mais perd ses
référents — Amy, Sarah, Mimi, Daisy, Daddy en trois phrases.

**Une mesure faite en chemin**

Sans amorce, le greedy rend une **chaîne vide**. Ce n'est pas un bug : la distribution
après un `'\n'` isolé donne

```
19,76 %  '\n'      <- le plus probable, donc l'arrêt immédiat
15,21 %  'One'
 9,16 %  'Once'
 7,77 %  'The'
```

Le token le plus probable après un saut de ligne est un autre saut de ligne — la
frontière entre histoires. Greedy le choisit et s'arrête. Avec `k=3`, le même effet donne
un texte vide **deux fois sur trois** : renormalisés sur trois candidats, le `'\n'` pèse
44,8 % au lieu de 19,8 %. Le top-p, qui garde une dizaine de candidats, n'y tombe jamais.

C'est le défaut du greedy observé sur des chiffres : il prend le chemin le plus probable
pas à pas, et ici le premier pas est déjà dégénéré. L'amorce le règle — le même greedy
produit alors six phrases cohérentes.

**Ce qui m'a bloqué** *(observé pendant la session)*

- **L'échantillonnage placé dans la boucle d'entraînement**, autour de la loss et du
  `backward` — première tentative. Avec `mode='topk'`, il n'y aurait eu ni loss ni
  rétropropagation.
- **`from transformer import layernorm`** : `transformer.py` n'ayant pas de bloc
  `__main__`, cet import aurait relancé 30 000 pas d'entraînement pour récupérer une
  fonction de quatre lignes.
- **`pos_emb` et `b_out` réinitialisés au hasard** au lieu d'être lus dans le checkpoint,
  alors qu'ils y étaient. Bug silencieux : le texte produit aurait été incohérent sans
  qu'aucune erreur ne le signale, et le symptôme ressemble à ce qu'on attend d'un petit
  modèle.
- **Le changement de référentiel après un tri**, rencontré cinq fois. `torch.sort` rend
  un couple `(valeurs, indices)` ; une position dans le tableau trié n'est **pas** un
  numéro de token. La table `y2[1]` fait la conversion — on la lit, on ne la parcourt
  pas. Trois boucles successives ont été écrites pour chercher ce qu'une indexation
  donne. Symptôme mémorable : `'nOnO'`, c'est-à-dire les tokens 0, 1 et 2 du vocabulaire.
- **`torch.max` contre `torch.argmax`** : la valeur maximale contre son indice.
- **Un tenseur glissé dans `seq`** au lieu d'un entier : PyTorch bascule alors en
  indexation multidimensionnelle et `c[seq]` sort avec une forme absurde. La génération
  s'arrêtait au deuxième token.
- **`y[:k]` pour « les k plus probables »** : c'est « les k premiers du vocabulaire ».
  Sans tri, le classement n'existe pas.
- **Le dictionnaire d'emballage `{'encode': [...]}`**, à nouveau. Format de fichier
  reproduit pour un échange entre deux fonctions en mémoire, où il n'a aucune raison
  d'être.

**Mesuré au passage**

```
tri par boucle Python : 40,90 ms      sur 384 tokens : 15,7 s
torch.sort            :  0,083 ms                      0,032 s     493× plus rapide
```

**Ce qui reste sur ce fichier**

- `k` et `p` sont codés en dur dans le corps de `genere` : les comparer demande de rouvrir
  le fichier, ce que le banc d'essai doit justement éviter.
- Au-delà de `max_len` tokens demandés, `pos_emb[:T]` rend moins de lignes que `x` n'en a
  et l'addition échoue. Deux réponses possibles : s'arrêter, ou ne garder que les derniers
  `max_len` tokens en contexte.
- Le coût est quadratique : chaque nouveau token repasse toute la séquence dans le modèle.
  Invisible à 45 tokens, sensible à 384. La parade est un cache des clés et valeurs.
- La passe avant existe maintenant en **trois** exemplaires — entraînement, validation,
  génération. Elles ne peuvent plus diverger par accident, mais toujours par oubli.

---

#### 15/08/2026 (suite) — le banc d'essai des réglages d'échantillonnage

> Log factuel tenu par Claude. Rubriques de compréhension laissées à Ethan.

**Le dispositif**

`generation.py` produit une batterie de textes sur une grille de réglages ; chaque texte
porte **avec lui** les paramètres qui l'ont produit, dans un dictionnaire. C'est ce qui
permet ensuite de regrouper sans jamais rejouer l'ordre des boucles — la première version
accumulait des chaînes dans une liste plate, où la 137ᵉ ne disait plus d'où elle venait.

```
6 valeurs de k × 4 températures × 100 répétitions × 4 amorces  =  4 800 textes
                                                                  ~20 min de calcul
```

Les quatre amorces (`Once upon a time`, `One time in a castle`, `One day i will`,
`Once `) sont choisies par le numéro de répétition, avec des seuils **relatifs** au
nombre total — sinon un essai court à 4 répétitions n'en couvrirait qu'une seule.

**Le vocabulaire de référence**

Aucun fichier du projet ne contenait la liste des mots : `bpe_liste.json` a des
fragments (`' stuck'`, `' del'`, `'arm'`), `bpe.json` a des règles. Comparer des mots
générés au vocabulaire de tokens déclarerait **tout** inexistant — `beautiful`, `castle`
et `grandmother` n'y sont pas plus que `strengue`.

`Vocabulaire.py` dérive donc l'ensemble des mots depuis `stories.train.txt` :
**26 107 mots distincts**, écrits une fois dans `data/vocabulaire.json` pour ne pas
relire 450 Mo à chaque analyse.

Convention figée, à appliquer **identiquement** des deux côtés : `re.findall` avec le
motif `[a-zA-Z][a-zA-Z']*` sur le texte en minuscules. Le motif exige au moins une lettre
en tête — sans quoi une suite d'apostrophes forme un « mot » absent de tout vocabulaire.

**Résultats — top-k, taux de mots inexistants (%)**

```
         k=3     k=5     k=10    k=20    k=40    k=80
T=0.4   0.038   0.000   0.346   0.078   0.146   0.035
T=0.8   1.050   0.586   0.429   0.819   0.401   0.632
T=1.2   0.544   0.727   1.132   1.132   1.675   1.886
T=1.6   0.619   1.506   1.685   2.950   4.409   5.284
```

**La température commande, et `k` n'agit qu'à haute température.** À `T=0.4`, passer de
3 à 80 candidats ne change rien de mesurable : la distribution est trop concentrée pour
que le 80ᵉ token pèse quoi que ce soit. À `T=1.6`, le même passage multiplie les erreurs
par huit.

C'est une **interaction** : elle n'apparaît sur aucune des deux coupes prises séparément,
et c'est ce qui justifiait la grille plutôt que deux séries de mesures indépendantes.

**Les mots inventés sont typés**

`velve`, `quey`, `yories`, `yester`, `vels` reviennent sur presque tous les réglages —
des assemblages de fragments BPE individuellement fréquents qui ne forment aucun mot. Pas
du bruit : des chemins probables dans le vocabulaire qui ne mènent nulle part.

**Un artefact de mesure repéré et corrigé**

La case `(k=10, T=0.4)` sortait à 1,256 % quand ses voisines étaient à 0,1 %. Ses
« erreurs » étaient `"'''"`, `"''''"`, `"'i''''''"` — des apostrophes. Le motif initial
`[a-zA-Z']+` les acceptait comme mots. Après correction : **0,346 %**, soit les deux tiers
de la case qui étaient un défaut de l'instrument, pas du modèle.

Corollaire : le vocabulaire a dû être régénéré avec le même motif. Il contenait
1 174 entrées commençant par une apostrophe, devenues inatteignables par la nouvelle
règle. **Celui qui construit la référence et celui qui l'interroge doivent appliquer la
même convention** — le même principe que le vocabulaire gelé du tokenizer.

**Un bug que seule la mesure a révélé**

La première version du mode `topp` calculait le cumul, trouvait la position où il
franchit `p`, et prenait **ce token-là** — sans aucun tirage. Conséquences :

```
p petit  ->  le cumul dépasse p au premier token  ->  toujours le plus probable = greedy
p grand  ->  le token du BORD du noyau, c'est-à-dire le moins probable des retenus
```

Ce que la table a montré immédiatement :

```
         p=0.2   p=0.4   p=0.6   p=0.9
T=0.4   0.000   0.000   0.000    2.532
T=1.6   1.887   8.025  17.787   31.930
```

Des zéros exacts à gauche (le greedy), et jusqu'à **32 %** de mots inventés à droite —
aucun modèle sain ne produit ça.

**Le texte, lui, paraissait correct** : *« Max rolled under the sticks to hide with Spot
until there was an impressive place »* se lit sans soupçon. C'est la mesure qui a révélé
le défaut, pas la lecture.

Le correctif tient en une ligne : le cumul détermine **combien** de tokens garder ; il
faut ensuite tirer parmi eux comme le fait déjà le mode `topk`. Les deux modes ne
diffèrent que sur ce nombre — donné pour `k`, calculé pour `p` — et le `+1` compte : le
token qui *fait franchir* `p` appartient au noyau.

Après correction :

```
p=0.2   Once upon a time there was a little girl named Lucy. She was three years old…
p=0.9   Once upon a time there was a bald man called Jack. He was very naughty…
```

**Outillage ajouté** *(zone verte, écrit par Claude)*

`src/tooling/tracer.py` — `taux`, `table`, `sauver`, `tracer_grille`, `tracer_carte`,
`tracer_compromis`.

Deux décisions de visualisation, prises sur mesure et non sur goût :

- **Palette catégorielle validée** pour le daltonisme (séparation ΔE conforme en
  protanopie, deutéranopie, tritanopie), assignée dans un **ordre fixe** — `k=3` garde sa
  teinte quel que soit le nombre de séries affichées. Trois teintes passent sous 3:1 de
  contraste, d'où la table texte systématique : l'identité d'une série ne dépend jamais
  de la seule couleur.
- **Carte de chaleur en une seule teinte**, du clair au foncé, avec **la valeur écrite
  dans chaque case**. L'étendue des mesures va de 0,035 % à 5,284 %, soit un rapport de
  152 : sur une échelle linéaire, dix-huit cases sur vingt-quatre tomberaient sous 20 %
  d'intensité et deviendraient indistinguables. La couleur donne la forme, le chiffre
  donne la précision.

**Ce qui m'a bloqué** *(observé pendant la session)*

- **Un plantage machine.** Le tueur de mémoire du noyau a supprimé un processus Python de
  **6,2 Go**, et emporté la session graphique avec — le processus tournait dans le
  terminal intégré de VSCode, donc dans le même groupe de contrôle que l'éditeur. Cause :
  une lecture du corpus d'un seul bloc (`open(...).read()` sur 450 Mo, puis `findall`
  dessus, fabrique 80 millions de chaînes avant d'en faire un ensemble de 26 000).
  Mesuré : **5,09 Go d'un bloc contre 0,01 Go ligne par ligne**, pour la même durée.
  Second suspect au même moment, `decoupage_texte.py`, qui construit une liste de
  450 millions de caractères — 3,6 Go rien qu'en pointeurs. Fichier renommé `OBSOLETE_`.
- **Le tri change de référentiel**, rencontré cinq fois. `torch.sort` rend un couple
  `(valeurs, indices)` ; une position dans le tableau trié n'est pas un numéro de token.
  Symptôme mémorable : `'nOnO'` — les tokens 0, 1 et 2 du vocabulaire.
- **`y[:k]` pour « les k plus probables »** : c'est « les k premiers du vocabulaire ».
  Sans tri, le classement n'existe pas.
- **Un tenseur glissé dans `seq`** au lieu d'un entier : PyTorch bascule en indexation
  multidimensionnelle et `c[seq]` sort avec une forme absurde.
- **`np.array(dict)`** emballe le dictionnaire entier dans un tableau à zéro dimension —
  il n'en extrait pas les valeurs. Et même s'il le faisait, `cpt` et `cptot` n'ont pas
  les mêmes clés : une case sans erreur est absente du premier. Il faut passer par les
  clés, pas par les positions.
- **`defaultdict` est une classe**, pas une méthode : elle s'utilise à la création. Et le
  type de défaut suit l'usage — `int` pour compter, `list` pour collectionner.
- **Le triple emballage** : `gen['topk']` est une liste de 100 listes de 24 dictionnaires,
  et `['texte']` rend **une chaîne**, pas une collection. Boucler dessus donne ses
  caractères.
- **Un tri par boucle Python** contre `torch.sort` : 40,90 ms contre 0,083 ms, soit
  **493×**. Sur une génération de 384 tokens, 15,7 s contre 0,032 s.

**Ce qui reste**

- Le **taux de répétition**, métrique opposée qui manque encore. Sans elle, la table
  pousse vers `T=0.4` — précisément le réglage qui fait radoter le modèle
  (*« liked to explore the world around her »* deux fois dans le même texte).
- Le **dépassement de `max_len`** en génération, toujours non traité.
- Les blocs `topk` et `topp` de l'analyse sont deux copies qui ne diffèrent que par un
  nom de paramètre.

---

**Ce que j'ai compris**

- **Pourquoi pas le niveau mot** : il faudrait connaître tous les mots, et à chaque mot
  jamais vu le modèle perdrait le sens. Le vocabulaire explose et l'inconnu est
  irrécupérable. Le BPE règle les deux : tout mot nouveau reste décomposable en
  fragments connus.
- **Pourquoi pas le niveau caractère** : ce n'est pas qu'il n'y arrive pas — mon
  transformer de l'étape 4 en est un et il fonctionne. C'est qu'il paie trop cher :
  3,4× plus de positions pour le même sens, et de la capacité gaspillée à réapprendre
  l'orthographe.
- **Une fusion est une règle, pas un token.** J'ai d'abord stocké la concaténation
  (`' the'`) au lieu du couple (`(' th', 'e')`). Une règle a une partie gauche et une
  partie droite ; la concaténation détruit la frontière, et sans elle l'encodeur ne
  sait pas où couper.
- **L'ordre des fusions est l'algorithme.** Une règle ne peut se déclencher que si ses
  deux morceaux existent déjà : la règle qui fabrique `' the'` lit `he`, qui n'existe
  qu'après une règle antérieure. C'est un ordre de dépendance, et le rejouer dans l'ordre
  reconstruit exactement l'historique de l'entraînement. Prendre « la plus longue
  d'abord » donnerait une segmentation différente de celle vue à l'entraînement :
  le modèle apprendrait sur un découpage et prédirait sur un autre.
- **L'espace collé en tête — le vrai, pas un supposé.** Il donne au fragment son statut
  de mot entier et non de chaîne intra-mot (` the` contre `the` dans *theater*), et comme
  il est *dans* le token, il n'est jamais perdu. Mais ma première version le collait à
  **tous** les mots par principe, y compris là où il n'y en avait pas. Un blanc supposé
  n'est pas un blanc : le mot suivant un `'\n'` n'a pas d'espace devant, aucune règle de
  début de mot ne pouvait se déclencher, et `'One'` sortait en `'O','n','e'`.
- **Le vocabulaire n'est écrit nulle part** : `bpe.json` contient des règles, pas des
  tokens. Je ne trouvais pas `' Once'` dedans, et c'était normal — il est produit par la
  règle 171, `(' On', 'ce')`. Le vocabulaire se dérive des règles, il ne se lit pas.
- **Le vocabulaire doit être gelé avant l'entraînement.** S'il est régénéré après, les
  indices changent et le modèle entraîné devient obsolète : plus personne ne parle la
  même langue. Et ça ne lève aucune erreur — la matrice d'embedding a une taille figée,
  on obtient du charabia silencieux. Même raison pour refuser d'ajouter un caractère
  inconnu à la volée : le vocabulaire deviendrait fonction du texte encodé.
- **Le retour à la ligne conservé** : sans frontière, les histoires se mélangent, mais
  surtout le modèle n'apprend jamais à s'arrêter ni ce qui est plausible en début
  d'histoire. C'est le même rôle que le symbole de frontière du bigramme à l'étape 0.
  Il n'est plus ajouté à la main à l'alphabet après coup : il est une **unité de
  pré-tokenisation à part entière**, vue par l'apprentissage des fusions comme les autres.
- **`s += x` dans une boucle est quadratique** : une chaîne Python est *immuable*, donc
  chaque ajout réalloue et recopie tout. Une liste est mutable et sur-alloue : `append`
  ne recopie rien. Mesuré : 1710 s contre 0,21 s sur 2 Mo.
- **Liste et ensemble ensemble** : la liste garde la position des tokens, donc les
  indices, ce qui est le contrat avec le modèle. L'ensemble n'a pas d'ordre — il n'est là
  que pour rendre le test d'appartenance instantané.
- **Une entame se prolonge, une terminaison ne dit rien.** Les premières fusions apprises
  avec l'espace en tête sont `' t'`, `' a'`, `' s'`, `' w'` : elles prennent l'entame et
  annoncent le début du mot, donc elles disent *quel* mot on lit, et elles se prolongent
  vers la droite en mots entiers (`' the'` au rang 6, `' to'` et `' and'` au rang 11).
  Avec l'espace en queue ce sont `'e '`, `'d '`, `'t '`, `'. '` : des terminaisons
  partagées par des milliers de mots, qui disent seulement qu'un mot s'est fini.
  Résultat mesuré : 433 fragments sans aucun espace dans le vocabulaire final contre 321.
- **`'One'` et `' One'` sont deux tokens pour un même mot, et il en faut deux.** `' One'`
  est un mot à part, avec son sens. `'One'` nu doit servir deux rôles à la fois : le mot
  entier en début de ligne, et le fragment intra-mot (*money*, *bone*, *one-handed*), qui
  n'a pas du tout le même sens. Ça coûte des créneaux de fusion, et c'est le prix à payer.
- **Deux découpages peuvent reconstruire le texte exactement et ne pas se valoir.**
  L'espace en tête et l'espace en queue passent tous les deux l'aller-retour au caractère
  près. Ce qui les départage n'est pas la justesse mais le rendement : 5 749 964 tokens
  contre 5 877 686, ratio 3,4642 contre 3,3889. Un tokenizer correct n'est pas
  automatiquement un bon tokenizer.
- **L'invariant qui règle tout le découpage** : le curseur avance exactement du nombre de
  caractères consommés, à chaque tour, sans exception — et une unité ne consomme jamais le
  blanc qui la suit, il appartient à l'unité d'après. Chaque fois que j'ai violé ça, j'ai
  eu soit une lettre doublée (`' uupon'`), soit un caractère sur deux sauté (`'Oneuo'`),
  soit une boucle infinie.
  - **Ce qu'est un exemple quand les données n'en donnent pas.** Sur les prénoms, une
  ligne était un exemple. Ici j'ai un flux de 22 millions d'entiers et c'est moi qui
  décide où couper. Un découpage régulier interdit à deux fenêtres de partager des
  caractères, donc il n'en existe que `N/T` — environ 88 000. Un tirage aléatoire les
  fait se chevaucher : il y en a ~`N`, soit 256 fois plus. Même corpus, sans une donnée
  de plus.
- **Une fenêtre doit pouvoir traverser une frontière d'histoire.** J'ai d'abord voulu
  l'interdire, en pensant qu'une rupture au milieu d'une fenêtre était du bruit. C'est
  l'inverse : si le modèle ne voit jamais ce qui **suit** une fin d'histoire, il
  n'apprend jamais à en commencer ni à en finir une. Le token de frontière ne sert à
  quelque chose que si on le lui montre en contexte.
- **L'écart train/val dit quoi agrandir.** À la fin du run : train 2,768, val 2,798.
  Trois centièmes. Le modèle ne mémorise pas — il **manque de capacité**, et c'est le
  réseau qu'il faut agrandir. Si `val` remontait pendant que `train` continue de
  descendre, la conclusion serait inverse : trop de modèle pour trop peu de données,
  et c'est le corpus qu'il faudrait agrandir.
- **Le local s'apprend avant le lointain.** Mon modèle écrit une grammaire correcte et
  invente `strengue` ou `eagle` pour *eager* : il sait à quoi un mot doit ressembler
  sans savoir lequel c'est. Il enchaîne des fragments BPE individuellement fréquents.
  La grammaire tient dans quelques tokens de contexte ; la sémantique demande de retenir
  un référent sur des dizaines de positions, et ça coûte des paramètres que je n'ai pas.



**Ce que je ne comprends pas encore / à réviser**

- **Le carré de l'attention.** J'ai répondu que le ratio de 3,41 caractères par token
  divise le coût par 3. C'est vrai pour les parties linéaires en longueur (embeddings,
  MLP), mais chaque position regarde toutes les autres : le coût de l'attention est en
  T². Diviser T par 3,41 divise ce coût par 3,41² ≈ 11,6. C'est ce facteur qui rend
  l'étape 5 possible, et c'est celui que je n'ai pas vu.
- **Ce que mesure le ratio.** J'ai cru que la présence de tokens intermédiaires comme
  `' th'` dans la liste des règles faisait baisser la moyenne. Faux : le 3,41 est mesuré
  sur la sortie réelle (5 846 939 tokens pour 19 918 766 caractères) et seuls les tokens
  finaux y figurent. Les règles intermédiaires sont un chemin de construction, jamais des
  unités de sortie.
- **Pourquoi stocker le couple et pas le token.** J'ai répondu « on perdrait le chemin »,
  ce qui est l'intuition sans la raison : `' the'` seul ne dit pas où couper — `' t'+'he'`,
  `' th'+'e'` ou `' '+'the'` donnent trois découpages différents pour la suite.
  J'avais aussi craint qu'une fusion en écrase une autre (« si on remplace `th`, on ne
  trouvera plus `the` ») : c'est infondé, l'ordre le garantit, chaque règle s'applique sur
  l'état laissé par les précédentes.
- **L'invariant qui rend l'aller-retour exact.** J'ai répondu « le vocabulaire est le
  même des deux côtés ». Ce n'est pas ça : **une fusion ne fait que concaténer**, elle
  n'ajoute et ne retire jamais un caractère. Donc la concaténation des tokens vaut
  toujours exactement `' ' + texte`. C'est ce qui justifie le `[1:]` de mon décodeur,
  et c'est ce qu'un token de remplacement pour caractère inconnu casserait.
- **Pourquoi un `for` ne peut pas découper un texte.** J'ai répondu « `i` reste fixe en
  Python ». Faux : s'il était fixe la boucle n'avancerait jamais. Il est **réaffecté** à
  chaque tour depuis l'itérateur, ce qui **écrase** ce que le corps a laissé dedans —
  mesuré, `i` prenait bien 10, 11, 12, 13, 14, puis repartait de 1, 2, 3, 4. La vraie
  raison est structurelle et je ne l'avais pas énoncée : `range` décide de l'avancée
  **à l'avance**, et elle vaut toujours 1 ; un découpage en unités avance de la longueur
  de ce qu'il vient de consommer, quantité connue seulement à l'exécution.
- **Quel test attrape quoi.** On m'a demandé lequel de mes découpages faux passait le
  contrôle « mêmes caractères, mêmes comptes ». J'ai répondu celui qui perdait les
  retours à la ligne — c'est l'inverse, celui-là échouait bruyamment (`'\n'` compté 0 fois).
  Celui qui **passait** est `mot = ' ' + mot` posé *après* avoir lu le mot : il donne au
  mot l'espace qui le **suit**, tous les blancs se décalent d'un mot (`" Once upon atime"`),
  mais les caractères et leurs comptes sont identiques. Un contrôle qui ignore l'ordre ne
  prouve pas la reconstruction : seul `''.join(tokens) == texte` le fait.
- **Débuts et fins de ligne ne sont pas symétriques.** J'ai supposé qu'il y avait « plus
  de fins de mot différentes que de débuts » — ce n'est pas l'axe. Il y a exactement
  autant de débuts que de fins de ligne (132 580). Ce qui diffère est la concentration :
  1 760 mots distincts en début de ligne dont les 100 premiers couvrent 90,1 %, contre
  9 470 en fin de ligne dont les 100 premiers ne couvrent que 46,1 %. Une ligne commence
  dans une position grammaticalement contrainte (`Once`, `The`, `They`, un prénom) ;
  elle finit sur la queue d'une phrase quelconque, et presque tout le vocabulaire peut
  terminer une phrase. C'est ce déséquilibre qui rend le dédoublement cinq fois moins
  cher en tête qu'en queue — et je ne l'avais pas vu.
  - **Ce qu'est un lot.** J'ai répondu que son principe est « de balayer les 1000 fenêtres
  avant de recommencer, pour éviter les biais liés à un seul passage ». Ça décrit un
  mélange d'époque, pas un lot. Un lot, c'est `B` séquences qui traversent la passe avant
  **ensemble** et produisent **un** gradient et **un** pas d'optimiseur — la moyenne sur
  les 8192 positions. Deux raisons d'être : le gradient moyenné sur 8192 positions est
  bien moins bruité que sur 256, et le GPU traite les 32 séquences en parallèle dans à
  peu près le temps d'une seule. Boucler dessus me donnait 32 pas bruités au lieu d'un
  bon, et 0,8 s au lieu de 0,1 s.
- **Pourquoi `.to(device)` casse un paramètre.** J'ai répondu que « torch a besoin de
  tous ses poids à l'endroit des calculs ». C'est vrai mais c'est un autre sujet — la
  cohérence d'appareil, que PyTorch signale bruyamment. Le vrai problème est autre :
  `.to(device)` est une **opération**, donc son résultat est un nœud intermédiaire du
  graphe, pas une feuille. Or seules les feuilles accumulent `.grad`. Mesuré :
  `créé puis .to(device) → is_leaf=False, grad=None` contre
  `créé avec device= → is_leaf=True, grad=ok`. Ma boucle Adam plantait sur
  `float * NoneType`, et le paramètre réellement optimisable serait resté sur le CPU.
- **La position dans la fenêtre.** En expliquant le tirage aléatoire, j'ai vu le gain en
  nombre de fenêtres mais pas l'autre effet : dans un découpage régulier, un token donné
  occupe **toujours la même place** dans sa fenêtre. Le modèle apprendrait une
  corrélation entre contenu et position qui n'existe pas dans la langue. Le tirage
  aléatoire fait voir chaque token à toutes les positions — c'est ce qui rend les
  embeddings de position utilisables.
- **Nuance sur « plus d'informations ».** J'ai dit que le recouvrement donne « des
  fenêtres avec des informations différentes ». Il donne des **points de vue**
  différents sur la même information : de nouveaux couples (contexte, cible), pas de
  nouvelles données. C'est pourquoi la courbe finit par s'aplatir malgré des fenêtres
  inédites à chaque pas — seul un corpus plus grand repousse ce plafond.

---

#### 15/08/2026 (suite) — la répétition, et le réglage qui trompe les deux métriques

> Log factuel tenu par Claude. Rubriques de compréhension laissées à Ethan.

**Pourquoi tous les chiffres de la section précédente ont changé**

Le défaut de top-p corrigé, les 4 800 textes ont été régénérés. Le fichier
`generation20260815-0230.json` a été remplacé : les tables de mots inexistants notées
plus haut portent sur la donnée d'avant et ne sont plus comparables ligne à ligne.
La forme, elle, tient — même monotonie, mêmes ordres de grandeur.

**La seconde métrique**

Le taux de mots inexistants ne mesure qu'un côté. Poussé seul, il désigne le réglage le
plus prudent, celui qui répète. Il fallait la métrique opposée : la proportion de
**4-grammes de mots vus plus d'une fois**.

Le piège, et il est unique : le comptage se fait **texte par texte**, seuls les deux
nombres sortent vers la case. Compté sur toute une case d'un coup, les cent textes
partageant `once upon a time there` produiraient un taux énorme — or répéter une formule
d'un texte à l'autre n'est pas un défaut. C'est le radotage **interne** qu'on cherche.

**Résultats — taux de répétition (%)**

```
top-k                                      top-p
      k=3    k=5    k=10   k=20   k=40   k=80        p=.2   p=.4   p=.5   p=.6   p=.8   p=.9
T=0.4 0.270  0.924  0.922  0.702  0.380  1.321      0.000  0.000  0.198  0.495  6.813  1.460
T=0.8 1.267  0.815  0.839  0.218  0.369  0.272      0.067  5.396  3.105  0.491  1.822  0.346
T=1.2 0.737  0.465  0.038  0.120  0.110  0.096      0.941  0.260  0.192  0.352  0.567  0.070
T=1.6 0.694  0.361  0.112  0.029  0.031  0.000      0.664  0.354  0.000  0.036  0.029  0.000
```

Les deux tables varient **en sens inverse** de celles des mots inexistants. C'est le
contrôle : si elles allaient dans le même sens, une des deux mesures serait fausse.

**La dégénérescence**

Deux cases de top-p sortent du lot d'un facteur dix : 6,813 % et 5,396 %. Ce n'est pas
du bruit. Dans la case `p=0.8, T=0.4`, **12 textes sur 100** entrent en boucle et n'en
sortent plus ; le pire fournit à lui seul 72 des 183 répétitions de la case.

```
« He put on his hat and his hat and his hat and his hat and… »
« She will be a good truck. She will be a good truck. She will be… »
```

C'est le phénomène décrit par **Holtzman et al., 2020, *The Curious Case of Neural Text
Degeneration*** — l'article qui a introduit top-p, précisément contre ça. La mesure l'a
reproduit sur ce modèle-ci.

Conséquence sur la métrique : elle est **à queue lourde**. Deux textes sur huit cents
fixent l'échelle du graphique et écrasent les vingt autres points contre l'axe. Une
variante plus robuste existe sur les mêmes compteurs — la proportion de textes contenant
au moins une répétition (12 % et 10 %, contre 6,8 % et 5,4 % en taux poolé) : elle répond
à « à quelle fréquence le modèle déraille » plutôt qu'à « combien de texte est gâché ».
Non écrite pour l'instant.

**Le graphique de compromis, et ce qu'il dit**

Un point par réglage, répétition en abscisse, mots inventés en ordonnée. Les réglages
qu'aucun autre ne bat sur les deux critères se lisent en bas à gauche.

- **top-k** : `k=3, T=0.4` à (0,270 ; 0,000). Tout ce qui est à sa droite est dominé.
  Seul `k=10, T=1.2` lui échappe, à (0,03 ; 0,78) — moins de radotage, payé en mots
  inventés. Arbitrage réel, pas erreur de mesure.
- La forme d'ensemble est un L : la température fait descendre la répétition et remonter
  les mots inventés. Le compromis est visible à l'œil, il n'était jusqu'ici que supposé.

**Le piège, et c'est le résultat le plus utile de la journée**

Le meilleur point de top-p est `p=0.2, T=0.4` à **(0,000 ; 0,000)**. Zéro partout.

Or `p=0.2` à basse température ne conserve qu'un ou deux tokens à chaque pas : c'est du
greedy déguisé, et les cent textes de cette case sont quasi identiques. Les deux métriques
le déclarent optimal parce que **ni l'une ni l'autre ne mesure la diversité**.

Maximiser la vraisemblance donne du texte propre et mort. Deux métriques opposées ne
suffisent pas si elles laissent un angle mort commun : il manque une troisième mesure —
nombre de textes distincts par amorce, ou vocabulaire employé rapporté au nombre de mots.
Sans elle, la table conduit vers le réglage qui ne raconte jamais qu'une seule histoire.

**Deux détails d'outillage qui ont coûté du temps**

- Trois fichiers (`mesures_*.json`, `carte_*.png`, `courbes_*.png`) dérivent d'**une**
  chaîne de nom. Réutiliser la même variable pour une seconde métrique écrase les trois
  sans un mot. La métrique fait donc partie du nom, pas seulement du contenu.
- Les versions de bibliothèques étaient recopiées à l'identique dans quatre appels.
  `importlib.metadata.version` les lit dans les métadonnées du paquet **sans l'importer** :
  plus de `import torch` à trois secondes pour une chaîne, et le fichier ne peut plus
  mentir sur ce qui a servi.

---

#### 16-17/08/2026 — le pre-norm débloque la profondeur

> Log factuel tenu par Claude. Rubriques de compréhension laissées à Ethan.

**La question ouverte depuis le 14/08**

Six blocs ne descendaient pas sous **6,32**, le plancher d'entropie unigramme du corpus
(6,3149) : le modèle n'apprenait que la fréquence des tokens. Deux blocs marchaient.
L'initialisation avait été mise hors de cause par l'expérience — 2000 pas, aucun effet.
Restait le placement des normalisations.

**La correction**

```
post-norm   x = LN( x + f(x) )      la normalisation est SUR le chemin résiduel
pre-norm    x = x + f( LN(x) )      la normalisation est SUR l'entrée de la sous-couche
```

En post-norm, six blocs = **douze** renormalisations du flux résiduel. Le gradient
n'atteint plus le bas. En pre-norm le flux traverse le réseau sans jamais être recalé ;
seules les sous-couches voient une entrée normalisée.

Contrepartie obligatoire : plus rien ne borne `x` à la sortie du dernier bloc, d'où une
**normalisation finale** avant `W_out` — deux paramètres, hors des listes par bloc.
Oubliée, elle ne lève aucune erreur : elle dégrade.

**Ce qu'il fallait toucher** : quatre passes avant vivantes (entraînement, validation,
`genere_prenom`, `generation.py`), plus la déclaration, `params`, la sauvegarde et la
relecture. Une seule copie oubliée fait diverger l'architecture entre entraînement et
génération, en silence.

**La mesure qui tranche, à 250 pas**

```
6 blocs   post-norm            6.3664     ne dépasse pas le plancher unigramme
6 blocs   pre-norm complet     5.0173     apprend
2 blocs   pre-norm complet     4.6706     (contrôle de non-régression, était 4.7648)
```

**Le run de 30 000 pas — loss d'entraînement, à budget de pas égal**

```
pas       2 blocs post-norm dim=512     6 blocs pre-norm dim=384
3 000     2.1130                        1.9513
15 000    1.7137                        1.5361
30 000    1.6132                        1.4283

perplexité finale        5.02                        4.17   (validation : 4.32)
paramètres               8,63 M                     12,39 M
```

**La réserve, et elle est réelle** : le modèle profond a aussi **44 % de paramètres en
plus**. L'expérience ne sépare pas l'effet de la profondeur de celui de la taille. Ce
qu'elle établit sans ambiguïté, c'est que la profondeur est devenue *utilisable* — elle
ne l'était pas du tout auparavant.

**Pas de surapprentissage** : écart entraînement/validation de 0,035, soit 2,5 %. Et la
loss a plafonné sur les 1000 derniers pas (1,4323 → 1,4283). Le budget est consommé :
12,39 M paramètres appellent ~250 M tokens selon Chinchilla, le run en a vu 369 M.
Le levier suivant est le corpus et la taille, pas le nombre de pas.

**Le texte produit**

Cohérent sur plusieurs phrases, avec une structure narrative tenue :

```
Once upon a time, there was a little boy named Tim. He loved to play with his
toys. One day, he found a big box in the attic. It was dark and full of old things.
```

Les défauts restants sont de deux sortes : des mots inventés (`swamf`, `microwaf`) et des
phrases grammaticalement correctes mais absurdes (`The water was going to rain`). La
première catégorie est ce que mesure le banc d'essai du 15/08 ; la seconde ne l'est par
aucune métrique écrite à ce jour.

**Point d'outillage** : `runs/save_runs/<date>/log.txt` existe désormais. Son absence
avait empêché toute comparaison chiffrée avec les runs précédents — seuls les
checkpoints subsistaient, et il a fallu les rouvrir un par un pour retrouver trois
valeurs de loss.




#### 23/08/2026 — noms, provenance et graines : rendre un run rejouable


**Le nom d'un fichier fait partie de sa donnée**

Le nom contenait la date du checkpoint mais pas d'horodatage de la génération, donc deux
fichiers issus du même modèle portaient le même nom : c'est ce qui a rendu la destruction
possible.

Trois mécanismes envisagés pour l'empêcher :

- **un entier qui s'incrémente** : nécessitait une mémoire, ne disait rien du moment où le
  run avait été fait — avant ou après telle modification — et n'apportait aucune
  traçabilité temporelle ;
- **attendre que l'horloge change** : embêtant, il faut attendre, vérifier des minuteurs ;
- **le refus d'écrire** : net et bruyant. En cas de problème, on le sait.

J'ai choisi le refus, en réservant le nom à l'ouverture du fichier, avant les calculs. Le
prix est de laisser le fichier ouvert pendant tout le run, et qu'une exécution interrompue
laisse derrière elle un fichier de 0 octet — que l'analyse choisira, puisqu'il porte
l'estampille la plus récente. En échange, la vérification se fait avant la partie longue :
plus de plantage après vingt minutes de calcul perdues.

**La provenance**

Un fichier de résultats doit pouvoir, seul, répondre à trois questions :

- **refaire** — relancer la même génération : checkpoint, graine, amorces, grilles ;
- **comparer** — dire ce qui diffère d'un autre run : pas d'entraînement, forme du modèle ;
- **interpréter** — savoir ce que les chiffres veulent dire. Une perplexité de 4,32 ne
  signifie rien sans le corpus et le vocabulaire sur lesquels elle est mesurée. C'est ce
  qui justifie d'y inscrire le nombre de paramètres, la perplexité et le tokenizer, qui ne
  servent ni à relancer ni à comparer.

Le critère pour retenir un champ : est-il obtenable à partir des autres, et sert-il à l'une
de ces trois questions ? Par exemple `dim`, `num_heads` et `head_dim` sont liés par une
relation : je n'en ai gardé que deux sur trois, parce que trop d'informations superflues
nuisent à la compréhension et à la lisibilité.

Le chemin du checkpoint ne suffit pas. Il dit **où**, pas **quoi** : si le `.pt` est
déplacé ou supprimé, il ne vaut plus rien. Les faits qu'on veut conserver — pas, loss,
perplexité, nombre de paramètres, forme du modèle — sont donc recopiés dans le JSON.
Quelques dizaines d'octets, et ils survivent au fichier qu'ils décrivent.

**La graine**

Une seule graine pour tout le run aurait ruiné la mesure : les 25 répétitions d'une même
amorce au même réglage seraient reparties du même état du générateur et auraient produit
25 textes identiques. La diversité aurait valu zéro partout, et j'aurais conclu que tous
les réglages sont dégénérés — alors que ce n'aurait été qu'un artefact de mon dispositif.

**Reproductible ne veut pas dire identique.** Ça veut dire *rejouable* : relancer le
programme redonne exactement les mêmes 24 020 textes, tous différents entre eux.

J'avais proposé un compteur stocké dans un fichier. Le défaut n'était pas le compteur, mais
sa persistance : un compteur qui survit entre les runs fait que le même programme lancé
deux fois tire des graines différentes — ce qui détruit la reproductibilité au lieu de la
donner. Et il place l'état du run dans un fichier extérieur que rien ne protège. Or
l'identité d'un appel existe déjà dans la boucle : il n'y avait rien à stocker.

La dérivation ne peut pas être une addition : 24 + 3 donne le même résultat que 23 + 4,
alors que ces deux appels n'ont pas la même identité et ne devraient donc pas recevoir la
même graine.

**Le run raté**

La température était modifiée par un facteur resté en place lors du passage d'une plage à
une liste prédéfinie — celle que j'avais justement créée pour pouvoir l'enregistrer. Les
températures employées n'étaient donc pas les bonnes : 0,16 à 0,64 au lieu de 0,4 à 1,6.
Vingt-cinq minutes de run à jeter.

Je l'ai vu en comparant la grille enregistrée aux valeurs présentes dans le fichier.

> **Leçon** : lors d'un changement qui ne conserve pas la forme, ou qui paraît simplifier,
> toujours regarder où cette modification peut avoir une répercussion, même minime.

**Performance**

Sortir le chargement du checkpoint de la boucle a rapporté plus que le coût du chargement
lui-même — 0,21 s gagnées pour 0,12 s de lecture — parce qu'à l'intérieur de la boucle il
allouait et libérait 141 Mo de VRAM à chaque appel, ce qui malmène l'allocateur bien au-delà
du temps de lecture. Le run est passé de 2 h 04 à 34 min.

Libérer de la mémoire vive n'aurait rien changé : rien n'était à l'étroit de ce côté-là.
Le goulot est le calcul — mais pas celui qu'on croit. À chaque nouveau token je recalcule
toute la séquence depuis le début : pour 100 tokens, cela fait 1+2+…+100 ≈ **5 050 passes
de position au lieu de 100**, soit un facteur ~50 de travail perdu. Les clés et valeurs des
positions déjà produites ne changent jamais ; les garder en cache rendrait chaque pas
linéaire au lieu de quadratique.

**Ce que les chiffres disent du modèle**

Entraîner plus longtemps éloigne de l'optimum : aucun intérêt en performance, seulement en
comparaison, et un risque de surapprentissage.

Deux rapports à ne pas confondre :

| | valeur | ce qui le fait baisser |
|---|---|---|
| tokens vus / paramètres | 29,8 (optimum ~20) | agrandir le modèle |
| tokens vus / corpus | 3,25 époques | agrandir le corpus |

On est déjà au-dessus de l'optimum en tokens par paramètre : il n'y a pas de temps de
calcul supplémentaire à investir. La VRAM est très en dessous de son maximum, autour de
30 %. C'est donc le **corpus** qui limite : l'agrandir fait baisser le nombre de relectures,
et c'est lui qui permettra ensuite d'agrandir le modèle.
`data/tinystories-raw-train.txt` fait 2,1 Go et je n'en utilise que 430 Mo.

**Ce que les quatre défauts ont en commun**

Ils sont tous liés à la reproductibilité et à la compréhension après coup. Quand je suis
dans le code, je comprends, et je ne pense pas à la relecture du moi de dans une semaine.

Mais ce cadre n'explique pas la température, qui m'a coûté vingt-cinq minutes le soir même,
sans aucun problème de lisibilité. Le fil commun est plus mécanique :

> **Chaque fois, une même information existait en deux exemplaires, et rien ne garantissait
> qu'ils restent d'accord.** Le nom et le contenu. La donnée et le sous-titre tapé. La
> graine appliquée et la graine enregistrée. La grille notée et la grille employée.

D'où la règle : **ne jamais laisser un fait exister en deux exemplaires — le dériver de sa
source, ou le vérifier automatiquement.** `textes_par_amorce` vaut `nb // taille` et non 25 ;
le nombre d'amorces se compte au lieu de s'écrire. Ces deux-là ne peuvent plus mentir.

La température montre la limite exacte de la règle : la source était bien unique, mais la
boucle la transformait après coup. **Une source unique ne suffit pas si quelque chose modifie
la valeur entre l'enregistrement et l'usage.** D'où le contrôle *annoncé contre employé*,
qui est la version « vérifier » quand la version « dériver » est impossible.


#### 28/08/2026 — SSM ou réseau « liquide » : pourquoi les modèles d'espace d'états

> Discussion avec Claude, orientation pour après l'étape 5. Rien n'est écrit à ce jour.

**Une racine commune.** Les deux familles décrivent un état caché qui évolue en temps
continu, régi par une équation différentielle, puis discrétisée pour tourner sur une
machine. C'est pour ça qu'elles se ressemblent de loin. Elles diffèrent sur **une seule
propriété : la linéarité**.

**Réseau liquide** (Hasani et al., 2021 ; forme close 2022). La constante de temps de
chaque neurone dépend de l'entrée et de l'état courant — c'est le sens de « liquide » :
la vitesse d'oubli varie selon ce qui est lu. Rien ne se réapprend, les poids sont figés
comme ailleurs ; c'est l'échelle de temps qui est mouvante. Mais l'équation est **non
linéaire en l'état** : pour connaître la mémoire à la position 384, il faut avoir enchaîné
les 383 précédentes. Il faut *marcher*.

**Modèle d'espace d'états** (S4, puis Mamba, 2023). `dh/dt = A·h + B·x`, **linéaire en
l'état**. Une récurrence linéaire se replie algébriquement : toutes les positions se
calculent d'un coup. Il *saute*. Autrement dit, il **s'entraîne comme une convolution et
s'exécute comme une récurrence**.

Mamba rend `A`, `B`, `C` dépendants de l'entrée — c'est exactement l'idée liquide, mais
logée dans une structure linéaire pour que le parallélisme survive.

**Ce que ça coûte, chiffré sur ma machine.** L'entraînement du transformer tourne à
298 ms par pas, 149 min pour 30 000 pas. En version séquentielle, l'arithmétique est la
même mais découpée en 384 lancements au lieu d'un : de l'ordre de **3× plus lent**, pas
400×. Ce n'est pas rédhibitoire aujourd'hui — mais le surcoût **suit la longueur du
contexte**, et allonger le contexte est justement un levier de l'étape 5. Un réseau
liquide ajoute par-dessus un solveur d'équation différentielle, six sous-étapes par pas
de temps.

**C'est l'argument fondateur du transformer.** Vaswani et al. 2017 : la récurrence
interdit de paralléliser à l'intérieur d'un exemple d'entraînement. Mon LSTM était déplié
sur 9 lettres, mon transformer sur 384 — ce n'était pas un choix esthétique. Le SSM est
la tentative de récupérer la mémoire d'une récurrence **sans** reperdre ce parallélisme.

**Pourquoi le SSM pour ce projet**

- remplacer l'attention par un SSM ne touche qu'**une couche** : embeddings, MLP,
  layernorm, boucle d'entraînement et échantillonnage ne bougent pas. Une seule variable
  change, la comparaison est honnête ;
- S4 et Mamba ont des articles complets et du code de référence ; les modèles actuels de
  Liquid AI ne sont pas documentés au point d'être réimplémentés ;
- Mamba a été mesuré **sur du langage**, à cet ordre de grandeur. Les réseaux liquides ont
  fait leurs preuves sur des capteurs et du pilotage, jamais sur le texte.

**Ce que le liquide garde pour lui.** Sur des signaux continus à échantillonnage
irrégulier — capteurs, séries médicales, contrôle — c'est la bonne famille, et elle
battrait un SSM sur ce pour quoi elle a été conçue. Ce n'est pas mon terrain.

**Ce que l'écriture d'un SSM apprendrait, et que le transformer ne peut pas enseigner**

- **le masque causal disparaît** : une récurrence ne *peut pas* voir le futur. On comprend
  alors pourquoi l'attention avait besoin d'un masque — parce qu'elle voit tout par défaut ;
- **le cache clés-valeurs n'existe plus** : l'état est de taille fixe, produire le 384ᵉ
  token coûte autant que le premier. Le facteur ~50 de recalcul mesuré le 23/08 sur ma
  génération est un problème que cette architecture n'a pas.


---

#### 06-07/09/2026 — l'encodeur : mémoïsation, puis passage en flux

> Log factuel tenu par Claude. Rubrique de compréhension dictée par Ethan en réponse à
> quinze questions, puis mise au propre par Claude — le fond et les erreurs sont les
> siens, la rédaction est partagée.

**Le point de départ.** Le bilan du 23/08 désignait le corpus comme facteur limitant :
450 Mo utilisés sur 2,2 Go, 29,8 tokens vus par paramètre contre 20 à l'optimum. Le
verrou n'était pas le modèle mais la chaîne d'encodage, pour deux raisons successives.

**La mémoïsation.** Le découpage d'un mot ne dépend que de lui-même : sur 25 Mo, 39 229
mots distincts pour 4,99 M d'occurrences, soit une redondance de 127×. Une table
mot → tokens ramène le travail à 0,8 % de ce qu'il était.

La validité repose sur une propriété du vocabulaire : aucune fusion ne peut chevaucher une
frontière de mot. Ethan l'a d'abord attribuée à « l'absence d'espace interne dans les
tokens » — vrai mais insuffisant, puisque `' the'` n'a pas d'espace interne et en porte un
en tête. Mesuré sur les 2 000 règles : 1 424 dont le **premier** membre commence par un
espace, **zéro** pour le second. C'est ce zéro qui ferme l'argument, et il est structurel.

**Trois passages en flux.** La mémoïsation seule laissait le corpus entier en mémoire.
Mesures propres sur 100 Mo, un processus par version :

| | RAM/Mo | 2,2 Go | commit |
|---|---|---|---|
| mémoïsé, `decouper` rend une liste | 12,9 | 28 Go | `1ed0971` |
| `decouper` générateur | 3,1 | 6,8 Go | `69f90a0` |
| lecture du corpus par blocs | 2,2 | 4,8 Go | `ccf6e7c` |
| sortie binaire `uint16` | 0,9 | ~1,3 Go | `cdb6fa1` |

La dernière ligne supprime `rencode.py` — renommé `OBSOLETE_rencode.py` — qui réclamait
28 Go pour 500 M tokens. Les indices sont rangés dans la table de mémoïsation : 67 000
conversions au lieu de 560 millions.

**Résultat** : 450 Mo réencodés en 156 s contre ~2 h ; les 2,2 Go tiennent en ~25 min.

**Le dispositif de vérification** (zone verte). Quatre outils, écrits avant les
modifications qu'ils devaient garder :

- `verifier_decoupage.py` — réversibilité du découpage, cas limites, redondance ;
- `essayer_tokkenisation.py` — un mot isolé contre la référence, en redécoupant la sortie
  de l'encodeur committé. 5 397/5 397 mots distincts ;
- `comparer_encodeurs.py` — l'encodeur entier contre sa version git, token pour token ;
- `verifier_rencode.py` — les `.npy` contre ceux d'août, entier pour entier.

Le troisième a servi de garde-fou permanent : chaque modification devait ressortir
`IDENTIQUE`. Les 126 150 131 tokens des trois splits le sont.

**Trois familles de bugs, revenues en boucle** *(observé pendant la session)* :

- **liste contre chaîne** — accumuler dans une liste puis appliquer `.replace` dessus,
  quatre fois de suite ; parcourir une liste avec du code écrit pour une chaîne
  (`for car in mot2` où `car` vaut `'[t]'`, jamais `'['`) ;
- **indice contre valeur** — `for car in mot2` donne la valeur, pas la position ; un
  compteur tenu à côté s'est désynchronisé, testant une position mobile et écrivant
  toujours à la position 0 ;
- **parcourir contre modifier** — `del` pendant un `for`, puis l'inverse : `append` sur la
  liste en cours de parcours, qui fait resservir par la boucle ce qu'on vient d'y déposer.

Deux autres, plus ponctuelles : un `yield` placé à chaque lettre au lieu de chaque mot
complet, et un compteur incrémenté sous condition — donc capable de ne plus avancer du
tout, d'où une boucle infinie.

**Ce que la mesure a corrigé.** Trois de mes estimations étaient fausses, toutes dans le
sens pessimiste, et toutes parce que le banc de mesure chargeait lui-même le corpus avant
d'appeler l'encodeur : 13 puis 8 Mo/Mo annoncés là où un processus dédié en mesurait 3,1
puis 2,2. Corollaire : *un instrument qui partage l'environnement de ce qu'il mesure
mesure aussi l'instrument.*

**Point expliqué par Claude, à retenir : auto-descriptif contre appendable.**

Un format qui porte sa structure dans des délimiteurs ne peut pas être prolongé ; un
format dont la structure est implicite, si.

- **JSON** contient exactement une valeur, terminée par `]` ou `}`. Écrire après le
  terminateur place des octets hors de la valeur — c'est l'`Extra data` du 11/08.
- **Le binaire brut** n'a ni en-tête, ni séparateur, ni fin : la structure est portée par
  la largeur fixe de chaque valeur. Deux morceaux d'`uint16` concaténés forment un fichier
  d'`uint16` valide. Le prix : le fichier ne se décrit pas, il faut connaître le `dtype`
  de l'extérieur — se tromper donne des nombres plausibles et faux.
- **`.npy`** réintroduit un en-tête de 128 octets décrivant forme et type. C'est ce qui le
  rend auto-descriptif, et ce qui l'empêche d'être prolongé.

D'où la structure en deux temps de l'encodeur : binaire brut pendant la boucle, `.npy` en
une passe à la fin. **Auto-descriptif et appendable sont contradictoires** — un format ne
peut pas se refermer proprement et rester ouvert.

**Rubrique de compréhension**

*Rédigé à partir de mes réponses aux questions de Claude, sans relire le code. Je note
aussi ce que je n'ai pas su répondre : c'est là que je devrai revenir.*

**La mémoïsation : rentable et correcte sont deux questions différentes.**

Ma première réponse a été « les mots reviennent souvent, donc on gagne du temps ». C'est
vrai, mais ça répond à *pourquoi c'est rentable*, pas à *pourquoi c'est permis*. Une table
peut être très rentable et parfaitement fausse.

Ce qui l'autorise, c'est que le découpage d'un mot ne dépend **que du mot** : ni de sa
position, ni de ce qui l'entoure, ni du nombre de fois qu'il est déjà passé. Une fonction
sans mémoire se met en table ; une fonction qui dépend de son passé, jamais. C'est
exactement la frontière avec le transformer : on ne peut pas mémoïser « ce qui suit le mot
*chat* », et c'est même tout le travail de l'attention causale.

Un cas dans mon propre code où la position comptait : le premier mot du corpus, qui ne
porte aucun séparateur et à qui j'en ajoute un. Je l'ai neutralisé en faisant porter
l'espace par la clé de la table.

**Pourquoi aucune fusion ne peut chevaucher une frontière de mot.** J'ai mis plusieurs
essais à formuler ça correctement. J'ai d'abord dit « les tokens n'ont pas d'espace
interne » — vrai, mais insuffisant : `' the'` n'a pas d'espace interne et en porte un en
tête. Puis « on exclut les espaces » — faux : 1 424 fusions sur 2 000 ont un premier membre
qui commence par un espace.

La formulation juste porte sur la **position dans la paire**, pas sur la forme des tokens :
une fusion assemble deux symboles adjacents à l'intérieur d'un mot ; le second a toujours
quelque chose devant lui, donc n'est jamais en position 0, donc ne porte jamais l'espace
initial. Ça découle de la procédure d'entraînement, pas de ce vocabulaire-là — donc ça
vaudra pour tous ceux que je réentraînerai.

**Pourquoi 15× et pas 127×.** La redondance mesurée est de 127, le gain final de ~15. Deux
raisons, et j'en avais trouvé une : le coût par occurrence ne disparaît jamais — parcourir
5 millions de mots et consulter la table reste linéaire, et à 24 Mo ce terme domine déjà.

Celle que je n'avais pas vue : sur un mot **inédit**, ma version est plus *lente* que
l'ancienne. L'ancienne applique les règles avec `str.replace`, écrit en C ; la mienne fait
une boucle Python. Je ne gagne que parce que je le fais 127 fois moins souvent. Le facteur
final est le produit de deux effets qui tirent en sens contraire, ce qui explique qu'il
grandisse avec le corpus : le vocabulaire croît beaucoup plus lentement que le texte.

**Le générateur.** Une fonction qui *rend* une liste la construit en entier ; une fonction
qui *produit* n'a jamais qu'une valeur à la fois en mémoire. La liste n'existe pas. Que la
fonction produise 3 valeurs ou 441 millions, l'empreinte est la même.

Un seul parcours suffisait parce que la table se remplit **en avançant** : rien dans ma
boucle ne regarde en arrière ni en avant. Ce que je n'avais pas vu : si j'avais mis deux
boucles — une pour compter, une pour encoder — la seconde n'aurait rien trouvé. Pas
d'erreur, pas de message : un générateur épuisé se comporte comme une séquence vide.

**La lecture par blocs.** J'ai perdu plusieurs essais à vouloir faire tomber les blocs sur
un séparateur. C'était inutile : le mot en cours de construction est une variable locale du
générateur, et le générateur est **une seule invocation** qui voit tous les blocs. Un mot
commencé à la fin d'un bloc se termine au début du suivant. Ça ne marche que si un seul
`decouper` voit toute la lecture — mes premières versions le relançaient bloc par bloc,
d'où les mots coupés, d'où ma tentative d'aligner les blocs, qui traitait le symptôme.

**Les entiers ne sont pas le but, ils sont le moyen.** Remplacer des chaînes par des entiers
dans la même liste Python ne gagne rien : une liste stocke 8 octets de pointeur par élément
quoi qu'elle pointe, et grâce à la table il n'existe que ~2 080 objets distincts dans les
deux cas. Le facteur 4 vient de **sortir du conteneur Python** — un `uint16` dans un tableau
NumPy occupe 2 octets. Les entiers sont ce qui rend le binaire possible, et le binaire ce
qui rend l'ajout à la suite possible.

La conversion token → indice est **absorbée par la table** : elle voyage avec la
tokenisation, dans la même branche, 67 000 fois au lieu de 560 millions. Ce qui l'autorise
est que le vocabulaire est **gelé** — un indice calculé une fois reste valable. C'est la
contrepartie de la décision du 11/08 : faire planter l'encodeur sur un caractère inconnu
plutôt que de l'ajouter à la volée. Les deux décisions se répondent à un mois d'écart.

**Ce que l'aller-retour ne teste pas.** J'ai répondu « il faut aussi comparer les
efficacités » — hors sujet. L'aller-retour teste la **réversibilité** : recoller les tokens
redonne le texte. C'est faible : `[' the']` et `[' t', 'h', 'e']` la satisfont toutes deux.
`comparer_encodeurs` teste l'**identité** : la même suite, élément par élément.

L'écart entre les deux est exactement ce qui me menaçait. Une tokenisation différente mais
recollable serait passée sans bruit, et mes checkpoints — entraînés sur des indices précis —
seraient devenus illisibles. Le modèle aurait continué à tourner en produisant du charabia.
**Le danger n'est pas ce qui plante, c'est ce qui passe.**

**Pourquoi les vérificateurs d'abord.** Si j'avais réécrit `encode()` puis cherché à
vérifier, je n'aurais eu que la nouvelle version à comparer à elle-même, ce qui ne prouve
rien. Le code, git l'aurait sauvé ; mais `data/` n'est pas versionné, et les `.npy` d'août
étaient irremplaçables. Une référence se capture **avant** le changement, et elle doit
couvrir ce que git ne couvre pas.

**Les trois familles de bugs.** Elles sont revenues plusieurs fois chacune :

- **l'objet n'est pas du type que le code suppose** — `.replace` sur une liste ; un parcours
  écrit pour une chaîne appliqué à une liste ; itérer un fichier, qui donne des lignes ;
- **l'indice et la valeur se désynchronisent** — `for car in mot2` donne la valeur, pas la
  position ; le compteur tenu à côté dérive, le test regarde une position mobile et
  l'écriture frappe toujours la position 0 ;
- **on modifie ce qu'on parcourt** — `del` pendant un `for`, puis l'inverse, `append` sur la
  liste parcourue, qui fait resservir par la boucle ce qu'on vient d'y déposer.

Ce qu'elles partagent, et que je n'avais pas vu : **Python ne refuse pas.** Liste, chaîne et
fichier sont tous parcourables et indexables ; une liste modifiée pendant un `for` continue
d'être parcourue. Le code tourne et produit du plausible et faux — 1 248 « mots » au lieu de
200 000, et l'aller-retour qui passe quand même. Ce sont des erreurs de **sens**, pas de
**syntaxe**, donc l'interpréteur se tait. Ce ne sont pas les messages d'erreur qui les ont
trouvées, ce sont les vérificateurs. Il ne me faut pas plus d'attention, il me faut un
contrôle qui compare à une référence.

**`rencode.py` n'a jamais été corrigé.** Ses 28 Go n'ont pas été résolus, ils ont cessé
d'être demandés : le fichier n'existait que parce que l'encodeur produisait des chaînes.
J'avais coupé le travail en « encoder en chaînes » puis « convertir en entiers », ce qui
obligeait 910 Mo de JSON à transiter entre les deux. Cet intermédiaire n'existait que pour
raccorder deux étapes qui n'avaient pas besoin de l'être. **Une étape qui n'existe que pour
réparer la sortie de la précédente signale une frontière mal placée.**

**Ce que ça débloque, et pourquoi le corpus décidait de la taille du modèle.** Je n'avais
pas su tenir ce raisonnement en entier.

Chinchilla : un modèle de N paramètres demande ~20 N tokens. Mon dernier run est à 29,8
tokens par paramètre — donc le modèle est *trop petit* pour le calcul que je lui ai
consacré. Le réflexe serait de l'agrandir, mais mon split d'entraînement ne fait que
113,6 M tokens et j'étais déjà à 3,25 époques : agrandir sans plus de texte revient à
relire davantage, donc à mémoriser au lieu d'apprendre.

Et ce n'est pas le matériel qui bloquait — la VRAM était à 30 %. C'est la **donnée**.
Les 2,2 Go donnent ~560 M tokens, donc ~28 M paramètres à une seule époque, plus du double
du modèle actuel. Avant aujourd'hui, produire ces tokens demandait ~8 h et 28 à 49 Go de
RAM : c'était impossible sur cette machine. Maintenant, 25 min et 1,3 Go.

**Le corpus ne limitait pas la qualité du modèle, il limitait sa taille possible** — et la
taille possible décide de ce que le modèle peut apprendre.

**Ce que je n'ai pas su répondre seul**, et sur quoi revenir : la distinction entre
rentabilité et validité d'une mémoïsation ; le coût C contre Python sur un mot inédit ; le
piège du générateur parcouru deux fois ; ce que teste réellement l'aller-retour ; la cause
commune des trois familles de bugs ; le raisonnement de Chinchilla en entier.

 — à écrire par Ethan, sans relire le code.

---

#### 08-09/09/2026 — corpus complet, puis le banc d'échelle

> Log factuel tenu par Claude. Rubrique de compréhension laissée à Ethan.

**Le corpus complet.** `prepare_stories.py` réécrit en flux (positions en mémoire, texte
sur disque, mélange des positions à permutation identique — sorties octet pour octet
égales à l'ancienne version, vérifié à 20 et 300 Mo). Mémoire plafonnée à ~456 Mo quelle
que soit la taille, contre ~6,9 Go auparavant.

2 717 221 histoires retenues, 274 écartées pour caractères hors ASCII, 2,18 Go nettoyés.
Encodage en 35 min : **552 375 763 tokens** (train 497 M, val 27,6 M, test 27,7 M), aller
-retour exact sur les trois splits. 4,4× le corpus précédent.

**Le banc d'échelle**, 9 h de GPU, six configurations, 90 min de chronomètre chacune,
protocole iso-calcul. Résultats détaillés dans le README. En deux lignes : courbe en U
avec un optimum vers 12 M de paramètres, et **divergence des deux configurations en
`dim=640`**, à pas d'apprentissage constant de 0,001.

**Trois erreurs de méthode, toutes de mon fait, toutes corrigées après mesure :**

- la sonde qui estimait le temps par pas retranchait un démarrage supposé de 12 s à 40 pas
  de mesure. Quand le démarrage réel varie, c'est cette constante inventée qui domine :
  361 puis 497 ms pour la même configuration à cinq minutes d'écart. Remplacée par un
  arrêt au chronomètre — on ne suppose plus, on coupe ;
- la sortie des runs était capturée en mémoire et écrite en fin de run, ce qui rendait
  aveugle pendant 90 minutes. Passée en écriture directe ;
- le tri des checkpoints était lexicographique : `sauvegarde_935` passe après
  `sauvegarde_1496`. J'ai donc supprimé les checkpoints finaux des deux premiers runs en
  croyant garder le dernier. Même famille que les bugs de la veille — comparer des
  chaînes là où il fallait comparer des nombres, sans que rien ne proteste.

**Une fonction de validation extraite** (`Bruit.py`) : la boucle de validation de
`transformer.py` rendue appelable, recevant les poids, le lot, les données et l'appareil.
Vérifiée contre le banc — 2,7558 mesuré contre 2,7587 journalisé au même pas. C'est la
brique nécessaire au balayage du bruit et de la quantification de l'étape 8.

**Rubrique de compréhension** — à écrire par Ethan, sans relire le code. Questions ouvertes
pour la remplir :

- pourquoi une instabilité apparaît-elle quand la largeur augmente, à pas d'apprentissage
  constant ? Qu'est-ce qui grandit avec `dim` dans la mise à jour d'un poids ?
- pourquoi un protocole iso-calcul répond-il à une question différente d'un protocole
  iso-pas, et laquelle des deux voulais-tu ?
- la courbe en U a-t-elle un creux réel à 12 M, ou ce creux est-il un artefact de
  l'instabilité ? Quelle mesure trancherait ?
- pourquoi `A1-petit`, avec 45 750 pas contre 17 750, fait-il moins bien que `A2` alors
  qu'il a vu 2,6 fois plus de tokens ?

---

## Log des `PSEUDOCODE`

Tenu par Claude. Une ligne par usage.

| Date | Étape | Sujet demandé | Revu à froid ? |
|---|---|---|---|
| 27/07/2026 | 1 | Relais 2 — gradient de la loss par rapport aux logits (softmax + cross-entropy → « distribution moins 1 à la cible ») | pas encore |
| 27/07/2026 | 1 | Relais 3 — gradient par rapport aux poids W (dépôt du relais 2 dans la ligne du contexte) + accumulation/moyenne sur le corpus | pas encore |
| 28/07/2026 | 1 (MLP) | Passe arrière complète du MLP à travers 2 couches : sortie (p−y, ∂W2/∂b2), traversée tanh (⊙ 1−h²), couche cachée (∂W1/∂b1), retour dans l'embedding (découpe de ∂E en tranches → lignes de C). Transposées pour propager, produits extérieurs pour les gradients de poids. | pas encore |
