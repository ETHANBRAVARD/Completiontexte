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

## Log des `PSEUDOCODE`

Tenu par Claude. Une ligne par usage.

| Date | Étape | Sujet demandé | Revu à froid ? |
|---|---|---|---|
| 27/07/2026 | 1 | Relais 2 — gradient de la loss par rapport aux logits (softmax + cross-entropy → « distribution moins 1 à la cible ») | pas encore |
| 27/07/2026 | 1 | Relais 3 — gradient par rapport aux poids W (dépôt du relais 2 dans la ligne du contexte) + accumulation/moyenne sur le corpus | pas encore |
| 28/07/2026 | 1 (MLP) | Passe arrière complète du MLP à travers 2 couches : sortie (p−y, ∂W2/∂b2), traversée tanh (⊙ 1−h²), couche cachée (∂W1/∂b1), retour dans l'embedding (découpe de ∂E en tranches → lignes de C). Transposées pour propager, produits extérieurs pour les gradients de poids. | pas encore |
