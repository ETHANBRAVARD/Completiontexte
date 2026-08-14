# Sur quoi partir — note de démarrage (étape 0)

Recherche du 24/07/2026. Trois questions : quel corpus, quel cours, et à quoi on
reconnaît que l'étape 0 est réussie.

**Résumé.** Commencer sur `names.txt` de *makemore* (32 000 prénoms anglais, 223 Ko),
en suivant l'épisode 2 de *Zero to Hero* — **pas** l'épisode 1. Rejouer ensuite le même
code sur la liste de prénoms de l'INSEE, en français. Raison du détour par l'anglais :
c'est le seul corpus pour lequel il existe un **chiffre de référence publié**, et sur
ton premier modèle tu auras besoin de savoir distinguer « mon code est faux » de « le
modèle est comme ça ».

---

## 1. Le corpus de l'étape 0

Un modèle de bigrammes au niveau caractère a besoin d'une liste de mots courts, un par
ligne. Deux candidats sérieux.

### Option A — `names.txt` (makemore)

`raw.githubusercontent.com/karpathy/makemore/master/names.txt` — **228 145 octets**
vérifiés, ~32 000 prénoms anglais en minuscules, un par ligne, licence MIT.
Extraits : `emma`, `olivia`, `ava`, `isabella`, `sophia`. Source : les prénoms les plus
donnés aux États-Unis en 2018 (`ssa.gov`).

- Vocabulaire de **27 symboles** : 26 lettres + un symbole de frontière. Rien à décider,
  aucun accent, aucun tiret, aucune casse.
- C'est le corpus exact de la vidéo, donc **ta loss est comparable à celle annoncée à
  l'écran** (de l'ordre de 2,45 pour le modèle par comptage — à vérifier toi-même dans
  la vidéo plutôt que de me croire). C'est le seul argument qui compte à ce stade.
- Inconvénient : c'est de l'anglais, et ce n'est pas ton projet.

### Option B — Fichier des prénoms de l'INSEE

Page officielle : `insee.fr/fr/statistiques/8595130`, mise à jour 2026, données 1900–2025.
Deux fichiers pertinents :

| Fichier | Contenu | Taille |
|---|---|---|
| `prenoms-2025-liste_csv.zip` | liste des prénoms donnés au moins 3 fois depuis 1900 | 377 Ko |
| `prenoms-2025-nat_csv.zip` | prénom × sexe × année × effectif, ~725 000 lignes | 4 Mo |

Colonnes du fichier national : `SEXE` (1/2), `PRENOM`, `PERIODE` (année), `VALEUR`
(effectif arrondi à 5 près), `RANG`.

- Du français, et un corpus qui a du sens pour toi. Le fichier national permet en plus de
  **pondérer par fréquence**, ce que `names.txt` ne permet pas : `Jean` compte alors pour
  ce qu'il pèse réellement, et non pour une ligne parmi 32 000.
- Le prix à payer : accents (`é`, `è`, `ï`, `ç`), tirets (`Jean-Pierre`), apostrophes,
  casse, et une ligne `_PRENOMS_RARES` à écarter. Ton vocabulaire ne fait plus 27 symboles
  et **tu dois décider** ce que tu gardes — c'est un vrai choix de modélisation, pas un
  détail technique.
- Aucun chiffre de référence externe : tu ne pourras comparer ta loss qu'à toi-même.

### Recommandation

**A d'abord, B juste après.** Une fois que le modèle tourne sur `names.txt` et que ton
chiffre tombe à côté de celui de la vidéo, rejouer sur l'INSEE est une demi-heure de
travail — et cette demi-heure t'apprend deux choses que l'anglais ne t'apprend pas : que
le modèle ne dépend pas du corpus, et que le choix du vocabulaire est une décision de
conception qui change les résultats.

**Attention à la frontière ici.** Décompresser le CSV, retirer `_PRENOMS_RARES`, choisir
un encodage, découper train/val : c'est du nettoyage de corpus, je peux le faire. En
revanche la **table de correspondance caractère → indice**, ce que tu décides de faire des
accents et des tirets, et la taille finale du vocabulaire sont de la modélisation : à toi.
Dis-moi quand tu veux le corpus, je te livre un fichier propre, un mot par ligne.

---

## 2. Le cours à suivre

**Andrej Karpathy — *Neural Networks: Zero to Hero*** (`karpathy.ai/zero-to-hero.html`).
Huit épisodes, gratuits. Correspondance avec ta feuille de route :

| Ép. | Titre | Durée | Étape |
|---|---|---|---|
| 1 | The spelled-out intro to neural networks and backpropagation: building micrograd | 2h25 | 1 — **voir §3** |
| 2 | The spelled-out intro to language modeling: building makemore | 1h57 | **0 → commence ici** |
| 3 | Building makemore Part 2: MLP | 1h15 | 1 et 2 |
| 4 | Building makemore Part 3: Activations & Gradients, BatchNorm | 1h55 | 1 et 2 |
| 5 | Building makemore Part 4: Becoming a Backprop Ninja | 1h55 | 1 |
| 6 | Building makemore Part 5: Building a WaveNet | 56 min | — (hors feuille de route) |
| 7 | Let's build GPT: from scratch, in code, spelled out | 1h56 | 4 |
| 8 | Let's build the GPT Tokenizer | 2h13 | 5 |

L'épisode 2 couvre l'étape 0 en entier : bigrammes par comptage, symbole de frontière,
matrice 27×27, normalisation en probabilités, échantillonnage, negative log likelihood.
Sa seconde moitié refait le même modèle en réseau de neurones — c'est déjà l'étape 1,
tu peux t'arrêter avant et y revenir.

### Le piège de l'ordre

Le cours commence par micrograd. **Ta feuille de route non**, et c'est délibéré : la
règle du projet est de lire `micrograd` *après* avoir écrit ton backward. L'épisode 1 a
le même problème que le dépôt — il donne la réponse à l'exercice principal de l'étape 1.

Compromis raisonnable le moment venu : regarder la **première moitié** de l'épisode 1,
celle du tableau blanc (dérivée numérique, règle de la chaîne à la main sur une petite
expression), s'arrêter dès qu'il commence à écrire la classe `Value`, écrire ton étape 1,
puis finir l'épisode. Tu gardes le cours, tu perds la triche.

Pour la théorie de l'étape 1 sans risque de copie, l'alternative est **Nielsen, chapitre 2**
(`neuralnetworksanddeeplearning.com/chap2.html`) : la dérivation complète de la
rétropropagation en mathématiques, avec les quatre équations fondamentales. Le chapitre
contient aussi du code Python plus bas — la partie utile est avant.

---

## 3. À quoi on reconnaît que l'étape 0 est finie

Pas « ça tourne sans erreur ». Tu as fini quand tu peux répondre à ça sans rouvrir ton code :

1. **Quel est le score d'un modèle qui ne sait rien ?** Un modèle uniforme sur 27 symboles
   donne une loss de $\ln 27 \approx 3{,}296$ et une perplexité de 27 — il hésite entre
   27 possibilités à chaque caractère. Tout ce que tu produis doit battre ce nombre, sinon
   ton code est faux. C'est ton premier vrai test.
2. **Pourquoi la perplexité vaut $e^{\text{loss}}$**, et pourquoi on l'interprète comme un
   nombre de choix équivalents.
3. **À quoi sert le symbole de frontière**, et ce que le modèle ne peut pas faire sans lui.
4. **Ce qui se passe pour un bigramme jamais observé** dans le corpus, quel effet ça a sur
   la loss, et pourquoi ce n'est pas un cas rare mais la situation normale.
5. **Pourquoi comparer deux loss n'a de sens qu'à corpus et vocabulaire identiques.**

Et un critère non chiffré : les noms générés doivent être mauvais **d'une manière précise**
— prononçables par bouts, incohérents sur la longueur. Si tu peux expliquer pourquoi c'est
exactement ce qu'un bigramme doit produire, tu as compris ce qu'est une distribution
conditionnelle d'ordre 1.

### Pièges connus, à surveiller

- Une probabilité nulle rend le log infini, et une seule suffit à faire exploser toute la
  loss. Le nom du remède est *lissage* (add-one / Laplace) ; le comprendre fait partie de
  l'étape.
- Une loss qui vaut exactement $\ln 27$ après entraînement veut dire que rien n'a été appris.
  Une loss très inférieure à 2 sur ce corpus veut dire que tu évalues sur tes données
  d'entraînement, ou que tu as un décalage d'indice entre l'entrée et la cible.
- Moyenne sur les caractères, pas sur les noms : sinon ton chiffre n'est comparable à rien.

---

## 4. Pour la suite — état des corpus français

Recherche faite maintenant pour éviter une mauvaise surprise à l'étape 3, sans t'avancer
sur ces étapes. Conclusion courte : **rien de prêt à l'emploi en français**, il faudra
fabriquer le corpus.

- **TinyStories en français** : le seul jeu trouvé (`iproskurina/TinyStories-French` sur
  Hugging Face) fait **1 000 histoires / 577 Ko**, sans licence déclarée ni split. Deux
  ordres de grandeur en dessous de l'original (2 M d'histoires). Inutilisable pour un
  entraînement, tout juste bon comme échantillon.
- **L'original anglais** (`roneneldan/TinyStories`) reste la voie la plus sûre pour l'étape 5
  si l'objectif est « du texte cohérent » plutôt que « du texte français ».
- **Français, à fabriquer** : Wikisource français (~390 000 textes, domaine public,
  transcriptions vérifiées) est la meilleure source pour un corpus de 1 à 100 Mo. Gutenberg
  a du français mais son domaine public est celui des États-Unis, pas celui de la France.
- **Lexique 3** (`lexique.org`) : 135 000 formes du français avec fréquences, phonétique et
  syllabation, licence libre. Pas un corpus de texte suivi — mais l'équivalent français de
  `names.txt` en bien plus riche, et une option à garder pour les étapes au niveau mot.

Quand tu y seras, dis-le : construire et nettoyer ce corpus est de la zone verte.

---

## Sources

- [karpathy/makemore](https://github.com/karpathy/makemore) — dépôt, `names.txt`, licence MIT
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — liste et durées des épisodes
- [The spelled-out intro to language modeling: building makemore](https://youtu.be/PaCmpygFfXo) — épisode 2
- [INSEE — Fichier des prénoms](https://www.insee.fr/fr/statistiques/8595130) — fichiers, tailles, colonnes
- [data.gouv.fr — Fichier des prénoms depuis 1900](https://www.data.gouv.fr/datasets/fichier-des-prenoms-depuis-1900)
- [Nielsen — Neural Networks and Deep Learning, chap. 2](http://neuralnetworksanddeeplearning.com/chap2.html)
- [roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) et [iproskurina/TinyStories-French](https://huggingface.co/datasets/iproskurina/TinyStories-French)
- [Wikisource — ressources libres de droit](https://fr.wikisource.org/wiki/Aide:Ressources_libres_de_droit)
- [Lexique.org — manuel de Lexique 3](http://lexique.org/_documentation/Manuel_Lexique.3.pdf)
