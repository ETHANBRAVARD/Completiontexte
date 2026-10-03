# Réponses au questionnaire de bilan

Mes réponses, dans leur dernière version. Orthographe et vocabulaire corrigés, rien
reformulé sur le fond. Les passages en italique sont des compléments factuels — chiffres
et précisions — ajoutés au fil de la relecture.

---

## 1. Fondations

**1. Distribution conditionnelle et bigramme.** La distribution conditionnelle, c'est la
distribution de probabilité d'obtenir une lettre sachant que la lettre qui précède est
telle lettre. Le bigramme par comptage estime donc la fréquence d'apparition d'une lettre
sachant la lettre d'avant.
*Complément : l'hypothèse de Markov — le bigramme suppose que tout ce qui précède la
lettre d'avant ne change rien. C'est le défaut que chaque étape suivante s'emploie à lever.*

**2. Log-vraisemblance négative plutôt que taux d'erreur.** Je ne savais pas.
Ce que j'ai compris du calcul : le réseau donne un score pour chaque token, le softmax en
fait des probabilités, on regarde la probabilité du bon token, et on passe au `−log` parce
qu'on passe ainsi du produit à la somme. On cherche à minimiser, et en minimisant on
maximise la probabilité du bon token.
*Complément : le taux d'erreur n'a pas de gradient. C'est une fonction en
escalier, nulle partout où elle existe — rien à descendre.*

**3. 2,45 contre ln 27.** La baseline uniforme veut dire que tous les tokens ont la même
probabilité. L'écart mesure donc l'amélioration du bigramme par rapport à un résultat
uniforme.
*Quantitativement : 0,85 nat, soit 1,22 bit. C'est ce que vaut la connaissance de la lettre
précédente.*

**4. Perplexité.** C'est l'exponentielle de la perte. C'est la donnée qui parle le plus,
parce que c'est en gros le nombre de tokens entre lesquels le modèle hésite.
*Réserve : comparable seulement à vocabulaire identique.*

**5. Lissage du bigramme.** Certaines paires n'apparaissent jamais dans le corpus. Comme on
passe au logarithme, leur perte vaut `−log 0 = +∞`. Ça rend leur tirage impossible, alors
que ce n'est pas impossible — c'est juste censé être très peu probable. Et normaliser n'y
change rien : ce qui vaut 0 reste à 0. La solution est de donner un petit poids de base à
toutes ces paires improbables.

**6. Embedding.** Si on avait gardé un un-parmi-N, le produit aurait donné exactement le
même résultat, puisqu'on ne multiplie que par des 0 et un 1. Ce qui change, c'est qu'à
chaque apparition d'un token, sa ligne est corrigée par le gradient. Les tokens qui
apparaissent dans des contextes semblables reçoivent des gradients semblables, donc leurs
lignes finissent proches — on garde l'idée qu'il y a des proximités entre caractères, par
exemple que les voyelles ont des rôles similaires.

**7. Règle de la chaîne.** C'est une règle mathématique : la dérivée d'une composée,
`(g∘f)'(x) = g'(f(x)) · f'(x)`. Pour la perte : `∂L/∂entrée = (∂L/∂sortie) × (∂sortie/∂entrée)`.
À chaque dérivée dans le réseau, il faudrait recalculer tout ce qui se trouve entre ce poids
et la sortie. Pour l'éviter, on fait d'abord les dernières couches puis on remonte : chaque
gradient intermédiaire est calculé une fois et réutilisé.

**8. Initialisation aléatoire.** Avec des poids à zéro, l'information est bloquée dès le
premier passage, donc on ne peut pas mettre à jour les couches précédentes. Et si tous les
poids sont égaux, le réseau fonctionne comme un seul neurone — tout le monde bouge de la
même façon, la distribution initiale ne change jamais.
*Le facteur `1/√n` compense la croissance de la variance de la somme avec le nombre de termes.*

**9. `loss.backward()`.** Il remonte sur l'ensemble des poids et calcule les gradients, en
faisant `+=`. Ce qu'il ne fait pas : mettre à jour les poids, ni remettre les gradients à
zéro.
Le `+=` vient de ce qu'un tenseur peut servir plusieurs fois dans la passe avant : la perte
dépend de lui par plusieurs chemins, et leurs contributions s'additionnent. Une matrice de
poids est utilisée à chaque position de chaque séquence du lot — 12 288 fois pour un lot de
32 × 384. On perd la décomposition, mais seul le total sert à la mise à jour.

**10. Pourquoi l'étape 2.** PyTorch est vraiment optimisé, il permet d'améliorer les
performances et de gagner en vitesse, et il débloque les étapes suivantes, beaucoup plus
difficiles avec le système rudimentaire que j'avais mis en place.
*Complément — la raison principale : la vérification. Les deux implémentations doivent donner
les mêmes gradients — c'est le seul moyen de savoir si ma backprop à la main était juste.*

## 2. Récurrence

**11. MLP à fenêtre fixe contre RNN.** Le MLP a une mémoire fixe en nombre de tokens : il
mémorise par exemple les 20 caractères précédents. Le RNN garde un état qui varie à chaque
passage de token, et cet état peut en théorie retenir une infinité de tokens — même si en
pratique il est limité.

**12. Rétropropagation à travers le temps.** Ce qui coûte, c'est que tout défile : on ne
peut pas paralléliser, tout doit être fait un par un, donc pas d'utilisation efficace du GPU.
*Et non le nombre de paramètres, qui est faible puisqu'ils sont partagés dans le temps.*

**13. Gradient qui explose ou s'évanouit.** Comme on a un enchaînement d'un grand nombre
d'états, la dérivée est le produit des dérivées. Et comme les facteurs ont tous à peu près
la même tendance, ça fait soit zéro très fort, soit l'infini très fort — même pour un
facteur de 0,9 ou 1,1. À exactement 1, il ne se passe rien. Même cause dans les deux cas.

**14. Portes du LSTM.** Une porte est un ensemble de poids qui prend en entrée le token
courant et l'état précédent, et qui ressort à quel point on garde cette information en
mémoire. Comme l'état est mis à jour par addition, il n'y a plus ce gradient multiplicatif.
*Complément : ce qu'elle résout est l'évanouissement du gradient, pas un problème de capacité mémoire.*

**15. Ce que le LSTM ne sait pas faire.** Sans attention, on a une fonction fixe qui applique
la même mémoire à tout. L'attention permet de garder en fonction de l'importance, y compris
des choses plus anciennes, sans appliquer tout le temps la même fonction de mémoire.
*Complément : le moment de la décision : le LSTM décide quoi garder à
l'écriture, sans savoir ce qui servira. L'attention garde tout et décide à la lecture.*

## 3. Attention et transformer

**16. `W_q`, `W_k`, `W_v`.** `W_q` produit la question qu'on pose. `W_k` est le lecteur de
carte d'identité : les mots passent par lui et ressortent leur étiquette, qu'on compare à
la requête. `W_v` produit ce qu'il y a dans le tiroir — ce que le token transmet s'il est
retenu.
*Complément : le softmax n'est pas `W_v` — il s'applique aux scores `Q·K`, sans aucune matrice.*

**17. `Q @ K.T`.** Ça donne la pertinence d'un couple : l'importance d'un mot en mémoire pour
la génération du mot en cours. Forme `(T, T)`, et la case `(i, j)` est le score entre la
requête de la position `i` et la clé de la position `j`.

**18. Division par `√head_dim`.** Comme pour le `1/√n` : plus la tête est grosse, plus la
somme grandit, donc on renormalise.
*Sans elle, le softmax sature et son gradient s'annule.*

**19. Masque triangulaire.** C'est le fait de ne pas pouvoir regarder les réponses futures
pour s'en servir — sinon on aurait cent pour cent de la réponse suivante. Triangulaire parce
qu'on annule tout ce qui est au-dessus de la diagonale.

**20. `-inf` plutôt que zéro.** Parce que le masque est additif : ajouter zéro ne changerait
rien. On ne veut pas que ce soit très peu probable, on veut que ce soit impossible.

**21. Plusieurs têtes.** Lors de la génération d'un mot, il peut y avoir plusieurs
informations importantes : la nature grammaticale, le contexte de l'histoire, le pluriel ou
le singulier, le temps. Ce sont des bouts différents de l'histoire, donc il faut plusieurs
têtes. En réalité je pense que ça a surtout été expérimenté plus que théorisé — on ne sait
pas vraiment quelles informations chaque tête va récupérer.

**22. `W_o`.** Elle mélange les têtes pour avoir une seule information plutôt qu'une par
tête.
*La forme est déjà correcte avant elle : ce n'est pas une question de format mais de mélange.*

**23. Connexion résiduelle.** Garder `x` règle le problème du gradient : lors des remontées,
on peut quand même avoir un impact sur les premières couches.
*Second effet manquant : le bloc n'apprend qu'une correction à ajouter, et peut devenir
transparent.*

**24. `layernorm`.** On prend chaque token — pas chaque couche — et on met ses composantes à
moyenne nulle et variance 1, dans le but de stabiliser. Puis on applique le gain `g` et le
biais `b`, appris, qui rendent au réseau le contrôle de l'échelle.

**25. Post-norm contre pre-norm.** C'est avant ou après l'addition de `x` avec `f(x)`, donc
sur la couche directement ou sur le flux résiduel. Le problème du post-norm, c'est qu'on
applique douze layernorms successives sur `x`, et les informations du début n'ont plus rien
de ce qu'elles étaient.
*L'effet décisif est sur la passe arrière : le gradient n'atteint plus les premières couches.*

**26. Normalisation finale en pre-norm.** Sans elle, `x` peut s'emballer, vu qu'il n'est
jamais renormalisé, et écraser l'information obtenue par les couches.
*Conséquence précise : des logits de grande amplitude font saturer le softmax.*

**27. Le `dim*4` du MLP.** C'est pour augmenter sa puissance de reconnaissance : chaque ligne
de `W_1` est un détecteur, et la ReLU ne garde que ceux qui s'allument. Le retour à `dim` est
imposé par le format du résiduel. Le facteur 4 est une convention, pas un optimum démontré.

**28. `pos_emb`.** Elle garde en mémoire la position de chaque token, et on la donne comme
information au réseau. Elle est apprise.
*Complément : sans elle, l'attention est indifférente
à l'ordre : le modèle verrait un sac de tokens.*

**29. 6,37 contre 6,3149.** L'entropie unigramme garde la distribution des mots : le modèle
a juste appris quel mot était plus fréquent que les autres, mais pas du tout quel mot va
avec quel mot. Être au niveau de ce plancher prouve qu'il n'a acquis aucune information
conditionnelle.
*Complément : l'uniforme vaudrait `ln 2080 ≈ 7,64`, bien au-dessus.*

## 4. Tokenisation BPE

**30. Algorithme BPE.** On cherche la paire de tokens la plus représentée, on la fusionne en
un seul token, et on répète. On s'arrête à un nombre d'étapes choisi. Et on garde en mémoire
chaque fusion — l'ordre compte, encoder un mot nouveau consiste à rejouer les fusions dans
le même ordre.

**31. Pourquoi 2 080 tokens.** 2 080 est beaucoup plus petit que 50 000, qui ferait une
couche de sortie énorme. Et plus gros que 256 caractères, qui n'ont aucun sens sémantique.
L'avantage principal, ce sont les morceaux intermédiaires : un mot jamais vu se reconnaît
par sa construction, préfixes et suffixes — en français beaucoup de mots viennent du latin.
*Second compromis : un petit vocabulaire allonge les séquences, et l'attention coûte en carré
de la longueur.*

**32. Pré-tokenisation.** Elle évite d'avoir des tokens à cheval sur deux mots, avec un
espace interne. C'est ce qui rend la mémoïsation correcte.

**33-34. Pourquoi aucune fusion ne chevauche une frontière de mot.** Garanti par la
pré-tokenisation, et vérifié sur le vocabulaire : 1 424 fusions dont le premier membre
commence par un espace, zéro pour le second. C'est sur cette propriété que repose la
mémoïsation — le découpage d'un mot ne dépend que de lui-même.

**35. Facteur de la mémoïsation.** Quasiment cent — exactement 127, soit 4,99 M
d'occurrences pour 39 229 mots distincts. C'est le nombre moyen de fois où chaque mot
réapparaît. Plafond théorique : le gain réel est moindre, le reste de la chaîne n'en
profitant pas.

**36. Le `set` dans `tokkenisation`.** `bpeset` est l'alphabet de base, utilisé pour vérifier
que chaque caractère du mot est connu. Un `set` plutôt qu'une liste parce que le test
d'appartenance y est à coût constant, contre proportionnel à la taille pour une liste — un
demi-million de tests sur l'encodage complet.
*Le second `set`, `lettre_non_reconnu`, ne peut jamais contenir plus d'un élément puisqu'on
sort immédiatement. Coût mesuré : 15 ms sur 156 s. Laissé tel quel.*

**37. `uint16`.** C'est le plus petit format qui contienne amplement tous nos tokens. Le
maximum est 2¹⁶ = 65 536. Au-delà, on ne pourrait plus dissocier les tokens : il y en aurait
plus que de façons de les écrire.
*Et NumPy ne lèverait pas d'erreur — il replierait silencieusement.*

**38. Générateur et `yield`.** Un générateur, c'est une fonction classique où le `return` est
remplacé par un `yield`. Le `yield` met la fonction en pause jusqu'à l'appel de la valeur
suivante, en gardant son état. On ne détient donc jamais la collection entière en mémoire.

**39. Les trois transformations de l'encodeur.** Je ne m'en souvenais plus.
*Générateur (12,9 → 3,1), lecture par blocs (3,1 → 2,2), sortie binaire `uint16` (2,2 → 0,9).
En facteurs : ×4,16, ×1,41, ×2,44 — le générateur en premier, mais l'écart est moins écrasant
qu'en différences absolues. (Remarque que j'ai soulevée et qui était juste.)*

## 5. Entraînement

**40. Corps de la boucle.** On tire un lot au hasard dans le corpus. On passe entièrement
dessus — ce qui génère non pas un token mais une distribution de probabilité pour chaque
position. On calcule la perte par rapport à ce qui était réellement écrit. Puis
l'accumulation des gradients avec `backward`, l'écrêtage, Adam, et il ne faut pas oublier la
remise à zéro, puisqu'on a fait `+=`. Et de temps en temps une validation.

**41. `tireur_de_lot(tableau, B, T)`.** `B` est le nombre de séquences dans un lot, donc le
nombre de décalages tirés au hasard dans le corpus. `T` est la longueur des séquences qu'on
tire. On prend `T` tokens consécutifs en entrée, et les mêmes décalés d'un cran en cible.

**42. Le tirage n'est pas un mélange.** C'est un tirage aléatoire du point de départ, avec
remise. On peut donc tirer deux fois le même endroit. Conséquence : tirer le volume d'une
époque ne veut pas dire avoir parcouru le corpus — la couverture réelle vaut `1 − e^(−k)`,
soit 63 % pour une époque de volume, 95 % pour trois.

**43. Axe des classes dans `cross_entropy`.** Elle attend un format particulier ; si on ne
donne pas le bon, le résultat n'est pas cohérent.
*Complément : elle doit savoir quel axe énumère les 2 080 tokens pour y appliquer le
softmax. Inversé, elle normaliserait sur les positions au lieu du vocabulaire.*

**44. Ce qu'Adam garde.** Il garde les gradients aux bonnes proportions, avec une moyenne
mobile et une moyenne mobile du carré — l'équivalent d'une moyenne et d'un écart-type. Ça
évite des gradients qui partent dans tous les sens et donne une cohérence structurelle.
*Rôle manquant du `√v` : il rend le pas indépendant de l'échelle du gradient, donc le même
`lr` convient à des paramètres très différents. Et c'est pour ça qu'un checkpoint pèse trois
fois les poids.*

**45. `m_hat` et `v_hat`.** Correction de biais : `m` et `v`
partent de zéro et sous-estiment les premiers pas, mais pas dans les mêmes proportions —
le rapport `m/√v` serait environ trois fois trop grand au pas 1. La correction disparaît
d'elle-même quand `t` grandit.

**46. `eps`.** Il empêche la division par zéro quand `v_hat` est nul, et borne le pas maximal.

**47. Pourquoi l'écrêtage malgré Adam.** Adam renormalise par rapport au gradient passé : il
garde une cohérence **temporelle**, coordonnée par coordonnée. L'écrêtage, lui, agit sur
l'ensemble des coordonnées à un instant donné — c'est un écrêtage **spatial**. Les deux ne
font pas le même travail.
*Mécanisme : si un lot inhabituel produit un gradient énorme sur toutes les coordonnées à la
fois, `v` ne l'a pas encore vu et Adam laisse passer. Pire, ce gradient entre ensuite dans `v`,
qui reste gonflé pendant des centaines de pas.*

**48. Norme globale contre coordonnée par coordonnée.** L'écrêtage global multiplie tous les
gradients par le même facteur, donc il **préserve la direction** : seule la longueur du pas
change. Plafonner chaque composante séparément écraserait les grandes et pas les petites, donc
déformerait le vecteur — on ne descendrait plus le long du gradient.
*Seconde différence : le global ne se déclenche qu'en cas d'anomalie d'ensemble ; un écrêtage
par coordonnée agirait en permanence sur les coordonnées naturellement grandes.*

**49. Les trois formes de `config_pas`.** Cosinus, `1/√i`, et pas constant. `amorce` est le
nombre de pas de croissance avant d'atteindre le pas de croisière ; `fin` est le nombre de pas
sur lesquels s'applique la décroissance. **Le quatrième élément est le pas de base**, la valeur
du plateau.
*Classement cos > racine > constant : seul l'écart contre le pas constant (0,099) est établi.
Cos contre racine vaut 0,011, sous le plancher de bruit.*

**50. Pourquoi un échauffement.** Au début, les poids sont tirés au hasard, donc on est
sûrement très loin — et c'est aussi là que le gradient est le moins pertinent. On avance donc
prudemment.
*Deux causes qui se cumulent : `v` n'est bâti que sur un ou deux échantillons, donc très
bruité ; et le gradient initial reflète l'initialisation plus que les données. Mesuré :
`dim=640` à 10⁻³ sans échauffement, la norme passe de 0,49 à 8,39 en 1 200 pas, pic à 179.*

**51. Pourquoi une décroissance.** On a cessé de se rapprocher du minimum : on tourne autour,
on le dépasse dans un sens puis dans l'autre. Réduire le pas permet d'affiner. Et ça donne une
fin claire — à partir de là on rejoint le minimum le plus proche.
*Formulation quantitative : avec un pas fixe, on converge vers une **boule** dont le rayon est
proportionnel au pas, à cause du bruit d'échantillonnage. Réduire le pas rétrécit la boule.
Mesuré : les 4 000 derniers pas du run de 120 000 valent 0,064.*

**52. `1/i²` et `1/ln i`.** Conditions de Robbins–Monro : `Σ lr = ∞` (la somme **diverge**,
donc la distance parcourable est infinie) et `Σ lr² < ∞` (la somme des carrés **converge**,
donc le bruit s'amortit). `1/i²` échoue sur la première — distance bornée. `1/ln i` échoue sur
la seconde — décroît trop lentement. `1/i` satisfait les deux.
*Mon `racine`, en `1/√i`, viole formellement la seconde ; il fonctionne parce qu'on n'entraîne
pas jusqu'à l'infini.*

**53. Enregistrer `config_pas` et `i`.** Avec eux on connaît tout le schedule : on peut
recalculer le pas à n'importe quelle position, passée comme **future**. Avec la seule valeur
courante, on saurait où on en est mais pas où on va — reprise impossible. Et le checkpoint se
documente lui-même.
*Ce n'est pas une question de poids : une valeur serait plus légère que quatre.*

**54. `.to(device)` et Adam.** `.to()` est une **opération** : elle crée un nouveau tenseur,
qui n'est plus une **feuille** du graphe. Or `backward()` n'écrit de `.grad` que sur les
feuilles. `p.grad` reste donc à `None` et Adam plante. D'où l'ordre dans mon code : créer le
tenseur **sur le GPU**, puis marquer `requires_grad`.

## 6. Lois d'échelle et budget

**55. Chinchilla.** C'est un ratio **tokens par paramètre**, pas un nombre de balayages du
corpus. Environ **20 tokens par paramètre**. Chez moi : 24,38 M de paramètres → 488 M de tokens
visés, pour un corpus de 497 M.
*Ça répond à « budget de calcul fixé, comment le répartir entre taille et tokens », avec
`C ≈ 6ND`. C'est l'optimum **à calcul fixé**, pas le meilleur modèle possible — mon run à
35 tokens/paramètre continuait de s'améliorer.*

**56. Iso-pas contre iso-calcul.** On compare à nombre de pas égal ou à temps de calcul égal.
L'iso-calcul est plus pertinent parce que le temps est la ressource réellement limitée — ce
qu'il y a derrière les pas, c'est de la puissance de calcul.
*Les deux répondent à des questions différentes : l'iso-pas isole l'efficacité d'apprentissage,
l'iso-calcul mesure le résultat avec le budget réel. Mon résultat profondeur/largeur l'illustre :
égalité à pas égal (0,007), la largeur gagne à calcul égal (0,021) — l'avantage vient
entièrement de la vitesse. Et un résultat iso-calcul dépend de la machine.*

**57. Les deux branches de la courbe en U.** Ce sont **deux formes de sous-apprentissage**, pas
un sur- et un sous-apprentissage. À gauche, le petit modèle fait énormément de pas et voit
énormément de tokens, mais n'a pas assez de paramètres : il **manque de capacité**. À droite,
le gros a la capacité mais coûte cinq fois plus par pas, donc il **manque de tokens**.

**58. Pourquoi la courbe était fausse.** Ce qui était confondu avec le budget de tokens, c'est
**l'instabilité du pas d'apprentissage**. Un seul pas de 10⁻³ pour toutes les tailles, sans
échauffement ni écrêtage — or le pas maximal stable varie comme l'inverse de la largeur. La
branche droite mesurait donc en partie la divergence.
*Preuve : avec échauffement et écrêtage, `A4` passe de 4,91 à 1,5199 et rejoint `A2` (1,5228).*

**59. Vérification de la loi de largeur.** Une batterie de sondes, mais surtout une
**prédiction** : à `dim=512`, 6·10⁻³ tenait, donc seuil > 6·10⁻³ ; `512 → 640` fait ×1,25, donc
le seuil prédit tombe à **4,8·10⁻³**. Mesuré à `dim=640` : 4,5·10⁻³ tient, 6·10⁻³ diverge. La
valeur prédite est dans l'intervalle.

**60. Ce que le facteur huit suggère.** L'instabilité a **deux composantes**. Une
**structurelle** — un modèle large tolère réellement un pas plus petit, et elle survit intacte,
toujours en `1/largeur`. Une **transitoire** — la phase initiale où le gradient n'est pas
fiable, accidentelle et entièrement supprimable. Les garde-fous ne retirent que la seconde.

**61. ×1,76 contre ×1,30.** Sur le portable, la carte est **saturée** dans les deux cas, donc
le coût est proportionnel au travail. Sur la 3090, `A3` ne la remplit pas : une partie attend.
Passer à `A4` l'occupe mieux, donc le surcroît est partiellement absorbé.
*Deux mesures le confirment : le rapport contre le portable croît avec la taille (×2,16 puis
×2,85), et le balayage de lot n'a gagné que 10 %. Conséquence : **la courbe en U dépend de la
machine**, son creux se déplace vers la droite sur la 3090.*

**62. Les 10 % du balayage de lot.** La carte était **déjà saturée** à `lot=32` pour `A4`.
C'est cohérent avec la 61 : `A3` ne sature pas, `A4` sature. En montant de 24 à 42 M on finit
de la remplir, après quoi augmenter le lot fait simplement plus de travail — 346 → 626 ms pour
un lot doublé.

**63. Pourquoi refaire les mesures de stabilité.** Une moyenne sur 64 au lieu de 32 change le
**bruit du gradient** : sa variance est divisée par deux. Moins de bruit, c'est un pas plus
grand supportable — le seuil de stabilité se déplace. Tous mes seuils ont été mesurés à
`lot=32`.
*Seconde raison : à budget de tokens fixé, doubler le lot divise le nombre de pas par deux, et
l'échauffement et la décroissance sont comptés en pas.*

## 7. Génération et échantillonnage

**64. La boucle de `genere()`.** On part d'une amorce, **encodée** en indices de tokens. À
partir des tokens en mémoire, le modèle prédit le **token** suivant — pas le mot. Plusieurs
façons de choisir : pour l'instant top-k avec `k=5`. Le token tiré entre dans le contexte, et
on recommence. À la fin, les tokens sont **décodés** en texte.
*À chaque itération, une passe avant complète est refaite sur toute la séquence — d'où le coût
quadratique.*

**65. La température.** On **divise les logits par `T`** avant le softmax. `T < 1` écarte les
logits et pique la distribution ; `T > 1` les resserre et l'aplatit. `T → 0` donne l'argmax,
`T → ∞` l'uniforme. Chez moi `T = 1,2`.
*C'est une division des **logits**, pas des probabilités — avant la normalisation.*

**66. top-k et top-p.** top-k garde les `k` premiers tokens, donc un **nombre fixe**. top-p
cumule les probabilités triées jusqu'à atteindre `p`, donc une **masse fixe** et un nombre
variable : peu quand le modèle est sûr, beaucoup quand il hésite.

**67. Pourquoi top-p dégénère à basse température.** La température basse **pique** la
distribution : le meilleur token monte vers 0,95. Le cumul dépasse `p = 0,2` dès le premier
token, donc le noyau n'en contient **qu'un**. Avec un seul candidat, la génération devient
**déterministe** — la même amorce redonne toujours le même texte, d'où 95,8 % de redite.
*La dégénérescence vient d'un excès de déterminisme, pas de hasard. Et le tirage dans le noyau
est proportionnel aux probabilités renormalisées, pas uniforme.*

**68. Pourquoi le bug de top-p était invisible.** Le token de la frontière reste un token
**plausible**, donc le texte paraît correct. Entre deux continuations également acceptables,
l'œil ne distingue rien. Un bug qui change **lequel** des bons tokens est choisi ne produit
aucune absurdité visible.
*C'est la mesure qui l'a révélé : 0,000 % de redite à bas `p` — comportement déterministe — et
31,9 % à `p=0,9`. Profil impossible pour une implémentation correcte.*

**69. Le plafond de `max_len`.** `pos_emb` est une table **apprise** de forme
`(max_len, dim)` : il n'existe pas de ligne 385, aucun vecteur n'y a jamais été entraîné.
La ligne qui plante est `x = x + pos_emb[:T]` — si `T > 384`, la tranche ne renvoie que
384 lignes et l'addition échoue sur les formes.
*Le `if nombre_de_car > max_len` est un garde-fou incomplet : il borne les tokens **générés**,
alors que la séquence contient déjà l'amorce.*

**70. La fenêtre glissante.** On ne donne au modèle que les `max_len` derniers tokens, et la
fenêtre avance avec la position. On perd **tout ce qui en sort** : au token 700, le modèle ne
voit plus les 316 premiers. Il peut produire un texte long mais oublie son propre début.
*Ça permet de **produire** du texte long, pas de **tester** la cohérence longue.*

**71. Le coût en `T²`.** Tout le passé est nécessaire pour générer la suite, et une somme de 1
à `n` donne `n²/2`.
*Précision : ce calcul décrit le cas **optimisé**. Dans mon code, chaque token déclenche une
passe complète dont le coût est déjà en `T²`, et il y a `T` passes — donc **`T³`** au total. Ce
qui est recalculé inutilement, ce sont les clés et valeurs de tous les tokens précédents, qui
ne changent jamais.*

## 8. Mesure et méthode

**72. Le plancher de bruit.** C'est la variation minimale qu'on obtient sans rien changer de
pertinent — en ne touchant qu'à la graine, donc aux lots tirés et à l'initialisation. Si nos
écarts d'entraînement sont plus faibles que ça, on ne peut pas distinguer un effet réel du
hasard.
*Chez moi : 0,013, pour un balayage du départ de la décroissance dont l'étendue valait 0,018.
Rapport 1,4 — indissociable.*

**74. Dix écarts positifs.** On peut conclure parce qu'un bruit est censé être réparti
symétriquement autour de zéro. Dix signes identiques ont une probabilité de `2 × (1/2)¹⁰`,
soit une chance sur 512.
*Deux questions distinctes : l'effet est-il **réel** (oui, par le test des signes) et
est-il **important** (non, 0,025, et la validation descendait encore).*

**75. La mesure appariée.** Comparer deux conditions **sur les mêmes échantillons**, et
calculer la différence échantillon par échantillon : tout ce qui varie entre échantillons
s'annule.
*Utilisée trois fois : mêmes lots de validation dans le banc de bruit (dispersion 0,0009 contre
0,0057 sur la perte brute), mêmes amorces et graines dans l'échelle de texte, même graine dans
le run de décomposition.*

**76. Une calibration de 60 pas.** Soixante pas, ce n'est rien, et le **temps de démarrage** —
qui dépend de la machine — domine la mesure. Ce n'est pas lui qu'on veut voir, c'est le régime
permanent.
*Mesuré : `etalon` donnait 401 à 918 ms/pas, le régime établi 253 à 329. Facteur 2,6. Et la
signature est lisible : sur 56 pas, `A1` et `A4` ne diffèrent que d'un facteur 2,3, contre 8 en
régime établi — un coût fixe écrase les petites configurations.*

**77. Pourquoi « elles saturent » était faux.** Les textes ne faisaient que **120 caractères**,
la génération s'arrêtant au premier saut de ligne. Sur 120 caractères il n'y a qu'une vingtaine
de 4-grammes : les métriques tombaient à zéro **par manque de matière**. Ce qui l'a révélé,
c'est ma modification du compteur de sauts de ligne — avec 340 tokens, la corrélation apparaît.
*Et sous 1,30, la saturation est réelle : seuls les mots inexistants gardent un signal
(`+0,386`). L'IA avait fini d'apprendre cette partie-là et continuait sur des choses que ces
critères ne mesurent pas — la cohérence sur la longueur.*

**78. Deux métriques et un angle mort commun.** Le réglage `p=0,2 · T=0,4` donnait 0,417 % de
mots inexistants et 0,477 % de répétition — deux chiffres honorables — pour **95,8 % de
redite**. Les deux premières mesurent la qualité **interne** d'un texte ; aucune ne regarde les
autres textes. Il fallait une troisième métrique qui les compare **entre eux**.

**79. Le `source.py` déposé.** Chaque run emporte le code exact qui l'a produit, donc il reste
rejouable même si `transformer.py` change ensuite — et il change tout le temps. Sans ça, un
résultat devient invérifiable.
*Et `echelle.py` ne modifie jamais le fichier de zone rouge : il le lit, substitue dans une
copie, exécute la copie.*

**80. Pourquoi le banc s'arrête sur une substitution absente.** Parce qu'elle ne planterait
pas. La substitution ne ferait simplement rien, et le run tournerait sur les **valeurs par
défaut** — le banc annoncerait `dim=512` en exécutant `dim=384`. On obtiendrait un chiffre
crédible attaché à la mauvaise configuration. Et on ne veut rien qui ne plante pas.

## 9. Branche 8

**81. Bruit multiplicatif et additif.** L'additif s'ajoute aux poids avec la **même amplitude
pour tous** (`σ = α·max|w|`), le multiplicatif est **proportionnel à chaque poids**
(`σ = α·|w|`).
*Correspondance matérielle : l'**additif** correspond au crossbar analogique, dont la précision
est **absolue** — un pas de quantification constant. Le **multiplicatif** correspond au
flottant, dont la précision est **relative**. D'où « fp8 fonctionne là où int4 échoue ».*

**82. Pente de 2 en log-log.** La perte est **quadratique** autour du minimum — une parabole.
Doubler le bruit quadruple la dégradation.
*Pourquoi exactement 2 : à un minimum le gradient est nul, donc le terme linéaire du
développement de Taylor disparaît et le premier terme non nul est d'ordre deux. C'est donc
aussi un **test** : un modèle non convergé donnerait une pente proche de 1.*

**83. Du bruit aux bits.** Un bruit d'écart-type `σ` rend indistinguables deux valeurs plus
proches que `σ` : c'est la résolution. On compte combien de niveaux tiennent dans la plage
utile, puis on prend le logarithme en base 2.

```
additif        plage 2·max|w|, résolution α·max|w|  ->  log2(2/α) = 8,3 bits
multiplicatif  résolution relative α                ->  log2(1/α) = 4,0 bits
```

**84. Le rapport 2,6.** C'est le rapport des **courbures**, pas des tolérances — celui-là
vaudrait 1,6. Il vient de la pente 2 : à seuil égal, `courbure_A1/courbure_A2 = (6,1/3,8)²`.
Le minimum d'`A1` est 2,6 fois plus **étroit**. Un petit modèle n'est pas seulement moins bon,
il est posé dans une vallée plus resserrée — moins de redondance entre ses poids.

**85. `W_1` contre `W_2`.** Même forme, même nombre de poids, et pourtant deux à trois fois
plus fragile. Ça élimine la **taille** et la **forme** comme explication : la fragilité tient
au **rôle** de la matrice dans le calcul.
*Confirmé par `W_q`, `W_k`, `W_v`, identiques en dimensions, de 2,8 % à 5,6 %. Conséquence :
on ne peut pas prédire la fragilité depuis le schéma, il faut la mesurer.*

**86. Les 8 % de la précision mixte.** Allouer un nombre de bits différent par famille devrait
rapporter beaucoup, puisque l'écart vaut 2,3 bits. Mais **les familles fragiles sont aussi les
plus grosses** : `W_1` et `W_2` pèsent 7 M des 11,4 M de poids. Les bits économisés portent sur
les petites matrices. C'est une moyenne pondérée, et la pondération joue contre.
*Leçon générale : un gain relatif ne dit rien tant qu'on ne l'a pas pondéré par ce sur quoi il
porte.*

**87. `√N` contre `N`.** Mettre `N` cellules en parallèle pour un poids : leurs **conductances
s'additionnent** (le signal croît en `N`), leurs **bruits sont indépendants donc s'additionnent
en quadrature** (le bruit croît en `√N`). Le rapport signal/bruit gagne donc `√N`. Le coût est
`N` cellules, donc linéaire en surface.
*Rendement décroissant : amener `W_2` au niveau de `W_q` demanderait 27 cellules, et l'ensemble
du modèle 14,7 fois la surface.*

## 10. Outillage

**88. `scripts/tokenizer.py`.** C'est un **orchestrateur**, en quatre étapes dans un ordre
contraint : apprendre les 2 000 fusions, reconstruire le vocabulaire, encoder le corpus en
tokens, découper en train/val/test selon les proportions données. Puis les vérifications.
*L'ordre n'est pas décoratif : on ne peut pas encoder avant d'avoir appris les fusions. Il est
en zone verte — il n'implémente rien, il appelle les modules de `src/model/`.*

**89. `scripts/echelle.py`.** Il lance plusieurs configurations sous un protocole contrôlé, en
déposant pour chacune une copie paramétrée de `transformer.py` et en exécutant cette copie. Il
relève le temps par pas, les pas faits et la perte finale. Il sait travailler à pas égal
(`--pas`) ou à temps égal (`--minutes`).
*Séquentiellement, pas en parallèle — un seul GPU. Il ne modifie jamais `transformer.py` :
zone rouge, et reproductibilité.*

**90. `scripts/sonde_gradient.py`.** Il mesure la **norme globale du gradient à chaque pas**,
avant écrêtage. Il ne retire pas l'appel à `clip_grad_norm_` : il dépose une copie avec
`max_norm=1e30`, seuil jamais atteint, et **capture la valeur de retour** de la fonction — qui
est précisément la norme avant écrêtage. La fonction est détournée en instrument de mesure.
*Résultat : médianes de 0,48 à 0,64 sur quatre tailles, indépendantes de la taille du modèle.
D'où le seuil unique à 1,0.*

**91. `scripts/banc_bruit.py`.** Il balaie **alpha**, le niveau de bruit, pour les deux lois, et
mesure l'écart apparié contre la mesure sans bruit. `--par-famille` change **ce qui est
bruité** : une seule famille à la fois au lieu de toutes ensemble. C'est ce qui a produit le
tableau de fragilité.

**92. `scripts/echelle_texte.py`.** Il compare les **textes produits par les checkpoints
successifs d'un même run** — le même modèle à différents niveaux de perte. Il fixe les graines
par amorce pour que la mesure soit appariée : l'amorce `j` tire la même séquence aléatoire à
tous les paliers, seul le modèle change.

**93. `scripts/programme.py`.** Une surcouche d'orchestration : il ne mesure rien, il enchaîne
des appels à `echelle.py`. Quatre raisons — la **portabilité** (bash n'existe pas sous Windows),
les **programmes nommés** (une nuit devient un mot), la **trace de la machine** affichée avant
de démarrer, et le garde-fou `--go`.

**94. Les vérificateurs.** Ils sont en zone verte parce qu'ils **vérifient un résultat sans
révéler comment l'obtenir** — ils posent une question, ils ne donnent pas la réponse.

```
verifier_decoupage      le découpage est réversible
essayer_tokkenisation   un mot isolé se tokenise comme en contexte (5 397/5 397)
comparer_encodeurs      l'encodeur entier contre sa version committée
verifier_rencode        les fichiers de tokens entiers
```

*Leur conjonction a permis d'affirmer que les 126 150 131 tokens sortaient identiques, entier
pour entier, après réécriture complète de l'encodeur.*

**95. `src/model/Vocabulaire.py`.** Il produit la liste des **26 107 mots distincts** du corpus,
et sert à la métrique des **mots inexistants** — pas aux 4-grammes, qui se calculent directement
sur les textes générés sans référence externe.
*C'est la seule des trois métriques qui ait besoin d'une vérité extérieure, et la seule qui
garde un signal sous 1,30.*

**96. `runs/` et `data/` ignorés.** Ils sont beaucoup trop lourds pour un dépôt — 3,3 Go de
données, 484 Mo par checkpoint, 20 Go pour un run. Et git conserve toutes les versions pour
toujours.
*Plus profondément : git versionne la **recette**, pas le **produit**. Le `source.py` et la
graine suffisent à reproduire — deux machines que tout opposait ont redonné 1,9453 contre
1,9456.*

## 11. Lire un résultat

**97. Expliquer 1,1433 et 3,14.** *(version retenue)* Le modèle doit deviner le prochain
morceau de mot parmi 2 080 possibles. Avant d'apprendre, il hésiterait entre les 2 080. Après
dix heures d'entraînement, il hésite entre **trois**. C'est la perplexité de 3,14. L'autre
chiffre, 1,1433, est le même en logarithme — ce que la machine optimise, parce que les
logarithmes s'additionnent là où les probabilités se multiplient.
*La perte est le logarithme du nombre de candidats, **positif** : `1,1433 = ln(3,14)`.*

**98. L'écart train/val de 0,049.** Pris isolément il est dans le bruit et ne signifie rien.
Mais les **dix** écarts sont positifs, ce qui a une chance sur 512 d'arriver par hasard : l'effet
est établi même si aucune mesure seule ne l'établit.
*Et le critère d'inquiétude n'est pas une valeur d'écart : c'est le moment où la validation
**remonte** pendant que l'entraînement descend. Ici elle descendait encore.*

**99. Pourquoi s'arrêter.** Parce qu'on est à iso-calcul, et parce qu'on pourrait entraîner
indéfiniment jusqu'à la mémorisation sans que ce soit la meilleure version.
*Mais la raison déterminante est le **schedule** : la décroissance en cosinus est calculée
relativement à `nb_passage` et atteint zéro au pas 120 000. L'arrêt a été décidé **au
lancement**, pas à la fin. C'est un défaut reconnu du cosinus — il oblige à connaître la durée
d'avance. Indice qu'on n'était pas loin du bout : l'entraînement aussi avait plafonné.*

**100. 1 479 M dans un corpus de 497 M.** Trois époques en volume, et une couverture de
**94,9 %** — `1 − e^(−2,98)`. Pas 100 % parce que le tirage est **avec remise** : rien
n'empêche de retomber sur une zone déjà vue, rien ne garantit d'atteindre les autres.
*Les 63 % correspondent à une seule époque de volume.*

**101. Reconnaître une divergence.** Trois signes. **La perte monte** au lieu de descendre —
`B1-large` : 4,0203 puis 4,4309 puis 4,4512. **Elle se pose près du plancher unigramme**
(6,31), donc le modèle n'a plus que les fréquences. **Et la norme du gradient s'emballe** —
0,49 → 8,39, pic à 179, contre 0,48-0,64 en régime sain.
*Le test décisif : ajouter des pas ne change rien. Un mauvais réglage continue de progresser.
Et l'ordre de grandeur le dit aussi : 4,1992 contre 1,9269, c'est 66,6 de perplexité contre 6,9.*

**102. L'ordre d'apprentissage.** D'abord la fréquence du vocabulaire, puis la cohérence interne
aux phrases, puis la cohérence au long d'une histoire, et en dernier la cohérence des dialogues.
Les dialogues ont l'air d'être la difficulté principale.
*Ce que ça dit : le dernier acquis est le plus **long**. Savoir qui parle exige de suivre qui
est qui à travers la séquence, et aucun indice local ne le donne. Réserve : trois exemples, à un
seul endroit de la courbe — une hypothèse, pas un résultat établi.*

## 12. Pièges

**103. `=` ne copie pas une liste.** Il lie un **second nom au même objet** : les deux pointent
vers le même endroit en mémoire, donc modifier l'un modifie l'autre. Le symptôme était une liste
qui grandissait silencieusement — une fonction allongeait la liste de l'appelant.
*Piège de second niveau : `list(x)` fait une copie **superficielle**. Pour mes listes de
tenseurs, la liste est neuve mais les tenseurs restent partagés.*

**104. Modifier une liste en la parcourant.** Supprimer l'élément `i` décale tous les suivants
d'un cran, la boucle passe à `i+1` et saute ce qui vient d'y arriver. Et c'est silencieux.
*La règle n'est pas absolue : `tokkenisation` fait exactement ça et c'est correct, parce que le
décalage est pris en compte. La formulation juste est « sans tenir compte du décalage ».*

**105. `sauvegarde_935` après `sauvegarde_1496`.** Ordre lexicographique : `'9' > '1'`. Comme
on prenait la dernière sauvegarde, on pensait avoir la bonne et on avait une intermédiaire.
*Coût concret : le banc ne garde que le dernier checkpoint et supprime les autres. Il a donc
**détruit les checkpoints finaux** de la première partie du run. Irrattrapable, sans message.*

**106. `CUDA_VISIBLE_DEVICES=3`.** `nvidia-smi` numérote par **position sur le bus PCI**, le
runtime CUDA trie par défaut **de la plus rapide à la plus lente**. Sur `gpu01`, la 3090 est
l'index 3 pour `nvidia-smi` et l'index 0 pour CUDA — l'index 3 tombait donc sur une GTX 1660.
Correction : `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
*Danger silencieux : le run aurait tourné trois fois plus lentement sur une carte de 6 Gio.*

**107. « No route to host ».** Le DNS publiait **les deux** adresses, et SSH préfère l'IPv6. Le
préfixe `2001:660:330f:b040::` de l'école n'était pas routé depuis le VPN. `-4` force l'IPv4 et
tout a marché.
*Ce qui rendait le diagnostic difficile : GitLab Rezel répondait en IPv6 sur un autre préfixe,
donc l'IPv6 paraissait fonctionnelle, et le message ressemblait à un problème d'accès.*

**108. Ce qui rend un run reproductible.** Le triplet : la **source exacte** (`source.py`
déposé), la **graine**, et les **données**. Changer de serveur a montré que ça tenait — même
programme, même résultat.
*Ce qui ne suffit pas : le chiffre seul, les hyperparamètres notés dans un journal (ils
décrivent ce qu'on croyait lancer), le checkpoint seul, la graine seule sur un code qui a
changé. Et la reproductibilité pratique ne demande pas l'exactitude au bit près — 1,9453 contre
1,9456 malgré Python, torch et l'architecture GPU différents.*

## 13. Vue d'ensemble

**109. Du caractère brut au token d'entrée.**

```
1. pré-tokenisation          découpage en mots, espace rattaché au mot suivant
2. mémoïsation               déjà vu -> lecture ; sinon les 2 000 fusions dans l'ordre
3. passage aux entiers       par le dictionnaire, une fois par mot distinct
4. écriture uint16           par tampons dans un binaire brut, relu en .npy
5. tireur_de_lot             un décalage au hasard, 384 entiers consécutifs
6. c[seq]                    chaque entier devient un vecteur de dim nombres
7. + pos_emb[:T]             on y ajoute le vecteur de position
```

*L'entrée du modèle est un **vecteur**, pas un entier — les étapes 6 et 7 manquaient.*

**110. Du token de sortie au caractère affiché.** Ce n'est **pas** le chemin inverse.

```
1. layernorm finale
2. @ W_out.T + b_out         2 080 logits
3. / température
4. softmax                   une distribution
5. sélection                 top-k=5, renormalisation, tirage
6. alphabet[idx]             l'indice devient une chaîne
7. concaténation             le texte
```

*Deux asymétries. L'étape 5 n'a aucun équivalent à l'entrée : l'encodage est une **fonction**
déterministe, le décodage un **échantillonnage**. Et le décodage ne « défait » rien — chaque
token contient déjà ses caractères, donc décoder est une simple concaténation, là qu'encoder
exige de rejouer 2 000 fusions.*

**111. Ce qui limite aujourd'hui.** *(ma réponse était « le calcul » — corrigée)* C'est le
**corpus**. Deux mesures du run de 120 000 pas le disent : la perte d'**entraînement** a
plafonné elle aussi (1,0924 → 1,0940), donc ce n'est pas un manque de temps ; et le
surapprentissage a commencé, signature d'une limite de **données**. Arithmétiquement : 497 M de
tokens, quand un modèle de 80 M en demanderait 1,6 milliard.
*Le calcul limite la **vitesse d'itération**, pas la qualité atteignable — et il s'est desserré
avec la 3090. Le contexte est une limite d'un troisième type : il ne borne pas la perte, il
borne **la question qu'on peut poser**.*

**112. Ce que je referais autrement.** Pas grand-chose — tout m'a appris quelque chose. Mais je
ne referais pas les batteries de tests, puisqu'on en connaît maintenant les valeurs : le nombre
de caractères, la meilleure structure, les niveaux de bruit. Je lancerais directement le run qui
a donné le meilleur résultat. Et je pousserais pour avoir des mesures différentes — le but reste
l'apprentissage, donc voir des résultats variés.
