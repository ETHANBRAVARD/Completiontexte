"""Tracé et sauvegarde des mesures d'échantillonnage.

Zone verte (tooling) : graphiques et I/O. Aucune logique de modèle.

Trois fonctions, à appeler depuis analyse_generation.py :

    taux(cpt, cptot)                       -> {(param, temp): taux}
    sauver(mesures, chemin)                -> écrit un JSON relisible
    tracer_grille(mesures, ...)            -> courbes : taux selon la température
    tracer_carte(mesures, ...)             -> carte de chaleur de la grille
    tracer_compromis(mes_a, mes_b, ...)    -> nuage : deux métriques l'une contre l'autre
    table(mesures, ...)                    -> la même chose en texte

Palette validée pour le daltonisme (six teintes, séparation ΔE conforme en
protanopie, deutéranopie et tritanopie). Les couleurs sont assignées **dans un
ordre fixe** : la série `k=3` garde sa teinte quel que soit le nombre de séries
affichées.
"""

import json
from pathlib import Path

# Palette catégorielle validée — ordre fixe, jamais recyclé.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
          "#4a3aa7", "#8a5a2b"]
# Rampe séquentielle : UNE teinte, du clair au foncé. L'ordre est alors évident
# — plus foncé veut dire plus grand — sans avoir à apprendre un sens de lecture,
# et l'information survit au daltonisme.
RAMPE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
         "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
         "#0d366b"]
SURFACE = "#fcfcfb"
ENCRE = "#0b0b0b"
ENCRE_2 = "#52514e"
GRILLE = "#dcdbd6"


def taux(cpt, cptot):
    """Rend {(param, temp): taux} depuis les deux compteurs de l'analyse.

    `cpt` peut être un defaultdict : une case sans aucune erreur y vaut 0,
    ce qui est la bonne réponse. On parcourt les clés de `cptot`, qui les a
    toutes puisque chaque texte y a compté ses mots.
    """
    return {cle: cpt[cle] / cptot[cle] for cle in cptot if cptot[cle]}


def sauver(mesures, chemin, **extra):
    """Écrit les mesures en JSON. Les clés étant des couples, on les aplatit."""
    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    contenu = {"mesures": [{"parametre": p, "temperature": t, "valeur": v}
                           for (p, t), v in sorted(mesures.items(),
                                                   key=lambda x: (x[0][1], x[0][0]))]}
    contenu.update(extra)
    chemin.write_text(json.dumps(contenu, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  mesures écrites dans {chemin}")
    return chemin


def table(mesures, nom_parametre="k", unite="%", facteur=100):
    """Affiche les mesures en table texte — le relief exigé quand le contraste
    des couleurs seul ne suffit pas à distinguer les séries."""
    params = sorted({p for p, _ in mesures})
    temps = sorted({t for _, t in mesures})
    print(f"\n  {' ' * 7}" + "".join(f"{nom_parametre}={p:<7}" for p in params))
    for t in temps:
        ligne = f"  T={t:<5.1f}"
        for p in params:
            v = mesures.get((p, t))
            ligne += f"{'—':<9}" if v is None else f"{facteur * v:<9.3f}"
        print(ligne)
    print(f"  (valeurs en {unite})")


def _cadre(ax, titre, sous_titre, x_label, y_label):
    """Habillage commun : grille discrète, axes recessifs, textes en encre."""
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.grid(True, color=GRILLE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for cote in ("top", "right"):
        ax.spines[cote].set_visible(False)
    for cote in ("left", "bottom"):
        ax.spines[cote].set_color(GRILLE)
    ax.tick_params(colors=ENCRE_2, labelsize=9, length=0)
    ax.set_xlabel(x_label, color=ENCRE_2, fontsize=10)
    ax.set_ylabel(y_label, color=ENCRE_2, fontsize=10)
    ax.set_title(titre, color=ENCRE, fontsize=13, loc="left", pad=18, fontweight="bold")
    if sous_titre:
        ax.text(0, 1.02, sous_titre, transform=ax.transAxes,
                color=ENCRE_2, fontsize=9.5, va="bottom")


def tracer_grille(mesures, chemin, nom_parametre="k",
                  titre="Taux de mots inexistants",
                  sous_titre=None, y_label="taux (%)", facteur=100):
    """Une courbe par valeur du paramètre, la température en abscisse.

    Forme choisie parce que la donnée montre une **évolution** le long d'une
    variable continue (la température), comparée entre plusieurs séries.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = sorted({p for p, _ in mesures})
    fig, ax = plt.subplots(figsize=(8, 5))
    _cadre(ax, titre, sous_titre, "température", y_label)

    for i, p in enumerate(params):
        pts = sorted((t, v) for (pp, t), v in mesures.items() if pp == p)
        if not pts:
            continue
        xs = [t for t, _ in pts]
        ys = [facteur * v for _, v in pts]
        couleur = SERIES[i % len(SERIES)]
        ax.plot(xs, ys, color=couleur, linewidth=2, zorder=3,
                marker="o", markersize=6, markeredgecolor=SURFACE,
                markeredgewidth=1.5, label=f"{nom_parametre} = {p}")

    leg = ax.legend(frameon=False, fontsize=9.5, labelcolor=ENCRE_2,
                    loc="upper left", handlelength=1.6)
    for texte in leg.get_texts():
        texte.set_color(ENCRE_2)

    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(chemin, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"  graphique écrit dans {chemin}")
    return chemin


def tracer_carte(mesures, chemin, nom_parametre="k",
                 titre="Taux de mots inexistants",
                 sous_titre=None, facteur=100, unite="%", log=False):
    """Carte de chaleur : la grille (paramètre × température) en une teinte.

    La valeur est écrite dans chaque case. C'est indispensable ici : l'étendue
    des mesures va de 0,03 % à 5,3 %, soit un rapport de 150. Sur une échelle
    linéaire, dix-huit cases sur vingt-quatre tomberaient sous 20 % d'intensité
    et deviendraient indistinguables. La couleur donne la forme, le chiffre
    donne la précision.

    `log=True` compresse l'échelle de couleur si la forme prime sur les valeurs.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize
    import numpy as np

    params = sorted({p for p, _ in mesures})
    temps = sorted({t for _, t in mesures})
    grille = np.array([[mesures.get((p, t), np.nan) * facteur for p in params]
                       for t in temps])

    cmap = LinearSegmentedColormap.from_list("bleu", RAMPE)
    cmap.set_bad(GRILLE)
    positifs = grille[np.isfinite(grille) & (grille > 0)]
    if log and positifs.size:
        norme = LogNorm(vmin=positifs.min(), vmax=np.nanmax(grille))
    else:
        norme = Normalize(vmin=0, vmax=np.nanmax(grille))

    fig, ax = plt.subplots(figsize=(1.15 * len(params) + 3, 0.85 * len(temps) + 2.6))
    fig.set_facecolor(SURFACE)
    im = ax.imshow(grille, cmap=cmap, norm=norme, aspect="auto")

    ax.set_xticks(range(len(params)), [f"{nom_parametre}={p}" for p in params])
    ax.set_yticks(range(len(temps)), [f"T={t:.1f}" for t in temps])
    ax.tick_params(colors=ENCRE_2, labelsize=9.5, length=0)
    for cote in ax.spines.values():
        cote.set_visible(False)
    # Séparateur de 2 px entre les cases, à la couleur de la surface.
    ax.set_xticks(np.arange(-.5, len(params), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(temps), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    seuil = np.nanmax(grille) * 0.55          # au-delà, le fond est trop foncé
    for i in range(len(temps)):
        for j in range(len(params)):
            v = grille[i, j]
            if not np.isfinite(v):
                continue
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=9,
                    color=SURFACE if v > seuil else ENCRE)

    barre = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    barre.set_label(f"taux ({unite})", color=ENCRE_2, fontsize=9.5)
    barre.ax.tick_params(colors=ENCRE_2, labelsize=8.5, length=0)
    barre.outline.set_visible(False)

    ax.set_title(titre, color=ENCRE, fontsize=13, loc="left", pad=20, fontweight="bold")
    if sous_titre:
        ax.text(0, 1.015, sous_titre, transform=ax.transAxes,
                color=ENCRE_2, fontsize=9.5, va="bottom")

    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(chemin, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"  graphique écrit dans {chemin}")
    return chemin


def tracer_compromis(mesures_x, mesures_y, chemin, nom_parametre="k",
                     titre="Compromis répétition / mots inexistants",
                     sous_titre=None,
                     x_label="taux de répétition (%)",
                     y_label="taux de mots inexistants (%)", facteur=100):
    """Un point par réglage, une métrique en abscisse, l'autre en ordonnée.

    C'est la forme qui montre un **compromis** : les réglages qu'aucun autre ne
    bat sur les deux critères à la fois se lisent en bas à gauche du nuage.
    Les deux dictionnaires doivent partager leurs clés.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    communes = sorted(set(mesures_x) & set(mesures_y), key=lambda c: (c[0], c[1]))
    params = sorted({p for p, _ in communes})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    _cadre(ax, titre, sous_titre, x_label, y_label)

    for i, p in enumerate(params):
        cles = [c for c in communes if c[0] == p]
        couleur = SERIES[i % len(SERIES)]
        ax.plot([facteur * mesures_x[c] for c in cles],
                [facteur * mesures_y[c] for c in cles],
                color=couleur, linewidth=1.2, alpha=0.55, zorder=2)
        ax.scatter([facteur * mesures_x[c] for c in cles],
                   [facteur * mesures_y[c] for c in cles],
                   s=70, color=couleur, edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label=f"{nom_parametre} = {p}")
        for c in cles:  # la température, sur le point, en encre et non en couleur
            ax.annotate(f"{c[1]:.1f}",
                        (facteur * mesures_x[c], facteur * mesures_y[c]),
                        textcoords="offset points", xytext=(7, 4),
                        fontsize=7.5, color=ENCRE_2)

    leg = ax.legend(frameon=False, fontsize=9.5, loc="best", handlelength=1.2)
    for texte in leg.get_texts():
        texte.set_color(ENCRE_2)

    chemin = Path(chemin)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(chemin, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    print(f"  graphique écrit dans {chemin}")
    return chemin
