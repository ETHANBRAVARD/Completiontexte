"""Banc de bruit : perte de validation en fonction de l'intensité du bruit.

Zone verte (tooling) : orchestration, mesure, tracé. N'implémente aucun bruit et
aucune passe avant — il appelle `bruit_mult`, `bruit_add` et `loss_validation`
de `src/model/Bruit.py`, telles qu'Ethan les a écrites.

Le point de méthode : **les lots de validation sont les mêmes pour tous les
alpha.** `tireur_de_lot` tire ses positions avec le module `random` de Python,
les fonctions de bruit tirent avec le générateur de torch. On fixe donc les deux
séparément : la graine `random` ne dépend que du numéro de répétition, la graine
torch dépend aussi de l'alpha. Pour une répétition donnée, la mesure sans bruit
et toutes les mesures bruitées portent sur exactement les mêmes textes — l'écart
mesuré est dû au bruit, pas au hasard de l'échantillonnage.

On rapporte donc deux choses par point :
  - la perte moyenne et sa dispersion entre répétitions ;
  - l'écart **apparié** à la mesure sans bruit de la même répétition, beaucoup
    moins bruité puisque la variance des lots s'y annule.

Usage :
    python3 scripts/banc_bruit.py
    python3 scripts/banc_bruit.py --repetitions 8 --alphas 0.01,0.05,0.1
    python3 scripts/banc_bruit.py --checkpoint runs/.../sauvegarde_45000.pt
"""

import argparse
import contextlib
import io
import json
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "model"))
sys.path.insert(0, str(ROOT / "src" / "tooling"))

CHECKPOINT = ROOT / "runs/echelle-20260908-2339/A2-actuel/20260909-0239-sauvegarde_17000.pt"
ALPHAS = [0.003, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]


def mesurer(Bruit, modele, lot, tok_val, repetition):
    """Une perte de validation, lots fixés par la répétition, affichage étouffé."""
    random.seed(repetition)
    with contextlib.redirect_stdout(io.StringIO()):   # loss_validation imprime
        return Bruit.loss_validation(modele, lot, tok_val, "cuda")


def tracer(resultats, base, chemin, titre_ckpt, vocabulaire=2080):
    """Écart apparié à la mesure sans bruit, axe vertical logarithmique.

    Tracer la perte brute écrase tout : l'additif monte à ~80 et la zone où le
    modèle commence à se dégrader — celle qui compte — tient dans un pixel.
    L'écart apparié est aussi la grandeur la plus précise que mesure le banc.
    """
    import math
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import tracer as T

    plancher = 1e-4   # un écart nul ne se place pas sur un axe logarithmique
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    T._cadre(ax, "Dégradation sous bruit",
             f"{titre_ckpt} — écart de perte à la mesure sans bruit "
             f"({base['moyenne']:.4f}), mêmes lots, ± écart-type",
             "alpha (échelle logarithmique)", "écart de perte (échelle logarithmique)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    etiquettes = {"mult": "multiplicatif  (σ = alpha × |w|)",
                  "add": "additif  (σ = alpha × max|w| de la matrice)"}
    for i, loi in enumerate(("mult", "add")):
        pts = resultats[loi]
        xs = [q["alpha"] for q in pts]
        ys = [max(q["ecart_apparie"], plancher) for q in pts]
        sg = [q["ecart_apparie_sigma"] for q in pts]
        c = T.SERIES[i]
        ax.fill_between(xs, [max(y - e, plancher) for y, e in zip(ys, sg)],
                        [y + e for y, e in zip(ys, sg)], color=c, alpha=0.15,
                        linewidth=0, zorder=2)
        ax.plot(xs, ys, color=c, linewidth=2, marker="o", markersize=5.5,
                markeredgecolor=T.SURFACE, markeredgewidth=1.3, zorder=3,
                label=etiquettes[loi])

    # Repère : un modèle qui répartit uniformément sa probabilité sur le
    # vocabulaire obtient ln(V). Au-dessus, le bruit rend le modèle pire que le
    # hasard : il se trompe avec assurance.
    uniforme = math.log(vocabulaire) - base["moyenne"]
    ax.axhline(uniforme, color=T.ENCRE_2, linewidth=1.1, linestyle="--", zorder=2)
    ax.text(ax.get_xlim()[0], uniforme,
            f"  pire qu'un tirage uniforme sur {vocabulaire} tokens",
            color=T.ENCRE_2, fontsize=9, va="bottom")
    ax.axhline(0.01, color=T.GRILLE, linewidth=1.1, linestyle=":", zorder=2)
    ax.text(ax.get_xlim()[0], 0.01, "  +0,01", color=T.ENCRE_2, fontsize=8.5,
            va="bottom")

    leg = ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    for t in leg.get_texts():
        t.set_color(T.ENCRE_2)
    fig.tight_layout()
    fig.savefig(chemin, dpi=160, facecolor=T.SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--val", default="data/encode_tok_val.npy",
                   help="données de validation cohérentes avec le checkpoint")
    p.add_argument("--alphas", default=",".join(str(a) for a in ALPHAS))
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--lot", type=int, default=32)
    p.add_argument("--retracer", default="",
                   help="dossier d'un banc existant : refaire le graphique depuis "
                        "son resultats.json, sans rien mesurer")
    args = p.parse_args()

    if args.retracer:
        d = Path(args.retracer)
        d = d if d.is_absolute() else ROOT / d
        c = json.loads((d / "resultats.json").read_text(encoding="utf-8"))
        tracer(c["resultats"], c["sans_bruit"], d / "bruit.png",
               f"{Path(c['checkpoint']).parent.name}, pas {c['pas']}",
               c.get("vocabulaire", 2080))
        print(f"graphique refait : {(d / 'bruit.png').relative_to(ROOT)}")
        return 0

    import Bruit
    ckpt = Path(args.checkpoint)
    if not ckpt.is_absolute():
        ckpt = ROOT / ckpt
    modele = torch.load(ckpt, map_location="cuda", weights_only=False)
    tok_val = np.load(ROOT / args.val, mmap_mode="r")
    alphas = [float(a) for a in args.alphas.split(",")]
    reps = list(range(args.repetitions))

    dossier = ROOT / "runs" / f"bruit-{datetime.now():%Y%m%d-%H%M}"
    dossier.mkdir(parents=True, exist_ok=True)
    n_par = sum(t.numel() for k in ("W_q", "W_k", "W_v", "W_o", "W_1", "W_2")
                for t in modele[k]) + modele["W_out"].numel()

    print(f"banc      : {dossier.relative_to(ROOT)}")
    print(f"checkpoint: {ckpt.relative_to(ROOT)}  (pas {modele.get('i')}, "
          f"perte enregistrée {modele.get('loss', float('nan')):.4f})")
    print(f"bruité    : {n_par/1e6:.2f} M poids dans les produits matriciels")
    print(f"protocole : {len(alphas)} alphas × 2 lois × {len(reps)} répétitions, "
          f"lots identiques à répétition égale\n")

    t0 = time.perf_counter()
    base_par_rep = [mesurer(Bruit, modele, args.lot, tok_val, r) for r in reps]
    base = {"moyenne": statistics.mean(base_par_rep),
            "ecart_type": statistics.stdev(base_par_rep) if len(reps) > 1 else 0.0,
            "par_repetition": base_par_rep}
    print(f"  sans bruit : {base['moyenne']:.4f} ± {base['ecart_type']:.4f}   "
          f"({(time.perf_counter()-t0)/len(reps):.2f} s par mesure)\n")

    resultats = {"mult": [], "add": []}
    fonctions = {"mult": Bruit.bruit_mult, "add": Bruit.bruit_add}
    print(f"  {'loi':5}{'alpha':>8}{'perte':>10}{'± σ':>9}{'écart apparié':>16}{'± σ':>9}")
    for loi in ("mult", "add"):
        for ia, alpha in enumerate(alphas):
            pertes, ecarts = [], []
            for r in reps:
                torch.manual_seed(100_000 * r + 1_000 * ia + (0 if loi == "mult" else 1))
                bruite = fonctions[loi](modele, alpha)
                perte = mesurer(Bruit, bruite, args.lot, tok_val, r)
                del bruite
                pertes.append(perte)
                ecarts.append(perte - base_par_rep[r])
            point = {"alpha": alpha,
                     "moyenne": statistics.mean(pertes),
                     "ecart_type": statistics.stdev(pertes) if len(reps) > 1 else 0.0,
                     "ecart_apparie": statistics.mean(ecarts),
                     "ecart_apparie_sigma": statistics.stdev(ecarts) if len(reps) > 1 else 0.0,
                     "par_repetition": pertes}
            resultats[loi].append(point)
            print(f"  {loi:5}{alpha:8g}{point['moyenne']:10.4f}{point['ecart_type']:9.4f}"
                  f"{point['ecart_apparie']:+16.4f}{point['ecart_apparie_sigma']:9.4f}")
        print()

    contenu = {"checkpoint": str(ckpt.relative_to(ROOT)), "pas": modele.get("i"),
               "vocabulaire": modele.get("alph", 2080),
               "validation": args.val, "lot": args.lot, "lots_par_mesure": 10,
               "repetitions": len(reps),
               "graines": "random.seed(r) pour les lots ; "
                          "torch.manual_seed(100000*r + 1000*i_alpha + loi) pour le bruit",
               "sans_bruit": base, "resultats": resultats,
               "duree_s": time.perf_counter() - t0}
    (dossier / "resultats.json").write_text(json.dumps(contenu, indent=1,
                                                       ensure_ascii=False), encoding="utf-8")
    tracer(resultats, base, dossier / "bruit.png",
           f"{ckpt.parent.name}, pas {modele.get('i')}", modele.get("alph", 2080))
    print(f"durée : {(time.perf_counter()-t0)/60:.1f} min")
    print(f"écrit : {(dossier / 'resultats.json').relative_to(ROOT)}, "
          f"{(dossier / 'bruit.png').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
