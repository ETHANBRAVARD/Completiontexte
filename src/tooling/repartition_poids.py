"""Répartition des poids d'un checkpoint, famille par famille.

Zone verte (tooling) : lecture et tracé. Aucune logique de modèle, aucune
quantification — le script montre à quoi ressemblent les poids, la grille est
à toi.

    python3 src/tooling/repartition_poids.py
    python3 src/tooling/repartition_poids.py --checkpoint runs/.../sauvegarde_17000.pt

Deux vues par famille :
  - la répartition brute des valeurs, en échelle log pour voir les queues ;
  - la répartition de |w| / max|w|, **matrice par matrice** — c'est l'échelle de
    la plage `±max|w|` du bruit additif. Une matrice dont presque tous les poids
    sont serrés près de 0 n'utilise qu'une petite partie de sa plage.

Chiffres imprimés par famille :
  max/σ        combien d'écarts-types sépare le plus gros poids de 0 ;
  kurtosis     excès d'aplatissement : 0 pour une gaussienne, grand = queues lourdes ;
  p50, p99     médiane et 99e centile de |w| / max|w| ;
  max ligne    médiane, sur les lignes, du max de la ligne rapporté au max de la
               matrice (< 1 : le max global est porté par quelques lignes).
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

RACINE = Path(__file__).resolve().parents[2]
CHECKPOINT = RACINE / "runs/gpu01-20260922-2051/A4-grand/20260922-2051-sauvegarde_120000-leger.pt"
FAMILLES = ["W_q", "W_k", "W_v", "W_o", "W_1", "W_2", "W_out"]


def matrices(ck, famille):
    """Les matrices d'une famille, en tableaux NumPy (W_out est seule)."""
    v = ck[famille]
    return [t.float().numpy() for t in (v if isinstance(v, list) else [v])]


def stats(mats):
    tout = np.concatenate([m.ravel() for m in mats])
    sigma = tout.std()
    kurt = ((tout - tout.mean()) ** 4).mean() / sigma**4 - 3
    norm = np.concatenate([np.abs(m).ravel() / np.abs(m).max() for m in mats])
    max_ligne = np.concatenate([np.abs(m).max(axis=1) / np.abs(m).max() for m in mats])
    return {
        "n": tout.size, "sigma": sigma, "max": np.abs(tout).max(),
        "max_sur_sigma": np.mean([np.abs(m).max() / m.std() for m in mats]),
        "kurtosis": kurt,
        "p50": np.percentile(norm, 50), "p99": np.percentile(norm, 99),
        "max_ligne": np.median(max_ligne),
    }, tout, norm


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--checkpoint", default=str(CHECKPOINT))
    p.add_argument("--sortie", default="")
    a = p.parse_args()

    ckpt = Path(a.checkpoint)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    sortie = Path(a.sortie) if a.sortie else RACINE / "runs" / f"poids-{ckpt.stem}.png"

    print(f"checkpoint : {ckpt.relative_to(RACINE) if ckpt.is_relative_to(RACINE) else ckpt}"
          f"  (pas {ck.get('i')}, loss {ck.get('loss', float('nan')):.4f})\n")
    print(f"  {'famille':8}{'poids':>11}{'σ':>9}{'max|w|':>9}{'max/σ':>8}"
          f"{'kurtosis':>10}{'p50':>7}{'p99':>7}{'max ligne':>11}")

    fig, axes = plt.subplots(2, len(FAMILLES), figsize=(3.2 * len(FAMILLES), 6.5),
                             constrained_layout=True)
    for j, fam in enumerate(FAMILLES):
        s, tout, norm = stats(matrices(ck, fam))
        print(f"  {fam:8}{s['n']:>11,}{s['sigma']:9.4f}{s['max']:9.3f}"
              f"{s['max_sur_sigma']:8.1f}{s['kurtosis']:10.1f}"
              f"{s['p50']:7.3f}{s['p99']:7.3f}{s['max_ligne']:11.3f}".replace(",", " "))

        ax = axes[0, j]
        effectifs, bords, _ = ax.hist(tout, bins=200, color="#3b6ea5", log=True)
        x = np.linspace(bords[0], bords[-1], 400)
        # effectif attendu par case si les poids étaient gaussiens, même σ
        gauss = tout.size * (bords[1] - bords[0]) \
            * np.exp(-0.5 * ((x - tout.mean()) / s["sigma"]) ** 2) / (s["sigma"] * np.sqrt(2 * np.pi))
        ax.plot(x, gauss, color="#c0504d", lw=1, label="gaussienne de même σ")
        ax.set_ylim(0.5, effectifs.max() * 3)
        ax.set_title(f"{fam}   max/σ = {s['max_sur_sigma']:.0f}")
        ax.set_xlabel("w")

        ax = axes[1, j]
        ax.hist(norm, bins=100, range=(0, 1), color="#5a9e6f", log=True)
        ax.axvline(s["p99"], color="#444", ls="--", lw=1)
        ax.set_ylim(bottom=0.5)
        ax.set_xlabel("|w| / max|w| de sa matrice")
    axes[0, 0].set_ylabel("nombre de poids (log)")
    axes[1, 0].set_ylabel("nombre de poids (log)")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Répartition des poids — {ckpt.name}   (tirets : 99 % des poids sont à gauche)")
    fig.savefig(sortie, dpi=110)
    print(f"\ngraphique : {sortie.relative_to(RACINE) if sortie.is_relative_to(RACINE) else sortie}")


if __name__ == "__main__":
    sys.exit(main())
