#!/usr/bin/env python3
"""
Allège un checkpoint : retire l'état de l'optimiseur.

Un checkpoint pèse trois fois le nombre de paramètres, parce qu'il embarque les
deux moments d'Adam (`m` et `v`) en plus des poids. Ils servent à REPRENDRE un
entraînement ; `generation.py` ne les lit jamais. Les retirer divise le fichier
par trois — décisif quand il faut le rapatrier par une liaison lente.

ZONE VERTE — n'écrit aucune logique de modèle. Lit un fichier, en écrit un autre,
et vérifie que toutes les clés dont `genere()` a besoin sont encore là.

    python3 src/tooling/alleger_checkpoint.py chemin.pt
    python3 src/tooling/alleger_checkpoint.py dossier/ --suffixe -leger
"""
import argparse
import sys
from pathlib import Path

import torch

# Ce que retire l'allègement : état de l'optimiseur uniquement.
OPTIMISEUR = ('m', 'v', 't')

# Ce que `generation.py` lit dans le checkpoint. Relevé sur le fichier, pas
# supposé : si une clé manque après allègement, le script refuse d'écrire.
REQUISES = (
    'c', 'pos_emb', 'W_q', 'W_k', 'W_v', 'W_o', 'W_1', 'b_1', 'W_2', 'b_2',
    'ln1_g', 'ln1_b', 'ln2_g', 'ln2_b', 'lnn_g', 'lnn_b', 'W_out', 'b_out',
    'dim', 'max_len', 'num_heads', 'num_blocs', 'i', 'loss', 'pas', 'seed',
)


def alleger(source: Path, sortie: Path):
    avant = source.stat().st_size
    cp = torch.load(source, map_location='cpu', weights_only=False)

    retires = [k for k in OPTIMISEUR if k in cp]
    for k in retires:
        del cp[k]

    manquantes = [k for k in REQUISES if k not in cp]
    if manquantes:
        sys.exit(f"{source.name} : clés manquantes après allègement — "
                 f"{', '.join(manquantes)}. Rien n'a été écrit.")

    torch.save(cp, sortie)
    apres = sortie.stat().st_size
    print(f"{source.name:<44} {avant/2**20:7.1f} Mo -> {apres/2**20:6.1f} Mo "
          f"(x{avant/apres:.2f})   retiré : {', '.join(retires) or 'rien'}")
    return avant, apres


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cible', help='un .pt, ou un dossier contenant des .pt')
    ap.add_argument('--suffixe', default='-leger',
                    help="suffixe du fichier produit (défaut : -leger)")
    ap.add_argument('--remplacer', action='store_true',
                    help="écraser l'original au lieu d'écrire à côté. "
                         "L'entraînement ne pourra plus être repris depuis ce "
                         "checkpoint — la génération, si.")
    a = ap.parse_args()

    cible = Path(a.cible)
    fichiers = sorted(cible.glob('*.pt')) if cible.is_dir() else [cible]
    if not fichiers:
        sys.exit(f'aucun .pt dans {cible}')

    total_avant = total_apres = 0
    for f in fichiers:
        if f.stem.endswith(a.suffixe):
            continue
        sortie = f if a.remplacer else f.with_name(f.stem + a.suffixe + f.suffix)
        av, ap_ = alleger(f, sortie)
        total_avant += av
        total_apres += ap_

    if len(fichiers) > 1:
        print(f"\ntotal : {total_avant/2**30:.2f} Go -> {total_apres/2**30:.2f} Go")


if __name__ == '__main__':
    main()
