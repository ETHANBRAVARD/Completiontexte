#!/usr/bin/env python3
"""
Vérifie un dossier de checkpoints : lisibles, cohérents, complets.

Motivation. Un transfert interrompu laisse un fichier tronqué **sous son nom
définitif** (rsync --partial), qu'un `ls` ne distingue pas d'un fichier complet.
Et un tri lexicographique fait passer `sauvegarde_935` après `sauvegarde_1496` —
c'est ce qui avait fait supprimer les bons checkpoints le 08/09, sans rien
afficher. Les deux fautes se ressemblent : on croit travailler sur un modèle,
on travaille sur autre chose.

Ce script charge chaque fichier et dit ce qu'il contient réellement.

ZONE VERTE — n'écrit aucune logique de modèle et ne modifie rien. Il lit.

    python3 src/tooling/verifier_checkpoints.py runs/mon-run/
    python3 src/tooling/verifier_checkpoints.py runs/mon-run/ --log chemin/log.txt
"""
import argparse
import re
import sys
from pathlib import Path

import torch

# Ce que `generation.py` lit dans un checkpoint. Un fichier auquel il manque
# l'une de ces clés se chargera sans erreur mais plantera à la génération.
REQUISES = (
    'c', 'pos_emb', 'W_q', 'W_k', 'W_v', 'W_o', 'W_1', 'b_1', 'W_2', 'b_2',
    'ln1_g', 'ln1_b', 'ln2_g', 'ln2_b', 'lnn_g', 'lnn_b', 'W_out', 'b_out',
    'dim', 'max_len', 'num_heads', 'num_blocs', 'i', 'loss',
)


def pas_du_nom(f: Path):
    """Le pas lu dans le nom de fichier — tri NUMÉRIQUE, jamais lexicographique."""
    m = re.search(r'sauvegarde_(\d+)', f.stem)
    return int(m.group(1)) if m else -1


def pertes_du_log(chemin: Path):
    """pas -> perte de validation, pour recouper ce qu'annonce le checkpoint."""
    if not chemin or not chemin.exists():
        return {}
    pertes, attend = {}, False
    for ligne in chemin.read_text(errors='replace').splitlines():
        if 'jeu de validation' in ligne:
            attend = True
            continue
        if attend:
            m = re.match(r'i=(\d+), loss=([\d.]+)', ligne)
            if m:
                pertes[int(m.group(1))] = float(m.group(2))
            attend = False
    return pertes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dossier')
    ap.add_argument('--log', default=None,
                    help="log.txt du run, pour recouper les pertes (défaut : "
                         "cherché à côté des checkpoints)")
    a = ap.parse_args()

    racine = Path(a.dossier)
    fichiers = sorted(racine.rglob('*.pt'), key=pas_du_nom)
    if not fichiers:
        sys.exit(f'aucun .pt sous {racine}')

    log = Path(a.log) if a.log else next(racine.rglob('log.txt'), None)
    pertes = pertes_du_log(log)
    if pertes:
        print(f"log de référence : {log}  ({len(pertes)} mesures)\n")

    print(f"{'fichier':<46} {'Mo':>7} {'pas':>7} {'perte':>8}  état")
    bons = casses = suspects = 0
    for f in fichiers:
        mo = f.stat().st_size / 2**20
        nom = f.name[-45:]
        try:
            cp = torch.load(f, map_location='cpu', weights_only=False)
        except Exception as e:
            print(f"{nom:<46} {mo:7.1f} {'—':>7} {'—':>8}  ILLISIBLE ({type(e).__name__})")
            casses += 1
            continue

        alertes = []
        manquantes = [k for k in REQUISES if k not in cp]
        if manquantes:
            alertes.append(f"clés manquantes : {', '.join(manquantes[:4])}")

        pas_fichier, pas_nom = cp.get('i'), pas_du_nom(f)
        if pas_nom >= 0 and pas_fichier != pas_nom:
            alertes.append(f"le nom dit {pas_nom}, le contenu dit {pas_fichier}")

        attendue = pertes.get(pas_fichier)
        if attendue is not None and abs(attendue - cp['loss']) > 1e-3:
            alertes.append(f"perte {cp['loss']:.4f} ≠ log {attendue:.4f}")

        if 'm' not in cp:
            alertes.append("allégé : génération oui, reprise d'entraînement non")

        etat = 'OK' if not alertes else ' | '.join(alertes)
        print(f"{nom:<46} {mo:7.1f} {pas_fichier:>7} {cp['loss']:8.4f}  {etat}")
        if alertes:
            suspects += 1
        else:
            bons += 1

    print(f"\n{bons} valides, {suspects} à regarder, {casses} illisibles "
          f"sur {len(fichiers)} fichiers")
    if pertes:
        absents = sorted(set(pertes) - {pas_du_nom(f) for f in fichiers})
        if absents:
            print(f"pas mesurés dans le log mais sans checkpoint : "
                  f"{', '.join(map(str, absents[:12]))}"
                  f"{' …' if len(absents) > 12 else ''}")
    if casses:
        sys.exit(1)


if __name__ == '__main__':
    main()
