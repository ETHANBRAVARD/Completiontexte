#!/usr/bin/env python3
"""
Échelle de texte : la loss prédit-elle la qualité du texte ?

Parcourt tous les checkpoints d'un run, génère les mêmes amorces avec les mêmes
graines à chaque palier, et mesure. Seul le modèle change d'un palier à l'autre :
la comparaison est appariée, comme le banc de bruit.

ZONE VERTE — n'écrit aucune logique de modèle. Appelle generation.genere().
"""
import argparse, json, re, sys, time
from pathlib import Path

import torch

sys.path.insert(0, 'src/model')
import generation

AMORCES = [
    'Once upon a time',
    'One day, a little girl named Lily',
    'Tim and his dog were playing in the garden when',
    'The old man opened the door and',
    '"I am scared," she said,',
    'In the big forest, there was a tiny',
    'It was a very hot day and',
    'Suddenly,',
    'Lucy and her mom went to the shop to buy',
    'The cat wanted the fish, but',
    'He wanted to fly, so he',
    'One rainy morning, the little boy found a',
    '"That\'s mine!" shouted',
    'Before she could answer, the bird',
    'The moral of the story is',
]


def pertes_validation(dossier):
    """pas -> perte de validation, lue dans log.txt."""
    log = Path(dossier) / 'log.txt'
    if not log.exists():
        return {}
    pertes, attend = {}, False
    for ligne in log.read_text(errors='replace').splitlines():
        if 'jeu de validation' in ligne:
            attend = True
            continue
        if attend:
            m = re.match(r'i=(\d+), loss=([\d.]+)', ligne)
            if m:
                pertes[int(m.group(1))] = float(m.group(2))
            attend = False
    return pertes


def checkpoints(dossier):
    fichiers = list(Path(dossier).glob('*-sauvegarde_*.pt'))
    return sorted(fichiers, key=lambda f: int(f.stem.split('_')[-1]))


def mots(texte):
    return re.findall(r"[a-z']+", texte.lower())


def quadrigrammes(texte):
    m = mots(texte)
    return [tuple(m[i:i + 4]) for i in range(len(m) - 3)]


def mesure(textes, vocab):
    """Les trois métriques du banc d'échantillonnage, appliquées à un palier."""
    tous_mots, absents = 0, 0
    rep_num, rep_den = 0, 0
    vus, partages, total_q = set(), 0, 0
    for t in textes:
        m = mots(t)
        tous_mots += len(m)
        absents += sum(1 for w in m if w not in vocab)
        q = quadrigrammes(t)
        interne = set()
        for g in q:
            rep_den += 1
            if g in interne:
                rep_num += 1
            interne.add(g)
        for g in set(q):
            total_q += 1
            if g in vus:
                partages += 1
        vus |= set(q)
    pc = lambda n, d: round(100 * n / d, 3) if d else None
    return {
        'mots': tous_mots,
        'inexistants_pc': pc(absents, tous_mots),
        'repetition_pc': pc(rep_num, rep_den),
        'redite_pc': pc(partages, total_q),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help="dossier d'un run (contient log.txt et les .pt)")
    ap.add_argument('--car', type=int, default=400, help='longueur demandée par texte')
    ap.add_argument('--amorces', type=int, default=len(AMORCES))
    ap.add_argument('--mode', default='topk')
    ap.add_argument('--temp', type=float, default=1.2)
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--p', type=float, default=0.9)
    ap.add_argument('--graine', type=int, default=20260920)
    ap.add_argument('--sortie', default=None)
    a = ap.parse_args()

    dossier = Path(a.run)
    cps = checkpoints(dossier)
    if not cps:
        sys.exit(f'aucun checkpoint dans {dossier}')
    pertes = pertes_validation(dossier)
    amorces = AMORCES[:a.amorces]
    vocab = set(json.loads(Path('data/vocabulaire.json').read_text()))

    sortie = Path(a.sortie or f'runs/generation/echelle_texte_{dossier.name}.json')
    sortie.parent.mkdir(parents=True, exist_ok=True)

    print(f'{len(cps)} paliers x {len(amorces)} amorces, {a.car} caractères, '
          f'{a.mode} temp={a.temp} k={a.k}')
    print(f'graines fixées : la même amorce garde la même graine à tous les paliers\n')

    resultats = []
    t0 = time.time()
    for n, cp in enumerate(cps, 1):
        pas = int(cp.stem.split('_')[-1])
        sauvegarde = torch.load(cp, weights_only=False)
        textes = []
        for j, am in enumerate(amorces):
            textes.append(generation.genere(
                sauvegarde, a.car, am, mode=a.mode, temp=a.temp,
                k=a.k, p=a.p, seed=a.graine + j))
        m = mesure(textes, vocab)
        m.update(pas=pas, perte_val=pertes.get(pas),
                 checkpoint=str(cp), textes=textes)
        resultats.append(m)
        print(f'[{n:>2}/{len(cps)}] pas {pas:>6}  val {str(m["perte_val"]):>6}  '
              f'inexistants {m["inexistants_pc"]}%  répétition {m["repetition_pc"]}%  '
              f'redite {m["redite_pc"]}%  ({time.time() - t0:.0f} s)', flush=True)
        sortie.write_text(json.dumps({
            'run': str(dossier), 'reglages': vars(a),
            'amorces': amorces, 'paliers': resultats}, ensure_ascii=False, indent=1))

    print(f'\nécrit : {sortie}   ({time.time() - t0:.0f} s)')


if __name__ == '__main__':
    main()
