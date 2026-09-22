#!/usr/bin/env python3
"""
Programme de nuit portable — zone verte (orchestration).

Même rôle que nuit6.sh / run_long.sh, mais en Python : le PC fixe est sous
Windows, où bash n'existe pas. Ce fichier tourne tel quel sur Linux, sur
Windows natif et sous WSL.

    python3 scripts/programme.py --lister
    python3 scripts/programme.py nuit6
    python3 scripts/programme.py etalon          # 60 pas par taille, ~5 min
    python3 scripts/programme.py sonde-long
    python3 scripts/programme.py run-long --go
"""
import argparse, platform, subprocess, sys, time
from datetime import datetime
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
BANC = [sys.executable, str(RACINE / 'scripts' / 'echelle.py')]

# valeurs gelées du 19-20/09 : cos, lr 2e-3, écrêtage 1.0, échauffement 1000
SOCLE = ['--configs', 'A2-actuel', '--pas', '15000', '--val-tous', '1000',
         '--apprentissage', '0.002', '--schedule', 'cos', '--amorce', '1000',
         '--clip', '1.0']

PROGRAMMES = {
    'etalon': (
        "60 pas sur chaque taille : mesure le temps par pas de CETTE machine. "
        "À lancer en premier sur une machine inconnue — tout le dimensionnement "
        "en dépend.",
        [['--calibrer', '--partie', '1'], ['--calibrer', '--partie', '2']],
    ),
    'nuit6': (
        "Où commence la décroissance ? Six runs de 15000 pas, seul --fin varie. "
        "Les deux derniers sont des témoins à graine différente : ils donnent le "
        "plancher de bruit, sans lequel un écart de 0,02 ne veut rien dire.",
        [SOCLE + ['--fin', f] for f in ('14000', '7000', '3750', '10500')]
        + [SOCLE + ['--fin', f, '--graine', '4242'] for f in ('1000', '14000')],
    ),
    'sonde-long': (
        "1500 pas à dim=512 : le pas d'apprentissage tient-il à cette largeur ? "
        "14 min pour ne pas perdre 6 h.",
        [['--configs', 'A3-moyen', '--pas', '1500', '--val-tous', '250',
          '--apprentissage', '0.0015', '--schedule', 'cos', '--amorce', '1000',
          '--fin', '500', '--clip', '1.0']],
    ),
    'sondes-seuil': (
        "Encadrer le seuil d'instabilité à dim=512 : 4,5e-3 puis 6e-3, 1500 pas. "
        "On cherche où ça casse pour se placer en dessous avec une marge, comme le "
        "15/09 pour la largeur. ~28 min.",
        [['--configs', 'A3-moyen', '--pas', '1500', '--val-tous', '250',
          '--apprentissage', lr, '--schedule', 'cos', '--amorce', '1000',
          '--fin', '500', '--clip', '1.0'] for lr in ('0.0045', '0.006')],
    ),
    'sonde-longue': (
        "4000 pas à 3e-3, échauffement 1000, SANS décroissance : le modèle passe "
        "3000 pas au pas de base, le régime où l'instabilité se déclare. Les sondes "
        "de 1500 pas n'y passent presque pas de temps. ~37 min.",
        [['--configs', 'A3-moyen', '--pas', '4000', '--val-tous', '500',
          '--apprentissage', '0.003', '--schedule', 'cos', '--amorce', '1000',
          '--fin', '1', '--clip', '1.0']],
    ),
    'sondes-longues': (
        "Deux sondes de 5500 pas à dim=512, échauffement 1000, SANS décroissance : "
        "4500 pas passés au pas de base, le régime où l'instabilité se déclare. "
        "6e-3 d'abord (une rupture y encadre le seuil), puis 4,5e-3. ~1 h 45.",
        [['--configs', 'A3-moyen', '--pas', '5500', '--val-tous', '500',
          '--apprentissage', lr, '--schedule', 'cos', '--amorce', '1000',
          '--fin', '1', '--clip', '1.0'] for lr in ('0.006', '0.0045')],
    ),
    'run-long': (
        "Le livrable de l'étape 5 : 40453 pas = 1 époque exacte du corpus "
        "complet, 24,4 M paramètres, 20,4 tokens par paramètre. ~6 h.",
        [['--configs', 'A3-moyen', '--pas', '40453', '--val-tous', '2000',
          '--apprentissage', '0.0045', '--schedule', 'cos', '--amorce', '1000',
          '--fin', '10000', '--clip', '1.0', '--garder-tout']],
    ),
}


def machine():
    infos = [f"{platform.node()} · {platform.system()} {platform.release()}",
             f"python {platform.python_version()}"]
    try:
        import torch
        infos.append(f"torch {torch.__version__}")
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            infos.append(f"{p.name} · {p.total_memory / 2**30:.1f} Gio VRAM")
        else:
            infos.append("AUCUN GPU VISIBLE — les runs prendraient des semaines")
    except ImportError:
        infos.append("torch ABSENT")
    return infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('programme', nargs='?')
    ap.add_argument('--lister', action='store_true')
    ap.add_argument('--go', action='store_true',
                    help="exiger explicitement les programmes de plus d'une heure")
    ap.add_argument('--lr', default=None,
                    help="remplace le pas d'apprentissage du programme "
                         "(ex. relancer sonde-longue au pas retenu par sondes-seuil)")
    ap.add_argument('--pause', type=int, default=65,
                    help='secondes entre deux runs, pour libérer la VRAM')
    a = ap.parse_args()

    if a.lister or not a.programme:
        for nom, (desc, etapes) in PROGRAMMES.items():
            print(f"\n{nom}  ({len(etapes)} run{'s' if len(etapes) > 1 else ''})")
            for ligne in desc.split('. '):
                if ligne.strip():
                    print(f"    {ligne.strip().rstrip('.')}.")
        return

    if a.programme not in PROGRAMMES:
        sys.exit(f"programme inconnu : {a.programme}\n"
                 f"connus : {', '.join(PROGRAMMES)}")

    desc, etapes = PROGRAMMES[a.programme]
    # Tout ce qui occupe le GPU exige --go : sans ça, un simple essai de la CLI
    # lance un entraînement et entre en collision avec le run en cours.
    longs = a.programme != 'etalon'
    if longs and not a.go:
        sys.exit(f"{a.programme} : {desc}\n\n"
                 f"Programme long. Relancer avec --go pour le lancer vraiment.")

    if a.lr is not None:
        etapes = [list(e) for e in etapes]
        for e in etapes:
            if '--apprentissage' in e:
                e[e.index('--apprentissage') + 1] = a.lr

    horodatage = datetime.now().strftime('%Y%m%d-%H%M')
    journal = RACINE / 'runs' / f'{a.programme}-{horodatage}.log'
    journal.parent.mkdir(parents=True, exist_ok=True)

    def log(*x):
        ligne = f"[{datetime.now():%H:%M:%S}] " + ' '.join(str(i) for i in x)
        print(ligne, flush=True)
        with journal.open('a', encoding='utf-8') as f:
            f.write(ligne + '\n')

    log(f"=== {a.programme} ===")
    for i in machine():
        log('   ', i)
    log(f"    {len(etapes)} run(s), journal : {journal}")

    t0 = time.time()
    for n, args in enumerate(etapes, 1):
        log(f"--- run {n}/{len(etapes)} : {' '.join(args)}")
        with journal.open('a', encoding='utf-8') as f:
            code = subprocess.call(BANC + args, cwd=RACINE, stdout=f,
                                   stderr=subprocess.STDOUT)
        log(f"    code de sortie {code}   ({(time.time() - t0) / 60:.0f} min écoulées)")
        if code != 0:
            log("    ÉCHEC — on continue, le run suivant est indépendant")
        if n < len(etapes):
            time.sleep(a.pause)
    log(f"=== terminé en {(time.time() - t0) / 60:.0f} min ===")


if __name__ == '__main__':
    main()
