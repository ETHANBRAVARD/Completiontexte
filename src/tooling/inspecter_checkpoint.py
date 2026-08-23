"""Inspection d'un checkpoint PyTorch (.pt).

Zone verte (tooling) : lecture et affichage. Aucune logique de modèle.

    python3 src/tooling/inspecter_checkpoint.py              # le plus récent
    python3 src/tooling/inspecter_checkpoint.py <chemin.pt>
    python3 src/tooling/inspecter_checkpoint.py --stats      # + min/max/moyenne/écart-type
    python3 src/tooling/inspecter_checkpoint.py --cle W_q    # une seule clé, en détail

Un `.pt` est une archive zip : l'éditeur n'y montre que du binaire. Ce script en
donne la structure — clés, types, formes, nombre d'éléments — **sans rapatrier
les poids en mémoire** (`mmap=True`), donc en une fraction de seconde.

Le script n'interprète rien : il ne décide pas ce qui est un paramètre et ce qui
est de l'état d'optimiseur. Il affiche ce que le fichier contient, clé par clé,
avec le compte d'éléments de chacune. La lecture, elle, est à toi.
"""

import argparse
import sys
from pathlib import Path

import torch

RACINE = Path(__file__).resolve().parents[2]
SAVE_RUNS = RACINE / "runs" / "save_runs"


# ----------------------------------------------------------------- sélection

def dernier_checkpoint(dossier_racine=SAVE_RUNS):
    """Le checkpoint du run le plus récent, au pas le plus élevé.

    Même règle que `generation.py` : le dossier de run au nom le plus grand
    (les noms sont horodatés, donc l'ordre alphabétique est l'ordre du temps),
    puis le fichier au numéro de pas le plus élevé — comparé comme un **entier**,
    pas comme une chaîne, sinon 9000 passerait devant 30000.
    """
    dossiers = [d for d in dossier_racine.iterdir() if d.is_dir()] if dossier_racine.is_dir() else []
    dossiers = [d for d in dossiers if any(d.glob("*-sauvegarde_*.pt"))]
    if not dossiers:
        return None
    run = max(dossiers, key=lambda d: d.name)
    return max(run.glob("*-sauvegarde_*.pt"), key=lambda f: int(f.stem.split("_")[-1]))


def charger(chemin):
    """Ouvre le checkpoint le moins cher possible, en dégradant si besoin.

    `mmap=True` laisse les tenseurs sur le disque : on ne lit que les métadonnées.
    `weights_only=True` interdit au dépicklage d'exécuter du code arbitraire —
    ouvrir un `.pt` d'origine inconnue est sinon équivalent à lancer un script.
    Les deux peuvent échouer selon le format ou le contenu ; on redescend d'un
    cran à chaque fois, en le disant.
    """
    tentatives = [
        (dict(map_location="cpu", weights_only=True, mmap=True), None),
        (dict(map_location="cpu", weights_only=True), "mmap indisponible, lecture complète"),
        (dict(map_location="cpu"), "weights_only refusé — le dépicklage exécute du code"),
    ]
    dernier = None
    for options, avertissement in tentatives:
        try:
            objet = torch.load(chemin, **options)
            if avertissement:
                print(f"  ! {avertissement}")
            return objet
        except Exception as e:  # noqa: BLE001
            dernier = e
    raise dernier


# ----------------------------------------------------------------- description

def elements(valeur):
    """Nombre d'éléments, en descendant dans les listes et tuples."""
    if torch.is_tensor(valeur):
        return valeur.numel()
    if isinstance(valeur, (list, tuple)):
        return sum(elements(e) for e in valeur)
    return 0


def octets(valeur):
    """Taille en octets, en tenant compte du dtype réel de chaque tenseur."""
    if torch.is_tensor(valeur):
        return valeur.numel() * valeur.element_size()
    if isinstance(valeur, (list, tuple)):
        return sum(octets(e) for e in valeur)
    return 0


def forme(valeur):
    """Une description courte : type, forme, et homogénéité des listes."""
    if torch.is_tensor(valeur):
        return f"tenseur {tuple(valeur.shape)} {valeur.dtype}".replace("torch.", "")
    if isinstance(valeur, (list, tuple)):
        nom = "liste" if isinstance(valeur, list) else "tuple"
        if not valeur:
            return f"{nom} vide"
        if all(torch.is_tensor(e) for e in valeur):
            formes = {tuple(e.shape) for e in valeur}
            if len(formes) == 1:
                return f"{nom} de {len(valeur)} × tenseur {formes.pop()}"
            return f"{nom} de {len(valeur)} tenseurs, {len(formes)} formes distinctes"
        return f"{nom} de {len(valeur)} éléments"
    if isinstance(valeur, (int, float, bool, str)) or valeur is None:
        return f"{type(valeur).__name__} = {valeur!r}"
    return type(valeur).__name__


def stats(valeur):
    """min / max / moyenne / écart-type sur l'ensemble d'un tenseur ou d'une liste.

    Force la lecture réelle des données : c'est le seul appel qui coûte.
    """
    tenseurs = [valeur] if torch.is_tensor(valeur) else \
               [e for e in valeur if torch.is_tensor(e)] if isinstance(valeur, (list, tuple)) else []
    if not tenseurs:
        return None
    plat = torch.cat([t.reshape(-1).float() for t in tenseurs])
    if plat.numel() == 0:
        return None
    return plat.min().item(), plat.max().item(), plat.mean().item(), plat.std().item()


# ----------------------------------------------------------------- affichage

def lisible(n):
    for unite in ("o", "ko", "Mo", "Go"):
        if n < 1024 or unite == "Go":
            return f"{n:.0f} {unite}" if unite == "o" else f"{n:.1f} {unite}"
        n /= 1024


def afficher(chemin, objet, avec_stats=False, cle=None):
    taille = Path(chemin).stat().st_size
    print(f"\n  fichier : {chemin}")
    print(f"  taille  : {lisible(taille)}")

    if not isinstance(objet, dict):
        print(f"  contenu : {forme(objet)}  (pas un dictionnaire)")
        return

    cles = list(objet) if cle is None else [k for k in objet if k == cle]
    if cle is not None and not cles:
        print(f"\n  clé absente : {cle!r}")
        print(f"  clés disponibles : {', '.join(map(str, objet))}")
        return

    print(f"  clés    : {len(objet)}\n")

    larg = max((len(str(k)) for k in cles), default=4)
    entete = f"  {'clé':<{larg}}  {'contenu':<44} {'éléments':>13} {'taille':>10}"
    if avec_stats:
        entete += f"  {'min':>10} {'max':>10} {'moyenne':>10} {'écart-type':>10}"
    print(entete)
    print("  " + "-" * (len(entete) - 2))

    total_e = total_o = 0
    for k in cles:
        v = objet[k]
        e, o = elements(v), octets(v)
        total_e += e
        total_o += o
        ligne = (f"  {str(k):<{larg}}  {forme(v):<44} "
                 f"{e if e else '':>13} {lisible(o) if o else '':>10}")
        if avec_stats:
            s = stats(v)
            ligne += (f"  {s[0]:>10.4g} {s[1]:>10.4g} {s[2]:>10.4g} {s[3]:>10.4g}"
                      if s else f"  {'':>10} {'':>10} {'':>10} {'':>10}")
        print(ligne)

    print("  " + "-" * (len(entete) - 2))
    print(f"  {'total':<{larg}}  {'':<44} {total_e:>13} {lisible(total_o):>10}")

    if cle is None and total_o:
        ecart = taille - total_o
        print(f"\n  {lisible(total_o)} de tenseurs pour un fichier de {lisible(taille)} "
              f"— {lisible(abs(ecart))} de {'plus' if ecart > 0 else 'moins'} "
              f"(index zip, scalaires, alignement).")


# ----------------------------------------------------------------- entrée

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Affiche la structure d'un checkpoint .pt sans le charger en mémoire.")
    p.add_argument("chemin", nargs="?", help="le .pt à lire (défaut : le plus récent de runs/save_runs)")
    p.add_argument("--stats", action="store_true",
                   help="ajoute min/max/moyenne/écart-type — force la lecture réelle des poids")
    p.add_argument("--cle", help="n'afficher qu'une clé")
    args = p.parse_args(argv)

    chemin = Path(args.chemin) if args.chemin else dernier_checkpoint()
    if chemin is None:
        print(f"aucun checkpoint trouvé dans {SAVE_RUNS.relative_to(RACINE)}/", file=sys.stderr)
        return 1
    if not chemin.is_file():
        print(f"fichier introuvable : {chemin}", file=sys.stderr)
        return 1

    afficher(chemin, charger(chemin), avec_stats=args.stats, cle=args.cle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
