"""Nettoyage et découpage du corpus TinyStories, en lecture incrémentale.

Zone verte (tooling) : préparation de données, aucune logique de modèle.

Remplace `src/tooling/prepare_tinystories.py`, qui chargeait le fichier entier en
mémoire (6 à 7 Go de pointe sur le fichier d'entraînement de 2,2 Go). Ici le fichier
est lu par blocs et découpé au fil de l'eau sur le séparateur d'histoires ; la mémoire
reste bornée par la taille du corpus retenu, pas par celle du fichier source.

Le découpage se fait par histoire, jamais au milieu d'une : une histoire coupée en
deux polluerait la validation avec du texte vu à l'entraînement.

Usage :
    python scripts/prepare_stories.py --max-mo 100
    python scripts/prepare_stories.py --raw data/tinystories-raw-train.txt --max-mo 100
    python scripts/prepare_stories.py --max-mo 0          # tout le fichier

`--max-mo` compte les mégaoctets de texte **nettoyé** conservés, pas les octets lus.
La coupure tombe toujours sur une frontière d'histoire.
"""

import argparse
import random
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SEP = "<|endoftext|>"
BLOC = 32 * 1024 * 1024  # 32 Mo lus à la fois

# Reliquats d'encodage cp1252 et ponctuation typographique -> ASCII.
REMPLACEMENTS = {
    "‘": "'", "’": "'", "\x92": "'",
    "“": '"', "”": '"', "\x93": '"', "\x94": '"',
    "–": "-", "—": "-", "…": "...",
    "\xa0": " ",
}


def normalise(texte: str) -> str:
    for avant, apres in REMPLACEMENTS.items():
        texte = texte.replace(avant, apres)
    # Accents résiduels (é, ñ) -> lettre de base : quelques dizaines d'occurrences,
    # pas de quoi justifier des symboles supplémentaires dans le vocabulaire.
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    lignes = [ligne.rstrip() for ligne in texte.split("\n")]
    return "\n".join(ligne for ligne in lignes if ligne)


def alphabet_gele():
    """Caractères connus du vocabulaire figé, s'il existe.

    Une histoire contenant un caractère absent de cet alphabet ferait s'arrêter
    l'encodeur — autant l'écarter ici, où c'est mesurable, plutôt qu'après une
    heure de calcul.
    """
    bpe = DATA / "tokenizer" / "bpe.json"
    if not bpe.exists():
        return None
    import json
    return set(json.loads(bpe.read_text(encoding="utf-8"))["alphabet"])


def lire_histoires(chemin: Path, limite_octets: int, log):
    """Rend les histoires nettoyées, en lisant le fichier par blocs.

    S'arrête dès que le volume nettoyé atteint `limite_octets` (0 = tout lire).
    Le premier morceau est écarté s'il commence par une minuscule : le fichier brut
    de HuggingFace peut débuter au milieu d'une histoire (repéré par Ethan, 31/07/2026).
    """
    histoires = []
    volume = 0
    reste = ""
    premier = True
    lus = 0
    ecartees = 0
    connus = alphabet_gele()
    if connus:
        log(f"  alphabet gelé : {len(connus)} caractères — les histoires qui en "
            f"contiennent d'autres seront écartées")

    with chemin.open(encoding="utf-8", errors="replace") as f:
        while True:
            bloc = f.read(BLOC)
            if not bloc:
                break
            lus += len(bloc)
            morceaux = (reste + bloc).split(SEP)
            reste = morceaux.pop()  # dernier morceau : peut-être incomplet

            for morceau in morceaux:
                if premier:
                    premier = False
                    if morceau.lstrip()[:1].islower():
                        log(f"  fragment initial tronqué écarté "
                            f"({len(morceau.strip())} caractères)")
                        continue
                propre = normalise(morceau).strip()
                if not propre:
                    continue
                # Quelques dizaines d'histoires du corpus contiennent des caractères
                # chinois ou des emojis égarés. On écarte l'histoire entière plutôt
                # que d'en retirer les caractères, ce qui laisserait des mots mutilés.
                if not propre.isascii() or (connus and not set(propre) <= connus):
                    ecartees += 1
                    continue
                histoires.append(propre)
                volume += len(propre) + 2  # le "\n\n" de jointure
                if limite_octets and volume >= limite_octets:
                    log(f"  limite atteinte après {lus/1e6:.0f} Mo lus sur "
                        f"{chemin.stat().st_size/1e6:.0f}")
                    if ecartees:
                        log(f"  {ecartees} histoires écartées (caractères hors ASCII)")
                    return histoires

    propre = normalise(reste).strip()
    if propre and propre.isascii() and not (connus and not set(propre) <= connus):
        histoires.append(propre)
    if ecartees:
        log(f"  {ecartees} histoires écartées (caractères hors ASCII)")
    return histoires


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", type=Path, default=DATA / "tinystories-raw-train.txt")
    p.add_argument("--max-mo", type=float, default=100.0,
                   help="mégaoctets de texte nettoyé à conserver (0 = tout)")
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--test-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--prefixe", default="stories",
                   help="préfixe des fichiers de sortie dans data/")
    args = p.parse_args()

    brut = args.raw if args.raw.is_absolute() else ROOT / args.raw
    if not brut.exists():
        print(f"fichier brut introuvable : {brut}", file=sys.stderr)
        return 2

    def log(m=""):
        print(m, flush=True)

    log(f"source : {brut.relative_to(ROOT)} ({brut.stat().st_size/1e6:.0f} Mo)")
    log(f"cible  : {args.max_mo:.0f} Mo de texte nettoyé"
        if args.max_mo else "cible  : tout le fichier")

    histoires = lire_histoires(brut, int(args.max_mo * 1e6), log)
    log(f"  {len(histoires)} histoires retenues")

    random.Random(args.seed).shuffle(histoires)

    n = len(histoires)
    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    parts = {
        "val": histoires[:n_val],
        "test": histoires[n_val:n_val + n_test],
        "train": histoires[n_val + n_test:],
    }

    log()
    alphabets = {}
    for nom, lot in parts.items():
        contenu = "\n\n".join(lot) + "\n"
        chemin = DATA / f"{args.prefixe}.{nom}.txt"
        chemin.write_text(contenu, encoding="utf-8")
        alphabets[nom] = set(contenu)
        log(f"  {str(chemin.relative_to(ROOT)):32} {len(lot):7} histoires "
            f"{len(contenu)/1e6:8.2f} Mo")
        del contenu

    alphabet = set().union(*alphabets.values())
    log(f"\nseed={args.seed}  {n} histoires  {len(alphabet)} caractères distincts")
    log("alphabet : " + "".join(sorted(alphabet)).replace("\n", "\\n"))

    # Le vocabulaire sera appris sur train. Tout caractère présent dans val ou test
    # mais absent de train serait inconnu à l'encodage — l'encodeur s'arrêterait
    # dessus après une demi-heure de travail. Autant le savoir maintenant.
    log()
    manquants = {nom: sorted(alphabets[nom] - alphabets["train"])
                 for nom in ("val", "test")}
    if any(manquants.values()):
        for nom, chars in manquants.items():
            if chars:
                log(f"ATTENTION — {len(chars)} caractères de {nom} absents de train : {chars}")
        log("           l'encodage de ce split échouera. Augmente --max-mo,")
        log("           ou normalise ces caractères ici.")
    else:
        log("alphabets de val et test entièrement inclus dans celui de train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())