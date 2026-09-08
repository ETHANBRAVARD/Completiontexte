"""Nettoyage et découpage du corpus TinyStories, entièrement en flux.

Zone verte (tooling) : préparation de données, aucune logique de modèle.

Le fichier brut est lu par blocs et découpé au fil de l'eau sur le séparateur
d'histoires. Chaque histoire nettoyée part immédiatement dans un fichier de
travail ; seule sa **position** (offset, longueur) est conservée en mémoire, soit
16 octets par histoire au lieu de son texte.

Conséquence : la mémoire ne dépend plus de la taille du corpus retenu. Sur les
2,2 Go du fichier d'entraînement, la pointe reste sous 300 Mo là où la version
précédente montait vers 6 Go — elle accumulait toutes les histoires en mémoire,
puis en fabriquait une seconde copie par `"\\n\\n".join(...)` avant d'écrire.

Le mélange porte sur les positions, pas sur le texte. `random.shuffle` ne dépend
que de la graine et du nombre d'éléments : la permutation est donc **identique** à
celle de l'ancienne version, et les fichiers produits sont octet pour octet les
mêmes à graine et `--max-mo` égaux.

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
from array import array
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


class Alphabet:
    """Ensemble des octets vus, tenu à jour sans reparcourir le texte en Python.

    `bytes.translate` supprime en C tous les octets déjà connus ; ce qui reste est
    forcément nouveau. Passé les premières histoires il ne reste jamais rien, et le
    coût tombe à un balayage C par histoire.
    """

    def __init__(self):
        self.vus = set()
        self._connus = b""

    def ajoute(self, donnees: bytes):
        reste = donnees.translate(None, self._connus)
        if reste:
            self.vus |= set(reste)
            self._connus = bytes(sorted(self.vus))

    def caracteres(self):
        return {chr(o) for o in self.vus}


def nettoyer_vers_fichier(chemin: Path, limite_octets: int, travail: Path, log):
    """Nettoie le corpus vers `travail` et rend les positions des histoires.

    Rend (offsets, longueurs) : deux tableaux d'entiers, 8 octets par histoire.
    Le premier morceau est écarté s'il commence par une minuscule : le fichier brut
    de HuggingFace peut débuter au milieu d'une histoire (repéré par Ethan, 31/07/2026).
    """
    offsets, longueurs = array("q"), array("q")
    volume = position = lus = ecartees = 0
    reste = ""
    premier = True
    connus = alphabet_gele()
    if connus:
        log(f"  alphabet gelé : {len(connus)} caractères — les histoires qui en "
            f"contiennent d'autres seront écartées")

    def garder(propre: str, sortie) -> bool:
        """Écrit une histoire retenue et note sa position. Rend False si écartée."""
        nonlocal position, volume
        if not propre:
            return False
        # Quelques dizaines d'histoires du corpus contiennent des caractères
        # chinois ou des emojis égarés. On écarte l'histoire entière plutôt
        # que d'en retirer les caractères, ce qui laisserait des mots mutilés.
        if not propre.isascii() or (connus and not set(propre) <= connus):
            return None  # écartée, à distinguer du morceau vide
        donnees = propre.encode("ascii")
        sortie.write(donnees)
        offsets.append(position)
        longueurs.append(len(donnees))
        position += len(donnees)
        volume += len(donnees) + 2  # le "\n\n" de jointure
        return True

    with travail.open("wb") as sortie, \
            chemin.open(encoding="utf-8", errors="replace") as f:
        atteint = False
        while not atteint:
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
                etat = garder(normalise(morceau).strip(), sortie)
                if etat is None:
                    ecartees += 1
                elif etat and limite_octets and volume >= limite_octets:
                    log(f"  limite atteinte après {lus/1e6:.0f} Mo lus sur "
                        f"{chemin.stat().st_size/1e6:.0f}")
                    atteint = True
                    break

        if not atteint:
            if garder(normalise(reste).strip(), sortie) is None:
                ecartees += 1

    if ecartees:
        log(f"  {ecartees} histoires écartées (caractères hors ASCII)")
    return offsets, longueurs


def ecrire_split(travail, indices, offsets, longueurs, chemin: Path):
    """Recopie les histoires d'un split depuis le fichier de travail.

    Rend (nombre d'octets écrits, alphabet du split). Une histoire à la fois : la
    mémoire ne dépend pas de la taille du split.
    """
    alpha = Alphabet()
    ecrits = 0
    with chemin.open("wb") as sortie:
        for rang, i in enumerate(indices):
            if rang:
                sortie.write(b"\n\n")
                ecrits += 2
            travail.seek(offsets[i])
            donnees = travail.read(longueurs[i])
            sortie.write(donnees)
            alpha.ajoute(donnees)
            ecrits += len(donnees)
        sortie.write(b"\n")
        ecrits += 1
    alpha.ajoute(b"\n")
    return ecrits, alpha.caracteres()


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
    p.add_argument("--garder-travail", action="store_true",
                   help="ne pas supprimer le fichier de travail (débogage)")
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

    travail_chemin = DATA / f"{args.prefixe}.travail.tmp"
    offsets, longueurs = nettoyer_vers_fichier(
        brut, int(args.max_mo * 1e6), travail_chemin, log)
    n = len(offsets)
    log(f"  {n} histoires retenues  "
        f"({travail_chemin.stat().st_size/1e6:.0f} Mo de texte nettoyé)")

    # Mélange des positions, pas du texte. La permutation ne dépend que de la
    # graine et du nombre d'éléments : elle est identique à celle qu'on obtiendrait
    # en mélangeant les histoires elles-mêmes.
    ordre = list(range(n))
    random.Random(args.seed).shuffle(ordre)

    n_val = int(n * args.val_frac)
    n_test = int(n * args.test_frac)
    parts = {
        "val": ordre[:n_val],
        "test": ordre[n_val:n_val + n_test],
        "train": ordre[n_val + n_test:],
    }

    log()
    alphabets = {}
    try:
        with travail_chemin.open("rb") as travail:
            for nom, indices in parts.items():
                chemin = DATA / f"{args.prefixe}.{nom}.txt"
                taille, alphabets[nom] = ecrire_split(
                    travail, indices, offsets, longueurs, chemin)
                log(f"  {str(chemin.relative_to(ROOT)):32} {len(indices):7} histoires "
                    f"{taille/1e6:8.2f} Mo")
    finally:
        if not args.garder_travail:
            travail_chemin.unlink(missing_ok=True)

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
