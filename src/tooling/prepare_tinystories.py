"""Nettoyage et découpage du corpus TinyStories.

Lit data/tinystories-raw.txt (téléchargé depuis HuggingFace), normalise les
caractères parasites, découpe histoire par histoire en train/val/test et écrit
les trois fichiers dans data/.

Le découpage se fait par histoire, jamais au milieu d'une : une histoire coupée
en deux polluerait la validation avec du texte vu à l'entraînement.
"""

import random
import unicodedata

RAW = "data/tinystories-raw.txt"
SEP = "<|endoftext|>"
SEED = 1337
PART_VAL = 0.05
PART_TEST = 0.05

# Reliquats d'encodage cp1252 et ponctuation typographique -> ASCII.
REMPLACEMENTS = {
    "‘": "'", "’": "'", "\x92": "'",
    "“": '"', "”": '"', "\x93": '"', "\x94": '"',
    "–": "-", "—": "-", "…": "...",
    "\xa0": " ",
}


def normalise(texte):
    for avant, apres in REMPLACEMENTS.items():
        texte = texte.replace(avant, apres)
    # Accents résiduels (é, ñ) -> lettre de base : quelques dizaines d'occurrences,
    # pas de quoi justifier des symboles supplémentaires dans le vocabulaire.
    texte = unicodedata.normalize("NFKD", texte)
    texte = "".join(c for c in texte if not unicodedata.combining(c))
    lignes = [ligne.rstrip() for ligne in texte.split("\n")]
    return "\n".join(ligne for ligne in lignes if ligne)


def main():
    brut = open(RAW, encoding="utf-8", errors="replace").read()

    morceaux = brut.split(SEP)

    # Le fichier brut de HuggingFace commence au milieu d'une histoire : le premier
    # morceau est un fragment sans début. Repere par sa minuscule initiale, et ecarte
    # (repere par Ethan le 31/07/2026).
    if morceaux and morceaux[0].lstrip()[:1].islower():
        print(f"fragment initial tronque ecarte ({len(morceaux[0].strip())} caracteres)")
        morceaux = morceaux[1:]

    histoires = []
    for morceau in morceaux:
        propre = normalise(morceau).strip()
        if propre:
            histoires.append(propre)

    random.Random(SEED).shuffle(histoires)

    n = len(histoires)
    n_val = int(n * PART_VAL)
    n_test = int(n * PART_TEST)
    parts = {
        "val": histoires[:n_val],
        "test": histoires[n_val:n_val + n_test],
        "train": histoires[n_val + n_test:],
    }

    alphabet = set()
    for nom, lot in parts.items():
        contenu = "\n\n".join(lot) + "\n"
        chemin = f"data/stories.{nom}.txt"
        with open(chemin, "w", encoding="utf-8") as f:
            f.write(contenu)
        alphabet |= set(contenu)
        print(f"{chemin:28} {len(lot):6} histoires  {len(contenu)/1e6:7.2f} Mo")

    print(f"\nseed={SEED}  {n} histoires  {len(alphabet)} caracteres distincts")
    print("alphabet:", "".join(sorted(alphabet)).replace("\n", "\\n"))


if __name__ == "__main__":
    main()
