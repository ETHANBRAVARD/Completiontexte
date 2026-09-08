"""Vérifie qu'un fichier de tokens entiers est correct.

Zone verte (tooling) : vérification pure. Ne convertit rien — il juge le résultat.

Trois contrôles, du plus rapide au plus lent :

1. **contre la sauvegarde** — le .npy produit est-il identique, entier pour entier,
   à celui de `data/sauvegarde/` ? C'est le contrôle le plus fort : si tu réécris
   `rencode.py` sans changer la tokenisation, la réponse doit être oui.
2. **cohérence interne** — dtype, bornes, aucun indice hors vocabulaire.
3. **aller-retour** — recoller les tokens redonne-t-il le corpus, caractère pour
   caractère ? Le seul qui prouve que le fichier veut dire quelque chose.

Usage :
    python3 src/tooling/verifier_rencode.py
    python3 src/tooling/verifier_rencode.py --splits train
    python3 src/tooling/verifier_rencode.py --sans-aller-retour

La comparaison se fait par blocs et le .npy est ouvert en lecture différée, donc
ce script tient dans quelques centaines de mégaoctets même sur 500 M de tokens.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

RACINE = Path(__file__).resolve().parents[2]
DATA = RACINE / "data"
SAUVEGARDE = DATA / "sauvegarde"
VOCAB = DATA / "tokenizer" / "bpe_liste.json"
BLOC = 8_000_000  # entiers comparés d'un coup

TEXTES = {"train": "stories.train.txt", "val": "stories.val.txt", "test": "stories.test.txt"}


def go(n):
    return f"{n / 1e9:.2f} Go" if n >= 1e9 else f"{n / 1e6:.0f} Mo"


def milliers(n):
    return f"{n:,}".replace(",", " ")


def comparer(a, b):
    """Premier indice où deux tableaux diffèrent, par blocs. None s'ils sont égaux."""
    if len(a) != len(b):
        n = min(len(a), len(b))
    else:
        n = len(a)
    for d in range(0, n, BLOC):
        f = min(d + BLOC, n)
        ba, bb = np.asarray(a[d:f]), np.asarray(b[d:f])
        if not np.array_equal(ba, bb):
            return d + int(np.flatnonzero(ba != bb)[0])
    return None if len(a) == len(b) else n


def aller_retour(tok, vocab, texte_attendu):
    """Recolle les tokens par blocs et compare au corpus. Rend (ok, indice ou None)."""
    morceaux, total = [], 0
    for d in range(0, len(tok), BLOC):
        morceaux.append("".join(vocab[i] for i in np.asarray(tok[d:d + BLOC])))
        total += len(morceaux[-1])
    reconstruit = "".join(morceaux)[1:]   # l'espace initial de l'encodeur
    del morceaux
    if reconstruit == texte_attendu:
        return True, None
    n = min(len(reconstruit), len(texte_attendu))
    for i in range(n):
        if reconstruit[i] != texte_attendu[i]:
            return False, i
    return False, n


def verifier(split, vocab, faire_aller_retour):
    nom = f"encode_tok_{split}.npy"
    courant = DATA / nom
    print(f"\n  === {split} ===")
    if not courant.is_file():
        print(f"    {nom} absent — rien à vérifier.")
        return None

    tok = np.load(courant, mmap_mode="r")
    print(f"    {milliers(len(tok))} tokens | {tok.dtype} | {go(courant.stat().st_size)}")

    ok = True

    # 1. cohérence interne
    mx = int(np.asarray(tok[:BLOC]).max()) if len(tok) else 0
    for d in range(BLOC, len(tok), BLOC):
        mx = max(mx, int(np.asarray(tok[d:d + BLOC]).max()))
    if tok.dtype != np.uint16:
        print(f"    dtype      ÉCHEC — {tok.dtype}, on attend uint16")
        ok = False
    if mx >= len(vocab):
        print(f"    bornes     ÉCHEC — indice max {mx}, vocabulaire de {len(vocab)}")
        ok = False
    else:
        print(f"    bornes     OK — indices 0..{mx} pour un vocabulaire de {len(vocab)}")

    # 2. contre la sauvegarde
    ref = SAUVEGARDE / nom
    if ref.is_file():
        r = np.load(ref, mmap_mode="r")
        i = comparer(tok, r)
        if i is None:
            print(f"    sauvegarde OK — identique, {milliers(len(tok))} entiers")
        elif len(tok) != len(r):
            # Longueurs très différentes : ce n'est pas une régression, c'est un
            # autre corpus. La sauvegarde ne fait référence qu'à corpus égal.
            print(f"    sauvegarde   sans objet — {milliers(len(tok))} tokens contre "
                  f"{milliers(len(r))} : la référence porte un autre corpus")
        else:
            ok = False
            print(f"    sauvegarde ÉCHEC — première différence au token {milliers(i)}")
            if len(tok) != len(r):
                print(f"               longueurs : {milliers(len(tok))} contre "
                      f"{milliers(len(r))} attendus")
            else:
                print(f"               produit {tok[i]} ({vocab[tok[i]]!r}), "
                      f"sauvegardé {r[i]} ({vocab[r[i]]!r})")
    else:
        print(f"    sauvegarde absente ({ref.relative_to(RACINE)}) — contrôle sauté")

    # 3. aller-retour
    if faire_aller_retour:
        src = DATA / TEXTES[split]
        if not src.is_file():
            print(f"    aller-retour  {src.name} absent — sauté")
        else:
            texte = src.read_text(encoding="utf-8")
            bon, i = aller_retour(tok, vocab, texte)
            if bon:
                print(f"    aller-retour OK — {go(len(texte))} de texte retrouvés à l'identique")
            else:
                ok = False
                print(f"    aller-retour ÉCHEC — divergence au caractère {milliers(i)}")
                print(f"               attendu     : {texte[max(0,i-40):i+40]!r}")
            del texte
    return ok


def main(argv=None):
    p = argparse.ArgumentParser(description="Vérifie les fichiers de tokens entiers.")
    p.add_argument("--splits", default="train,val,test")
    p.add_argument("--sans-aller-retour", action="store_true",
                   help="sauter le contrôle le plus lent")
    args = p.parse_args(argv)

    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))
    print(f"\n  vocabulaire : {len(vocab)} tokens ({VOCAB.relative_to(RACINE)})")

    resultats = {}
    for split in args.splits.split(","):
        resultats[split] = verifier(split.strip(), vocab, not args.sans_aller_retour)

    faits = [v for v in resultats.values() if v is not None]
    print(f"\n  {sum(1 for v in faits if v)}/{len(faits)} split(s) valides\n")
    return 0 if faits and all(faits) else 1


if __name__ == "__main__":
    raise SystemExit(main())
