"""Compare deux versions de `encodeur.encode` token par token.

Zone verte (tooling) : vérification pure. Ce script n'encode rien lui-même — il
lance deux versions de TON encodeur sur le même texte et compare leurs sorties.

Pourquoi ce script plutôt que l'aller-retour habituel : ton contrôle
`''.join(tokens)[1:] == texte` teste la RÉVERSIBILITÉ. Il passerait même si la
nouvelle version découpait le texte autrement, du moment que le recollage
redonne le texte. Or ce qui doit être vrai ici est plus fort :

    la nouvelle version produit EXACTEMENT la même suite de tokens que l'ancienne

Sans quoi tes checkpoints existants, entraînés sur les indices de l'ancienne
tokenisation, deviennent silencieusement illisibles.

La référence est la version committée dans git (HEAD par défaut) : c'est celle
qui a produit `data/encode_tok_*.npy` et les modèles de `runs/`.

Usage :
    python3 src/tooling/comparer_encodeurs.py
    python3 src/tooling/comparer_encodeurs.py --max-mo 5
    python3 src/tooling/comparer_encodeurs.py --ref a5c3ef8 --corpus data/stories.train.txt

Chaque version tourne dans un bac à sable isolé, avec son propre `data/` : ton
`data/encode.json` réel n'est jamais touché.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
FICHIER = "src/model/encodeur.py"

# Le sous-processus importe le module posé dans son bac à sable et l'appelle
# avec la signature minimale — commune à toutes les versions.
LANCEUR = ("import sys; sys.path.insert(0, '.'); import encodeur; "
           "encodeur.encode(sys.argv[1], sys.argv[2])")


def duree(s):
    return f"{s:.1f} s" if s < 60 else f"{int(s // 60)} min {s % 60:.0f} s"


def poser_version(source_code, bac):
    """Installe un encodeur dans un bac à sable prêt à tourner."""
    bac.mkdir(parents=True, exist_ok=True)
    (bac / "encodeur.py").write_text(source_code, encoding="utf-8")
    (bac / "data").mkdir(exist_ok=True)
    # L'encodeur lit data/tokenizer/ par un chemin relatif : on le lui rend visible
    lien = bac / "data" / "tokenizer"
    if not lien.exists():
        lien.symlink_to(RACINE / "data" / "tokenizer")
    return bac


def executer(bac, corpus, bpe, libelle):
    """Lance l'encodeur du bac sur le corpus, et rend (tokens, secondes)."""
    t = time.perf_counter()
    r = subprocess.run([sys.executable, "-c", LANCEUR, str(corpus), str(bpe)],
                       cwd=bac, capture_output=True, text=True)
    dt = time.perf_counter() - t

    if r.returncode != 0:
        print(f"\n  {libelle} a planté (code {r.returncode}) :")
        print("  " + "\n  ".join((r.stderr or r.stdout).strip().splitlines()[-12:]))
        raise SystemExit(1)

    produits = sorted(f for f in (bac / "data").iterdir()
                      if f.suffix in (".json", ".npy"))
    if not produits:
        print(f"\n  {libelle} n'a rien écrit dans son dossier data/.")
        if r.stdout.strip():
            print("  sortie du programme :")
            print("  " + "\n  ".join(r.stdout.strip().splitlines()[-12:]))
        print("  (si ta nouvelle version rend la liste au lieu de l'écrire,\n"
              "   ce comparateur ne peut pas la lire — garde une écriture sur disque.)")
        raise SystemExit(1)
    if len(produits) > 1:
        print(f"  {libelle} : plusieurs fichiers écrits, "
              f"je prends {produits[0].name}")

    if produits[0].suffix == ".npy":
        tokens = np.load(produits[0]).tolist()
    else:
        contenu = json.loads(produits[0].read_text(encoding="utf-8"))
        tokens = contenu["encode"] if isinstance(contenu, dict) else contenu
    if tokens and isinstance(tokens[0], int):
        # sortie en indices : on repasse par le vocabulaire pour comparer des tokens
        vocab = json.loads((RACINE / "data" / "tokenizer" / "bpe_liste.json")
                           .read_text(encoding="utf-8"))
        print(f"  {libelle} rend des indices — traduits via bpe_liste.json ({len(vocab)} tokens)")
        tokens = [vocab[i] for i in tokens]
    if r.stdout.strip():
        print(f"  {libelle} a aussi affiché : {r.stdout.strip()[:200]}")
    return tokens, dt


def premiere_divergence(a, b):
    for i in range(min(len(a), len(b))):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def rapporter(ancien, nouveau, texte):
    print(f"\n  === comparaison ===")
    print(f"  référence : {len(ancien):,} tokens".replace(",", " "))
    print(f"  courant   : {len(nouveau):,} tokens".replace(",", " "))

    i = premiere_divergence(ancien, nouveau)
    if i is None:
        print(f"\n  IDENTIQUE — les deux versions produisent la même suite de tokens.")
        print(f"  ratio : {len(texte) / len(ancien):.4f} caractères par token\n")
        return True

    print(f"\n  DIVERGENCE au token n° {i:,}".replace(",", " "))
    d = max(0, i - 5)
    print(f"    référence [{d}:{i + 5}] : {ancien[d:i + 5]}")
    print(f"    courant   [{d}:{i + 5}] : {nouveau[d:i + 5]}")
    if len(ancien) != len(nouveau):
        print(f"    longueurs différentes : {len(nouveau):,} contre {len(ancien):,}"
              .replace(",", " "))
    contexte = "".join(ancien[max(0, i - 20):i])
    print(f"    texte juste avant : ...{contexte[-70:]!r}")
    print(f"    la suite du texte : {''.join(ancien[i:i + 12])!r}")
    print("\n  Les deux tokenisations sont incompatibles : un modèle entraîné sur\n"
          "  l'une ne peut pas lire l'autre.\n")
    return False


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare deux versions de encodeur.encode.")
    p.add_argument("--ref", default="HEAD",
                   help="version git servant de référence (défaut : HEAD)")
    p.add_argument("--corpus", default="data/stories.val.txt")
    p.add_argument("--max-mo", type=float, default=2.0,
                   help="mégaoctets de texte comparés, 0 = tout (défaut : 2)")
    p.add_argument("--bpe", default="data/tokenizer/bpe.json")
    args = p.parse_args(argv)

    corpus = RACINE / args.corpus
    bpe = RACINE / args.bpe
    for f in (corpus, bpe):
        if not f.is_file():
            raise SystemExit(f"fichier introuvable : {f}")

    courant = (RACINE / FICHIER).read_text(encoding="utf-8")
    git = subprocess.run(["git", "show", f"{args.ref}:{FICHIER}"],
                         cwd=RACINE, capture_output=True, text=True)
    if git.returncode != 0:
        raise SystemExit(f"impossible de lire {FICHIER} à la version {args.ref} :\n"
                         f"{git.stderr.strip()}")
    reference = git.stdout

    if reference == courant:
        print(f"\n  Le fichier courant est identique à la version {args.ref} : "
              f"rien à comparer.\n")
        return 0

    with tempfile.TemporaryDirectory(prefix="comparer_encodeurs_") as tmp:
        tmp = Path(tmp)
        texte = corpus.read_text(encoding="utf-8")
        if args.max_mo:
            texte = texte[:int(args.max_mo * 1e6)]
        echantillon = tmp / "echantillon.txt"
        echantillon.write_text(texte, encoding="utf-8")

        print(f"\n  corpus    : {corpus.name}, {len(texte) / 1e6:.2f} Mo")
        print(f"  référence : {FICHIER} à la version {args.ref}")
        print(f"  courant   : {FICHIER} tel qu'il est sur le disque\n")

        bac_ref = poser_version(reference, tmp / "reference")
        bac_new = poser_version(courant, tmp / "courant")

        ancien, t_ref = executer(bac_ref, echantillon, bpe, "la référence")
        print(f"  référence : {duree(t_ref)}")
        nouveau, t_new = executer(bac_new, echantillon, bpe, "la version courante")
        print(f"  courant   : {duree(t_new)}"
              + (f"   ->  {t_ref / t_new:.1f}× plus rapide" if t_new > 0 and t_new < t_ref
                 else f"   ->  {t_new / t_ref:.1f}× plus lent" if t_new > t_ref else ""))

        return 0 if rapporter(ancien, nouveau, texte) else 1


if __name__ == "__main__":
    raise SystemExit(main())
