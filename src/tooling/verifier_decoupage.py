"""Vérifie qu'un découpage en unités est réversible.

Zone verte (tooling) : vérification. Ne découpe rien lui-même — il appelle ta
fonction et juge son résultat.

    python3 src/tooling/verifier_decoupage.py
    python3 src/tooling/verifier_decoupage.py --fonction decouper
    python3 src/tooling/verifier_decoupage.py --corpus data/stories.train.txt

La propriété testée est la seule qui compte à ce stade :

    ''.join(decouper(texte)) == texte

Si elle est fausse, le script localise la première divergence et montre son
voisinage. Aucun BPE n'intervient : c'est un test de découpage pur.

Il rapporte aussi le rapport de redondance — le facteur que la mémorisation
transformera en gain de vitesse.
"""

import argparse
import itertools
import sys
import tempfile
from collections import Counter
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RACINE / "src" / "model"))


def trouver_fonction(nom):
    """Récupère la fonction de découpage dans encodeur.py, ou explique ce qui manque."""
    try:
        import encodeur
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"impossible d'importer src/model/encodeur.py — {type(e).__name__}: {e}")
    if not hasattr(encodeur, nom):
        dispo = [n for n in dir(encodeur) if callable(getattr(encodeur, n)) and not n.startswith("_")]
        raise SystemExit(f"encodeur.py n'a pas de fonction {nom!r}.\n"
                         f"fonctions trouvées : {', '.join(dispo) or '(aucune)'}\n"
                         f"utilise --fonction pour donner le bon nom.")
    return getattr(encodeur, nom)


def premiere_divergence(a, b):
    """Indice du premier caractère qui diffère entre deux chaînes."""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else None


def appeler(decouper, texte):
    """Appelle decouper, qu'il attende une chaîne ou un chemin de fichier.

    Un générateur ne lève rien à l'appel : on sonde le premier élément pour
    savoir quelle signature convient, puis on le remet en tête.
    """
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False)
    f.write(texte); f.close()
    try:                                   # 1. signature « chemin »
        it = iter(decouper(f.name))
        try:
            premier = next(it)
        except StopIteration:
            return []
        return itertools.chain([premier], it)
    except (OSError, TypeError, ValueError):
        return decouper(texte)             # 2. signature « chaîne »


def verifier(decouper, texte, nom_source):
    print(f"\n  source : {nom_source}  ({len(texte):,} caractères)".replace(",", " "))

    unites = appeler(decouper, texte)
    if not isinstance(unites, (list, tuple)):
        try:
            unites = list(unites)          # générateur : on matérialise pour le test
            print("  (la fonction est un générateur — matérialisé pour la vérification)")
        except TypeError:
            raise SystemExit(f"  la fonction rend un {type(unites).__name__} "
                             f"non parcourable ; on attend des unités.")
    non_texte = [u for u in unites[:1000] if not isinstance(u, str)]
    if non_texte:
        raise SystemExit(f"  les unités doivent être des chaînes ; "
                         f"trouvé un {type(non_texte[0]).__name__}.")

    recolle = "".join(unites)
    i = premiere_divergence(recolle, texte)

    print(f"  unités : {len(unites):,}".replace(",", " "))
    if i is None:
        print("\n  RÉVERSIBLE — recoller les unités redonne le texte exactement.\n")
    else:
        print(f"\n  ÉCHEC — divergence au caractère {i:,}".replace(",", " "))
        print(f"    attendu     : {texte[max(0, i-40):i+40]!r}")
        print(f"    reconstruit : {recolle[max(0, i-40):i+40]!r}")
        if len(recolle) != len(texte):
            print(f"    longueurs : {len(recolle):,} reconstruits contre {len(texte):,} attendus"
                  .replace(",", " "))
        print()
        return False

    # Ce que la mémorisation rapportera.
    c = Counter(unites)
    matiere = sum(len(u) for u in c)
    print(f"  unités distinctes : {len(c):,}".replace(",", " "))
    print(f"  redondance        : {len(unites)/len(c):,.0f}×  "
          f"(chaque unité distincte revient en moyenne autant de fois)".replace(",", " "))
    print(f"  matière à encoder : {matiere/1e6:.2f} Mo contre {len(texte)/1e6:.2f} Mo  "
          f"->  facteur {len(texte)/matiere:.0f}")
    print(f"\n  les 5 unités les plus fréquentes : "
          f"{', '.join(repr(u) for u, _ in c.most_common(5))}\n")
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description="Vérifie qu'un découpage en unités est réversible.")
    p.add_argument("--fonction", default="decouper",
                   help="nom de ta fonction de découpage dans encodeur.py (défaut : decouper)")
    p.add_argument("--corpus", default="data/stories.val.txt",
                   help="fichier à tester (défaut : data/stories.val.txt)")
    p.add_argument("--max-mo", type=float, default=0,
                   help="ne lire que les N premiers mégaoctets (0 = tout)")
    args = p.parse_args(argv)

    decouper = trouver_fonction(args.fonction)
    chemin = Path(args.corpus)
    if not chemin.is_file():
        raise SystemExit(f"fichier introuvable : {chemin}")

    texte = chemin.read_text(encoding="utf-8")
    if args.max_mo:
        texte = texte[:int(args.max_mo * 1_000_000)]

    ok = True
    # Cas limites d'abord : ils coûtent zéro et attrapent l'essentiel.
    limites = [("", "chaîne vide"), ("a", "un seul caractère"),
               (" ", "un seul espace"), ("\n\n", "deux sauts de ligne"),
               ("  a", "espaces consécutifs"), ("Once upon", "cas nominal")]
    print("\n  === cas limites ===")
    for t, libelle in limites:
        try:
            r = "".join(appeler(decouper, t))
            etat = "OK" if r == t else f"ÉCHEC — rend {r!r}"
        except Exception as e:  # noqa: BLE001
            etat = f"ÉCHEC — {type(e).__name__}: {e}"
        if not etat.startswith("OK"):
            ok = False
        print(f"    {libelle:24s} {t!r:12s} {etat}")

    print("\n  === corpus ===")
    ok &= verifier(decouper, texte, chemin.name)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
