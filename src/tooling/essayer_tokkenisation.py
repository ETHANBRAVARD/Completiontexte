"""Banc d'essai pour ta fonction d'encodage d'un mot.

Zone verte (tooling) : vérification pure. N'encode rien — il appelle TA fonction
et compare son résultat à la référence.

La référence n'est pas écrite à la main : elle est obtenue en faisant tourner
`encodeur.encode` dans sa version committée (celle qui a produit tes 126 M de
tokens), puis en redécoupant sa sortie mot par mot. Ce que ce script vérifie est
donc exactement la propriété qui compte :

    encoder un mot seul  ==  la part de ce mot dans l'encodage du texte entier

Usage :
    python3 src/tooling/essayer_tokkenisation.py
    python3 src/tooling/essayer_tokkenisation.py --fonction tokkenisation
    python3 src/tooling/essayer_tokkenisation.py --texte "Once upon a time, the house."
    python3 src/tooling/essayer_tokkenisation.py --mots " the" "e," " hous"
    python3 src/tooling/essayer_tokkenisation.py --corpus data/stories.val.txt --max-ko 300

Ton `data/encode.json` n'est jamais touché : la référence tourne dans un bac à sable.
"""

import argparse
import inspect
import json
import subprocess
import sys
import tempfile
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
FICHIER = "src/model/encodeur.py"
BPE = RACINE / "data" / "tokenizer" / "bpe.json"

TEXTE_DEFAUT = ("Once upon a time, there was a little girl who lived in a small "
                "house near the woods. She was happy.")

LANCEUR = ("import sys; sys.path.insert(0, '.'); import encodeur; "
           "encodeur.encode(sys.argv[1], sys.argv[2])")


def reference(texte, ref_git):
    """Encode le texte avec la version committée de l'encodeur. Rend la liste de tokens."""
    git = subprocess.run(["git", "show", f"{ref_git}:{FICHIER}"],
                         cwd=RACINE, capture_output=True, text=True)
    if git.returncode != 0:
        raise SystemExit(f"impossible de lire {FICHIER} à la version {ref_git}")

    with tempfile.TemporaryDirectory(prefix="essayer_tok_") as tmp:
        bac = Path(tmp)
        (bac / "encodeur.py").write_text(git.stdout, encoding="utf-8")
        (bac / "data").mkdir()
        (bac / "t.txt").write_text(texte, encoding="utf-8")
        r = subprocess.run([sys.executable, "-c", LANCEUR, "t.txt", str(BPE)],
                           cwd=bac, capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit("la référence a planté :\n" + (r.stderr or r.stdout)[-600:])
        return json.loads((bac / "data" / "encode.json").read_text(encoding="utf-8"))["encode"]


def attendus_par_mot(tokens, mots):
    """Redécoupe la suite de tokens de référence en un groupe par mot."""
    groupes, k = [], 0
    for m in mots:
        g, acc = [], ""
        while acc != m:
            if k >= len(tokens):
                raise SystemExit(f"la référence s'arrête au milieu du mot {m!r}")
            if not m.startswith(acc + tokens[k]):
                raise SystemExit(
                    f"un token de référence chevauche une frontière de mot :\n"
                    f"  mot {m!r}, déjà couvert {acc!r}, token suivant {tokens[k]!r}\n"
                    f"  -> l'encodage mot par mot ne peut pas être équivalent.")
            acc += tokens[k]; g.append(tokens[k]); k += 1
        groupes.append(g)
    return groupes


def appeler(f, mot, bpe):
    """Appelle ta fonction, quelle que soit sa signature. Rend (resultat, erreur)."""
    alpha = set(bpe["alphabet"])
    for args in ((mot,), (mot, bpe), (mot, bpe, alpha), (mot, bpe["fusions"]),
                 (mot, bpe["fusions"], alpha)):
        try:
            return f(*args), None
        except TypeError as e:
            if "positional argument" not in str(e) and "argument" not in str(e):
                return None, f"{type(e).__name__}: {e}"
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}"
    return None, "aucune signature essayée ne convient (mot / mot+bpe / mot+fusions)"


def main(argv=None):
    p = argparse.ArgumentParser(description="Banc d'essai de la tokenisation d'un mot.")
    p.add_argument("--fonction", default="tokkenisation")
    p.add_argument("--texte", default=None)
    p.add_argument("--corpus", default=None,
                   help="lire le texte dans ce fichier au lieu de --texte")
    p.add_argument("--max-ko", type=float, default=0,
                   help="ne garder que les N premiers kilooctets du corpus")
    p.add_argument("--mots", nargs="*", default=None,
                   help="tester ces mots précis au lieu du texte (donne-les avec leur espace)")
    p.add_argument("--ref", default="HEAD")
    args = p.parse_args(argv)

    if args.corpus:
        texte = (RACINE / args.corpus).read_text(encoding="utf-8")
        if args.max_ko:
            texte = texte[:int(args.max_ko * 1000)]
    else:
        texte = args.texte or TEXTE_DEFAUT
    args.texte = texte

    sys.path.insert(0, str(RACINE / "src" / "model"))
    try:
        import encodeur
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"encodeur.py ne s'importe pas — {type(e).__name__}: {e}")

    if not hasattr(encodeur, args.fonction):
        dispo = [n for n in dir(encodeur) if callable(getattr(encodeur, n)) and not n.startswith("_")]
        raise SystemExit(f"pas de fonction {args.fonction!r}. Trouvées : {', '.join(dispo)}")
    f = getattr(encodeur, args.fonction)
    bpe = json.loads(BPE.read_text(encoding="utf-8"))

    print(f"\n  fonction  : {args.fonction}{inspect.signature(f)}")
    print(f"  référence : {FICHIER} à la version {args.ref}")

    toks = reference(args.texte, args.ref)
    mots = encodeur.decouper(args.texte)
    # L'encodeur de référence préfixe le corpus d'un espace : il revient au 1er mot.
    mots_ref = [" " + mots[0]] + mots[1:]
    groupes = attendus_par_mot(toks, mots_ref)

    # On teste les mots qui portent déjà leur espace, donc pas le premier.
    a_tester = args.mots if args.mots else list(dict.fromkeys(mots[1:]))
    table = {m: g for m, g in zip(mots_ref[1:], groupes[1:])}

    print(f"  {len(a_tester)} mot(s) distinct(s) à tester\n")
    largeur = max((len(repr(m)) for m in a_tester), default=8)
    bons = 0
    for m in a_tester:
        attendu = table.get(m)
        obtenu, err = appeler(f, m, bpe)
        if err:
            etat, detail = "ERREUR", err
        elif attendu is None:
            etat, detail = "?", f"absent du texte de référence, obtenu {obtenu!r}"
        elif not isinstance(obtenu, list):
            etat, detail = "ÉCHEC", (f"rend un {type(obtenu).__name__} {obtenu!r}, "
                                     f"on attend la liste {attendu}")
        elif obtenu != attendu:
            etat, detail = "ÉCHEC", f"rend {obtenu}, attendu {attendu}"
        else:
            etat, detail = "OK", f"{obtenu}"
            bons += 1
        print(f"    {m!r:{largeur}}  {etat:6}  {detail}")

    testables = [m for m in a_tester if m in table]
    print(f"\n  {bons}/{len(testables)} corrects"
          f"{' — tout passe' if testables and bons == len(testables) else ''}\n")
    return 0 if testables and bons == len(testables) else 1


if __name__ == "__main__":
    raise SystemExit(main())
