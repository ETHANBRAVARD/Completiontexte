"""Orchestration de la chaîne de tokenisation BPE, et contrôle de ses invariants.

Zone verte (tooling) : ce fichier n'implémente **aucune** logique de tokenisation.
Il se contente d'appeler les fonctions de `src/model/` (bpe_stories, bpe_liste,
encodeur, decodeur), de les chronométrer, d'archiver leurs sorties dans `runs/`,
et de vérifier après coup des propriétés que le résultat doit satisfaire.

Les chemins de sortie sont codés en dur dans `src/model/` (zone rouge) ; ce script
ne les modifie pas, il les lit là où ils sont écrits.

Usage :
    python scripts/tokenizer.py                              # corpus d'entraînement complet
    python scripts/tokenizer.py --corpus data/stories.val.txt # essai rapide (~1 Mo)
    python scripts/tokenizer.py --etapes bpe,liste            # s'arrêter avant l'encodage
    python scripts/tokenizer.py --etapes verif                # re-vérifier des artefacts existants

Les artefacts existants sont sauvegardés dans le dossier du run avant écrasement.
"""

import argparse
import bisect
import collections
import contextlib
import inspect
import io
import json
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "model"))

# Chemins de sortie tels que src/model/ les écrit. Ne pas « corriger » ici :
# si ces constantes divergent du code modèle, c'est le code modèle qui fait foi.
BPE_JSON = ROOT / "data" / "tokenizer" / "bpe.json"
LISTE_JSON = ROOT / "data" / "tokenizer" / "bpe_liste.json"
ENCODE_JSON = ROOT / "data" / "encode.json"

# Références mesurées le 11/08/2026 sur stories.train.txt, avant correction de la
# pré-tokenisation (mots préfixés d'un espace, premier mot de ligne déchiqueté).
REF = {
    "ratio": 3.4067,
    "tokens": 5_846_939,
    "gaspilles_debut_ligne": 285_885,
    "tokens_par_mot_debut_ligne": 2.16,
}

ETAPES_CONNUES = ("bpe", "liste", "encode", "splits", "verif")

# encodeur.encode() écrit toujours dans data/encode.json (chemin en dur, zone rouge).
# Pour encoder val et test on l'appelle puis on déplace sa sortie — sans toucher
# au module.
SPLITS = {"val": ROOT / "data" / "stories.val.txt",
          "test": ROOT / "data" / "stories.test.txt"}


class Journal:
    """Écrit simultanément sur la sortie standard et dans le log du run."""

    def __init__(self, chemin: Path):
        self.fichier = chemin.open("w", encoding="utf-8")

    def __call__(self, ligne: str = "") -> None:
        print(ligne)
        self.fichier.write(ligne + "\n")
        self.fichier.flush()

    def close(self) -> None:
        self.fichier.close()


def duree(secondes: float) -> str:
    if secondes < 60:
        return f"{secondes:.1f} s"
    return f"{int(secondes // 60)} min {secondes % 60:.0f} s"


def archiver(chemins, destination: Path, log) -> None:
    """Copie les artefacts existants dans le dossier du run avant écrasement."""
    sauve = destination / "avant"
    for c in chemins:
        if c.exists():
            sauve.mkdir(parents=True, exist_ok=True)
            shutil.copy2(c, sauve / c.name)
            log(f"  sauvegarde : {c.name} ({c.stat().st_size / 1e6:.1f} Mo) → {sauve.relative_to(ROOT)}/")


# --------------------------------------------------------------------------- étapes


def etape_bpe(corpus: Path, log, fusions: int | None) -> float:
    import bpe_stories

    # Le nombre de fusions est un choix de modélisation : il vit dans src/model/.
    # On le passe si la fonction l'accepte, sinon on refuse plutôt que de laisser
    # croire qu'un budget différent a été appliqué.
    params = [p for p in inspect.signature(bpe_stories.bpe).parameters]
    accepte = len(params) >= 2
    if fusions is not None and not accepte:
        raise RuntimeError(
            f"--fusions {fusions} demandé, mais bpe_stories.bpe{inspect.signature(bpe_stories.bpe)} "
            f"ne prend pas ce paramètre : le compte est codé en dur dans la fonction. "
            f"Ajoute-le en argument avant de relancer."
        )

    budget = f"{fusions} fusions" if fusions is not None else "budget de fusions par défaut"
    log(f"\n[1/3] apprentissage des fusions sur {corpus.relative_to(ROOT)} "
        f"({corpus.stat().st_size / 1e6:.1f} Mo) — {budget}")
    BPE_JSON.parent.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    if fusions is not None:
        bpe_stories.bpe(str(corpus), fusions)
    else:
        bpe_stories.bpe(str(corpus))
    dt = time.perf_counter() - t
    log(f"      terminé en {duree(dt)} → {BPE_JSON.relative_to(ROOT)}")
    return dt


def etape_liste(log) -> float:
    import bpe_liste

    log("\n[2/3] dérivation du vocabulaire")
    t = time.perf_counter()
    bpe_liste.bpelist(str(BPE_JSON))
    dt = time.perf_counter() - t
    log(f"      terminé en {duree(dt)} → {LISTE_JSON.relative_to(ROOT)}")
    return dt


def etape_encode(corpus: Path, log) -> float:
    import encodeur

    log(f"\n[3/3] encodage de {corpus.relative_to(ROOT)}")
    log("      (l'encodeur applique les fusions une par une sur tout le corpus : c'est le pas le plus long)")
    t = time.perf_counter()
    try:
        encodeur.encode(str(corpus), str(BPE_JSON))
    except SystemExit:
        # encodeur.py appelle exit() quand il rencontre un caractère hors vocabulaire
        # ou un séparateur en collision. On ne laisse pas ça tuer le rapport.
        log("      ARRÊT : encodeur.py a appelé exit() — voir le message ci-dessus.")
        raise RuntimeError("encodage interrompu par encodeur.py")
    dt = time.perf_counter() - t
    log(f"      terminé en {duree(dt)} → {ENCODE_JSON.relative_to(ROOT)}")
    return dt


def etape_splits(log) -> float:
    """Encode val et test avec le vocabulaire déjà gelé, sans réapprendre les fusions."""
    import encodeur

    log("\n[+] encodage de val et test avec le vocabulaire gelé")
    garde = ENCODE_JSON.with_suffix(".train.json")
    if ENCODE_JSON.exists():
        shutil.move(ENCODE_JSON, garde)
        log(f"      {ENCODE_JSON.name} (train) mis de côté sous {garde.name}")

    t = time.perf_counter()
    for nom, chemin in SPLITS.items():
        if not chemin.exists():
            log(f"      {chemin.name} absent, ignoré")
            continue
        d = time.perf_counter()
        try:
            encodeur.encode(str(chemin), str(BPE_JSON))
        except SystemExit:
            raise RuntimeError(f"encodeur.py a appelé exit() sur {chemin.name}")
        cible = ENCODE_JSON.with_suffix(f".{nom}.json")
        shutil.move(ENCODE_JSON, cible)
        n = len(json.loads(cible.read_text(encoding="utf-8"))["encode"])
        texte = chemin.read_text(encoding="utf-8")
        log(f"      {nom:5s} : {len(texte):>10,} car → {n:>9,} tokens  "
            f"(ratio {len(texte)/n:.4f})  en {duree(time.perf_counter()-d)}"
            .replace(",", " "))

    if garde.exists():
        shutil.move(garde, ENCODE_JSON)
    return time.perf_counter() - t


# ---------------------------------------------------------------------- vérifications


def spans_tokens(tokens):
    """Décalage de départ de chaque token dans le texte reconstruit."""
    debuts = []
    pos = 0
    for tok in tokens:
        debuts.append(pos)
        pos += len(tok)
    return debuts, pos


def tokens_par_mot(texte: str, tokens, debuts, decalage: int):
    """Nombre de tokens chevauchant chaque mot, séparément pour les débuts de ligne.

    `decalage` = nombre de caractères que la reconstruction ajoute avant le texte
    (l'encodeur préfixe une unité d'espace).
    """
    en_debut = collections.Counter()
    ailleurs = collections.Counter()
    for m in re.finditer(r"\S+", texte):
        a = m.start() + decalage
        b = m.end() + decalage
        premier = bisect.bisect_right(debuts, a) - 1
        dernier = bisect.bisect_left(debuts, b) - 1
        n = dernier - premier + 1
        # un mot est en début de ligne s'il ouvre le texte ou suit un saut de ligne
        debut_de_ligne = m.start() == 0 or texte[m.start() - 1] == "\n"
        (en_debut if debut_de_ligne else ailleurs)[n] += 1
    return en_debut, ailleurs


def verifier(corpus: Path, log, budget: int | None = None) -> bool:
    log("\n" + "=" * 72)
    log("VÉRIFICATIONS")
    log("=" * 72)

    manquants = [p for p in (BPE_JSON, LISTE_JSON, ENCODE_JSON) if not p.exists()]
    if manquants:
        for p in manquants:
            log(f"  ABSENT : {p.relative_to(ROOT)}")
        return False

    bpe = json.loads(BPE_JSON.read_text(encoding="utf-8"))
    vocab = json.loads(LISTE_JSON.read_text(encoding="utf-8"))
    tokens = json.loads(ENCODE_JSON.read_text(encoding="utf-8"))["encode"]
    texte = corpus.read_text(encoding="utf-8")

    ok = True

    # 1. les fusions ont-elles toutes été apprises ?
    log("\n1. Fusions")
    n_fus = len(bpe["fusions"])
    log(f"   alphabet         : {len(bpe['alphabet'])} caractères")
    log(f"   fusions apprises : {n_fus}" + (f"   (demandées : {budget})" if budget else ""))
    if budget is not None and n_fus < budget:
        log("   ÉCHEC — moins que le budget demandé. Les paires se sont épuisées avant :")
        log("           la pré-tokenisation produit des unités trop courtes.")
        ok = False
    elif budget is None and n_fus < 1000:
        log("   ÉCHEC — moins de 1000, les paires se sont épuisées avant.")
        ok = False
    else:
        log("   OK")
    saut_dans_alphabet = "\n" in bpe["alphabet"]
    log(f"   saut de ligne dans l'alphabet : {saut_dans_alphabet}")
    if not saut_dans_alphabet:
        log("   ÉCHEC — le saut de ligne n'est jamais entré dans une unité.")
        ok = False

    # 2. le vocabulaire est-il sans doublon ?
    log("\n2. Vocabulaire")
    log(f"   taille : {len(vocab)}")
    doublons = [t for t, c in collections.Counter(vocab).items() if c > 1]
    if doublons:
        log(f"   ÉCHEC — {len(doublons)} doublons, donc deux indices pour un même token :")
        log(f"           {doublons[:10]}")
        ok = False
    else:
        log("   OK — aucun doublon")

    # 3. tout token produit existe-t-il dans le vocabulaire ?
    log("\n3. Couverture")
    inconnus = set(tokens) - set(vocab)
    log(f"   tokens produits : {len(tokens):,}".replace(",", " "))
    if inconnus:
        log(f"   ÉCHEC — {len(inconnus)} tokens produits absents du vocabulaire :")
        log(f"           {sorted(inconnus)[:10]}")
        ok = False
    else:
        log("   OK — tous présents")

    # 4. aller-retour exact
    log("\n4. Aller-retour")
    recolle = "".join(tokens)
    decalage = 0
    if recolle == texte:
        log("   OK — la concaténation des tokens redonne le texte exactement")
    elif recolle[1:] == texte:
        decalage = 1
        log(f"   OK — au préfixe {recolle[0]!r} près, ajouté par encodeur.py "
            f"et retiré par decodeur.py")
    else:
        ok = False
        log(f"   ÉCHEC — {len(recolle)} caractères reconstruits contre {len(texte)} attendus")
        n = min(len(recolle), len(texte))
        i = next((k for k in range(n) if recolle[k] != texte[k]), n)
        log(f"           première divergence au caractère {i}")
        log(f"           reconstruit : {recolle[max(0, i - 30):i + 30]!r}")
        log(f"           texte       : {texte[max(0, i - 30):i + 30]!r}")

    # 4 bis. le décodeur d'Ethan fait-il la même chose ?
    try:
        import decodeur

        tampon = io.StringIO()
        with contextlib.redirect_stdout(tampon):
            decodeur.decode(str(ENCODE_JSON))
        sortie = tampon.getvalue()
        # print() ajoute un saut de ligne final
        sortie = sortie[:-1] if sortie.endswith("\n") else sortie
        log(f"   decodeur.py : {'OK — identique au texte' if sortie == texte else 'ÉCART avec le texte'}")
        if sortie != texte:
            log(f"                 {len(sortie)} caractères contre {len(texte)}")
            ok = False
    except Exception as e:  # noqa: BLE001
        log(f"   decodeur.py : non exécutable ({type(e).__name__}: {e})")

    # 5. compression
    log("\n5. Compression")
    ratio = len(texte) / len(tokens)
    log(f"   {len(texte):,} caractères → {len(tokens):,} tokens".replace(",", " "))
    log(f"   ratio caractères/token : {ratio:.4f}   (référence avant correction : {REF['ratio']:.4f})")
    if corpus.name == "stories.train.txt":
        ecart = 100 * (ratio - REF["ratio"]) / REF["ratio"]
        log(f"   variation : {ecart:+.2f} %")
        gagnes = REF["tokens"] - len(tokens)
        log(f"   tokens économisés : {gagnes:+,}".replace(",", " "))

    # 6. les débuts de ligne
    log("\n6. Débuts de ligne")
    debuts, total = spans_tokens(tokens)
    if total != len(recolle):
        log("   (calcul de position impossible, aller-retour incohérent)")
    else:
        en_debut, ailleurs = tokens_par_mot(texte, tokens, debuts, decalage)
        n_deb = sum(en_debut.values())
        n_ail = sum(ailleurs.values())
        moy_deb = sum(k * v for k, v in en_debut.items()) / max(n_deb, 1)
        moy_ail = sum(k * v for k, v in ailleurs.items()) / max(n_ail, 1)
        log(f"   mots en début de ligne : {n_deb:,}".replace(",", " "))
        log(f"     tokens par mot : {moy_deb:.2f}   (référence avant correction : "
            f"{REF['tokens_par_mot_debut_ligne']:.2f})")
        log(f"   mots ailleurs          : {n_ail:,}".replace(",", " "))
        log(f"     tokens par mot : {moy_ail:.2f}")
        ecart_rel = moy_deb / moy_ail if moy_ail else float("inf")
        log(f"   rapport début/ailleurs : {ecart_rel:.2f}"
            f"   (1,00 = les débuts de ligne ne coûtent plus rien de spécial)")
        if ecart_rel > 1.5:
            log("   ATTENTION — les mots en début de ligne coûtent encore nettement")
            log("               plus cher que les autres.")

    # 7. échantillon autour d'un saut de ligne
    log("\n7. Échantillon autour des sauts de ligne")
    positions = [i for i, t in enumerate(tokens[:400_000]) if "\n" in t]
    if not positions:
        log("   aucun token contenant un saut de ligne dans les 400 000 premiers")
    else:
        for i in positions[:4]:
            log(f"   {[t for t in tokens[max(0, i - 3):i + 4]]}")
        colles = {t for t in set(tokens) if "\n" in t and t != "\n"}
        log(f"\n   tokens contenant un saut de ligne autres que '\\n' : {len(colles)}")
        if colles:
            log(f"   exemples : {sorted(colles)[:6]}")
            log("   (attendu : 0 si le saut de ligne est bien une unité isolée ;")
            log("    un token de paragraphe '\\n\\n' n'apparaîtra que si tu lis le")
            log("    corpus d'un seul tenant au lieu de boucler sur les lignes)")

    log("\n" + "=" * 72)
    log("RÉSULTAT : " + ("tout est cohérent" if ok else "au moins un contrôle a échoué"))
    log("=" * 72)
    return ok


# ------------------------------------------------------------------------------ main


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", type=Path, default=ROOT / "data" / "stories.train.txt",
                   help="corpus à apprendre et à encoder")
    p.add_argument("--etapes", default="bpe,liste,encode,verif",
                   help=f"étapes à exécuter, séparées par des virgules ({'|'.join(ETAPES_CONNUES)})")
    p.add_argument("--nom", default="tokenizer", help="suffixe du dossier de run")
    p.add_argument("--fusions", type=int, default=None,
                   help="budget de fusions à passer à bpe_stories.bpe (défaut : celui de la fonction)")
    args = p.parse_args()

    etapes = [e.strip() for e in args.etapes.split(",") if e.strip()]
    inconnues = [e for e in etapes if e not in ETAPES_CONNUES]
    if inconnues:
        print(f"étapes inconnues : {inconnues} — connues : {list(ETAPES_CONNUES)}", file=sys.stderr)
        return 2

    corpus = args.corpus if args.corpus.is_absolute() else ROOT / args.corpus
    if not corpus.exists():
        print(f"corpus introuvable : {corpus}", file=sys.stderr)
        return 2

    # Les modules de src/model/ écrivent leurs sorties sur des chemins relatifs
    # ("data/tokenizer/bpe.json", ...). On se place à la racine du dépôt pour que
    # ces chemins désignent le bon endroit quel que soit le répertoire d'appel.
    os.chdir(ROOT)

    run = ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M}-{args.nom}"
    run.mkdir(parents=True, exist_ok=True)
    log = Journal(run / "log.txt")

    log(f"run       : {run.relative_to(ROOT)}")
    log(f"corpus    : {corpus.relative_to(ROOT)}")
    log(f"étapes    : {', '.join(etapes)}")
    log(f"démarré à : {datetime.now():%Y-%m-%d %H:%M:%S}")

    (run / "config.json").write_text(
        json.dumps({"corpus": str(corpus.relative_to(ROOT)), "etapes": etapes,
                    "date": f"{datetime.now():%Y-%m-%d %H:%M:%S}"}, indent=2),
        encoding="utf-8")

    if {"bpe", "liste", "encode"} & set(etapes):
        log("\nartefacts existants :")
        archiver([BPE_JSON, LISTE_JSON, ENCODE_JSON], run, log)

    debut = time.perf_counter()
    code = 0
    try:
        if "bpe" in etapes:
            etape_bpe(corpus, log, args.fusions)
        if "liste" in etapes:
            etape_liste(log)
        if "encode" in etapes:
            etape_encode(corpus, log)
        if "splits" in etapes:
            etape_splits(log)
        if "verif" in etapes:
            if not verifier(corpus, log, args.fusions):
                code = 1
    except RuntimeError as e:
        log(f"\nINTERROMPU : {e}")
        code = 1
    except KeyboardInterrupt:
        log("\nINTERROMPU par l'utilisateur")
        code = 130

    log(f"\ndurée totale : {duree(time.perf_counter() - debut)}")

    for c in (BPE_JSON, LISTE_JSON):
        if c.exists():
            shutil.copy2(c, run / c.name)
    log(f"artefacts archivés dans {run.relative_to(ROOT)}/")
    log.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
