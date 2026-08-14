"""Encodage d'un gros corpus par morceaux, avec le vocabulaire gelé.

Zone verte (tooling) : orchestration et I/O. N'implémente aucune logique de
tokenisation — appelle `encodeur.encode` de src/model/ sur chaque morceau.

Pourquoi par morceaux : `encodeur.py` construit une chaîne Python de 3 caractères
par caractère du corpus, soit ~60 octets de RAM par caractère. Au-delà d'une
centaine de mégaoctets, la machine sature. Le découpage tombe sur des frontières
d'histoire (`\\n\\n`), jamais au milieu d'une.

Deux subtilités traitées ici :

1. `encodeur.encode` préfixe systématiquement sa sortie d'une unité espace
   (`encodeur.py`, ligne 16). Concaténer les morceaux tels quels insérerait un
   espace parasite à chaque jointure. On retire donc ce premier token sur tous
   les morceaux sauf le premier.
2. Le vocabulaire n'est pas réappris : les fusions de `data/tokenizer/bpe.json`
   sont appliquées à l'identique sur chaque morceau. C'est ce qui rend les
   morceaux indépendants et concaténables.

Usage :
    python scripts/encode_gros_corpus.py --mo 500
    python scripts/encode_gros_corpus.py --mo 500 --taille-morceau 85
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src" / "model"))

DATA = ROOT / "data"
BPE_JSON = DATA / "tokenizer" / "bpe.json"
LISTE_JSON = DATA / "tokenizer" / "bpe_liste.json"
ENCODE_JSON = DATA / "encode.json"          # sortie codée en dur dans encodeur.py
MORCEAUX = DATA / "morceaux"
SEP_HISTOIRE = "\n\n"


def duree(s):
    return f"{s:.0f} s" if s < 60 else f"{int(s//60)} min {s%60:.0f} s"


def log(m=""):
    print(m, flush=True)


def decouper(chemin: Path, taille_max: int, dossier: Path):
    """Découpe un fichier en morceaux <= taille_max, sur des frontières d'histoire."""
    dossier.mkdir(parents=True, exist_ok=True)
    for vieux in dossier.glob(f"{chemin.stem}_*.txt"):
        vieux.unlink()

    texte = chemin.read_text(encoding="utf-8")
    morceaux, debut, k = [], 0, 0
    while debut < len(texte):
        fin = min(debut + taille_max, len(texte))
        if fin < len(texte):
            coupe = texte.rfind(SEP_HISTOIRE, debut, fin)
            if coupe <= debut:
                raise RuntimeError(f"aucune frontière d'histoire dans {chemin.name} "
                                   f"entre {debut} et {fin}")
            fin = coupe + len(SEP_HISTOIRE)
        p = dossier / f"{chemin.stem}_{k:02d}.txt"
        p.write_text(texte[debut:fin], encoding="utf-8")
        morceaux.append(p)
        debut, k = fin, k + 1
    del texte
    return morceaux


def encoder_morceaux(morceaux, nom_split: str):
    """Encode chaque morceau dans un processus séparé, et rend les fichiers produits."""
    sorties = []
    for n, m in enumerate(morceaux):
        t = time.perf_counter()
        cible = MORCEAUX / f"{m.stem}.tokens.json"
        # Processus séparé : encodeur.py garde en mémoire une liste de plusieurs
        # gigaoctets ; on veut qu'elle soit rendue au système entre deux morceaux.
        code = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, 'src/model');"
             "import encodeur; encodeur.encode(sys.argv[1], sys.argv[2])",
             str(m), str(BPE_JSON)],
            cwd=ROOT, capture_output=True, text=True)
        if code.returncode != 0 or not ENCODE_JSON.exists():
            log(code.stdout[-500:]); log(code.stderr[-500:])
            raise RuntimeError(f"encodage de {m.name} échoué")
        shutil.move(ENCODE_JSON, cible)
        n_tok = len(json.loads(cible.read_text(encoding="utf-8"))["encode"])
        sorties.append(cible)
        log(f"      {nom_split} {n+1}/{len(morceaux)} : {m.stat().st_size/1e6:6.1f} Mo "
            f"→ {n_tok:>9,} tokens en {duree(time.perf_counter()-t)}".replace(",", " "))
    return sorties


def concatener(sorties, cible: Path, texte_attendu: Path):
    """Recolle les morceaux encodés en un seul fichier de tokens, et vérifie."""
    tokens = []
    for n, s in enumerate(sorties):
        t = json.loads(s.read_text(encoding="utf-8"))["encode"]
        # encodeur.py préfixe chaque sortie d'une unité espace : on ne garde
        # celle du premier morceau (le décodeur la retire déjà avec son [1:]).
        tokens.extend(t if n == 0 else t[1:])
    cible.write_text(json.dumps({"encode": tokens}, ensure_ascii=False), encoding="utf-8")

    texte = texte_attendu.read_text(encoding="utf-8")
    ok = "".join(tokens)[1:] == texte
    log(f"      → {cible.name} : {len(tokens):,} tokens | ratio {len(texte)/len(tokens):.4f} "
        f"| aller-retour {'exact' if ok else 'ROMPU'}".replace(",", " "))
    if not ok:
        raise RuntimeError(f"aller-retour rompu sur {cible.name}")
    return tokens


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mo", type=float, default=500.0, help="mégaoctets de texte à préparer")
    p.add_argument("--taille-morceau", type=float, default=85.0,
                   help="taille max d'un morceau à encoder, en Mo")
    p.add_argument("--sauter-preparation", action="store_true")
    p.add_argument("--splits", default="train,val,test",
                   help="splits à encoder, séparés par des virgules")
    p.add_argument("--sans-archivage", action="store_true",
                   help="ne pas déplacer l'état courant (reprise d'un split isolé)")
    args = p.parse_args()

    import os
    os.chdir(ROOT)

    run = ROOT / "runs" / f"{datetime.now():%Y%m%d-%H%M}-corpus{int(args.mo)}mo"
    run.mkdir(parents=True, exist_ok=True)
    debut = time.perf_counter()
    log(f"run    : {run.relative_to(ROOT)}")
    log(f"cible  : {args.mo:.0f} Mo, morceaux de {args.taille_morceau:.0f} Mo")

    # 1. archivage de l'état courant
    if args.sans_archivage:
        log("\n[1/4] archivage sauté (--sans-archivage)")
    else:
        avant = run / "avant"
        avant.mkdir(exist_ok=True)
        for f in list(DATA.glob("stories.*.txt")) + list(DATA.glob("encode*.json")) \
                + list(DATA.glob("encode_tok_*.npy")):
            shutil.move(f, avant / f.name)
        log(f"\n[1/4] état précédent archivé dans {avant.relative_to(ROOT)}/ "
            f"({len(list(avant.iterdir()))} fichiers)")

    # 2. préparation
    if not args.sauter_preparation:
        log(f"\n[2/4] préparation de {args.mo:.0f} Mo")
        r = subprocess.run([sys.executable, "scripts/prepare_stories.py",
                            "--max-mo", str(args.mo)], cwd=ROOT, capture_output=True, text=True)
        log("      " + "\n      ".join(r.stdout.strip().splitlines()[-6:]))
        if r.returncode != 0:
            log(r.stderr[-800:]); return 1

    # 3. découpage et encodage
    log(f"\n[3/4] encodage par morceaux avec le vocabulaire gelé "
        f"({len(json.loads(LISTE_JSON.read_text(encoding='utf-8')))} tokens)")
    MORCEAUX.mkdir(parents=True, exist_ok=True)
    total = 0
    finaux = {}
    for split in args.splits.split(","):
        src = DATA / f"stories.{split}.txt"
        morceaux = decouper(src, int(args.taille_morceau * 1e6), MORCEAUX)
        log(f"    {split} : {src.stat().st_size/1e6:.1f} Mo en {len(morceaux)} morceau(x)")
        sorties = encoder_morceaux(morceaux, split)
        # Étape intermédiaire obligatoire : encodeur.py écrit toujours dans
        # data/encode.json, qui est aussi la destination finale de train. Écrire
        # directement là ferait écraser train par le premier morceau du split
        # suivant. On assemble donc à l'écart, et on ne déplace qu'à la toute fin.
        provisoire = MORCEAUX / f"{split}.assemble.json"
        total += len(concatener(sorties, provisoire, src))
        finaux[provisoire] = DATA / ("encode.json" if split == "train"
                                     else f"encode.{split}.json")
        for f in morceaux + sorties:
            f.unlink()

    for provisoire, cible in finaux.items():
        shutil.move(provisoire, cible)
        log(f"    → {cible.relative_to(ROOT)}")

    log(f"\n[4/4] terminé : {total:,} tokens au total".replace(",", " "))
    log(f"durée : {duree(time.perf_counter()-debut)}")
    log(f"\nÉtape suivante : la conversion token → entier (rencode.py) sur les trois "
        f"fichiers data/encode{{,.val,.test}}.json")
    shutil.rmtree(MORCEAUX, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())