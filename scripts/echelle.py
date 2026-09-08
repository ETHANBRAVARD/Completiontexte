"""Banc d'échelle : plusieurs tailles de modèle, à budget de pas égal.

Zone verte (tooling) : orchestration, journalisation, tracé. N'implémente aucune
logique de modèle — il fait tourner `src/model/transformer.py`, tel qu'Ethan l'a
écrit, avec des hyperparamètres substitués.

**`src/model/transformer.py` n'est jamais modifié.** Pour chaque configuration, le
script en dépose une copie dans le dossier du run, avec les lignes d'affectation
remplacées, et exécute cette copie. Le run porte donc sa propre source : il est
rejouable à l'identique même si le fichier d'origine change ensuite.

Chaque substitution doit correspondre **exactement une fois**, sinon le script
s'arrête. Sans ça, un renommage dans transformer.py ferait tourner tout le banc
sur les hyperparamètres par défaut, sans que rien ne le signale.

Le banc se découpe en trois parties indépendantes, à lancer quand on veut. En
donnant le même `--dossier`, les résultats se cumulent : la courbe se complète au
fur et à mesure, et une configuration déjà mesurée n'est pas relancée.

Usage :
    python3 scripts/echelle.py --calibrer --partie 1     # 60 pas, mesure s/pas
    python3 scripts/echelle.py --partie 1 --pas 1500     # ouvre un dossier de banc
    python3 scripts/echelle.py --partie 2 --dossier runs/echelle-20260908-2010
    python3 scripts/echelle.py --partie 3 --dossier runs/echelle-20260908-2010
"""

import argparse
import json
import re
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELE = ROOT / "src" / "model" / "transformer.py"
ALPH = 2080          # taille du vocabulaire, data/tokenizer/bpe_liste.json
MAX_LEN = 384        # contexte, inchangé sur tout le banc
LOT = 32


def parametres(dim, blocs, alph=ALPH, max_len=MAX_LEN):
    """Nombre de paramètres, dérivé de la structure de transformer.py.

    Embeddings de tokens et de positions, 12·dim² + 9·dim par bloc (attention,
    MLP 4x, deux layernorms), layernorm final, projection de sortie.
    Contrôle : dim=384, blocs=6 -> 12 385 312, soit les 12,39 M du README.
    """
    return (2 * alph * dim + max_len * dim
            + blocs * (12 * dim * dim + 9 * dim)
            + 2 * dim + alph)


# --- le banc ------------------------------------------------------------------
# Axe A : la taille, à forme comparable. Axe B : profondeur contre largeur, à
# budget de paramètres voisin — ce que l'expérience d'août ne séparait pas.
BANC = [
    ("A1-petit",     256,  4, 4),
    ("A2-actuel",    384,  6, 4),
    ("A3-moyen",     512,  7, 8),
    ("A4-grand",     640,  8, 8),
    ("B1-large",     640,  5, 8),
    ("B2-profond",   448, 11, 8),
]


# Trois parties indépendantes, à lancer quand on veut. Chacune se verse dans le
# même dossier de banc, donc les résultats se cumulent et la courbe se complète.
PARTIES = {
    1: (["A1-petit", "A2-actuel"],
        "la pente à bas coût — deux points, le second est ton modèle d'août"),
    2: (["A3-moyen", "A4-grand"],
        "le haut de la gamme — complète l'axe de taille"),
    3: (["B1-large", "B2-profond"],
        "profondeur contre largeur, à budget de paramètres voisin"),
}


def source_parametree(dim, heads, blocs, pas_max, dossier, val_tous, echantillons):
    """Rend le source de transformer.py avec les hyperparamètres substitués."""
    src = MODELE.read_text(encoding="utf-8")
    remplacements = [
        (r"^dim=\d+$",                      f"dim={dim}"),
        (r"^num_heads=\d+$",                f"num_heads={heads}"),
        (r"^num_blocs=\d+$",                f"num_blocs={blocs}"),
        (r"^for i in range\(\d+\):$",       f"for i in range({pas_max}):"),
        (r"^chemin_dossier = .*$",          f"chemin_dossier = r'{dossier}'"),
        (r"^    if i%1500==0:$",            f"    if i%{val_tous}==0:"),
        (r"^for _ in range\(20\):$",        f"for _ in range({echantillons}):"),
    ]
    for motif, remplacement in remplacements:
        src, n = re.subn(motif, remplacement, src, flags=re.MULTILINE)
        if n != 1:
            raise SystemExit(
                f"substitution « {motif} » trouvée {n} fois au lieu d'une dans "
                f"{MODELE.name}.\nLe banc s'arrête : sans cette ligne, le run "
                f"tournerait sur les valeurs par défaut sans le dire.")
    return src


def lire_pertes(sortie: str):
    """Extrait les pertes d'entraînement et de validation du journal du run.

    Les deux impressions ont le même format ; seule la ligne « ### Loss sur le jeu
    de validation ### » distingue celle qui suit.
    """
    train, val = [], []
    suivante_est_val = False
    for ligne in sortie.splitlines():
        if "jeu de validation" in ligne:
            suivante_est_val = True
            continue
        m = re.match(r"i=(\d+), loss=([\d.]+)", ligne.strip())
        if m:
            point = (int(m.group(1)), float(m.group(2)))
            (val if suivante_est_val else train).append(point)
            suivante_est_val = False
    return train, val


def lancer(nom, dim, blocs, heads, pas_max, racine, val_tous, echantillons, log):
    dossier = racine / nom
    dossier.mkdir(parents=True, exist_ok=True)
    source = dossier / "source.py"
    source.write_text(
        source_parametree(dim, heads, blocs, pas_max, str(dossier),
                          val_tous, echantillons), encoding="utf-8")

    n_par = parametres(dim, blocs)
    log(f"  {nom:12} dim={dim:4} blocs={blocs:3} têtes={heads:2}  "
        f"{n_par/1e6:6.2f} M paramètres  {pas_max} pas", fin="")
    t = time.perf_counter()
    # La copie tourne depuis le dossier du run : src/model/ doit rester importable
    # (tireur_de_lot), sans quoi le module n'est pas trouvé.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src" / "model") + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-u", str(source)], cwd=ROOT,
                       capture_output=True, text=True, env=env)
    duree = time.perf_counter() - t
    (dossier / "log.txt").write_text(r.stdout + "\n=== stderr ===\n" + r.stderr,
                                     encoding="utf-8")

    if r.returncode != 0:
        derniere = (r.stderr.strip().splitlines() or ["(pas de message)"])[-1]
        log(f"   ÉCHEC en {duree:.0f} s — {derniere[:80]}")
        return {"nom": nom, "dim": dim, "blocs": blocs, "heads": heads,
                "parametres": n_par, "pas": pas_max, "duree_s": duree,
                "echec": derniere}

    train, val = lire_pertes(r.stdout)
    par_pas = duree / pas_max
    log(f"   {duree/60:5.1f} min  {par_pas*1000:5.0f} ms/pas"
        + (f"  val {val[-1][1]:.4f}" if val else "  (aucune mesure de validation)"))

    # On ne garde que le dernier checkpoint : les intermédiaires pèsent lourd et
    # ne servent à rien une fois la courbe tracée.
    # Tri numérique : "sauvegarde_935" passe après "sauvegarde_1496" en ordre
    # lexicographique, ce qui ferait supprimer le checkpoint final.
    points = sorted(dossier.glob("*-sauvegarde_*.pt"),
                    key=lambda f: int(f.stem.rsplit("_", 1)[1]))
    for vieux in points[:-1]:
        vieux.unlink()

    return {"nom": nom, "dim": dim, "blocs": blocs, "heads": heads,
            "parametres": n_par, "pas": pas_max, "duree_s": duree,
            "s_par_pas": par_pas, "tokens_vus": LOT * MAX_LEN * pas_max,
            "train": train, "val": val,
            "val_finale": val[-1][1] if val else None,
            "train_finale": train[-1][1] if train else None}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--partie", type=int, choices=sorted(PARTIES),
                   help="lancer une partie du banc (1, 2 ou 3)")
    p.add_argument("--configs", default="",
                   help="noms séparés par des virgules (prioritaire sur --partie)")
    p.add_argument("--dossier", default="",
                   help="dossier de banc existant : les résultats s'y ajoutent")
    p.add_argument("--pas", type=int, default=2000, help="pas par configuration")
    p.add_argument("--calibrer", action="store_true",
                   help="60 pas par configuration, pour mesurer le temps par pas")
    p.add_argument("--val-tous", type=int, default=0,
                   help="mesurer la validation tous les N pas (0 = pas/8)")
    p.add_argument("--refaire", action="store_true",
                   help="relancer une configuration déjà présente dans le dossier")
    args = p.parse_args()

    if args.configs:
        voulus = [c.strip() for c in args.configs.split(",")]
    elif args.partie:
        voulus = PARTIES[args.partie][0]
    else:
        voulus = [c[0] for c in BANC]
    banc = [c for c in BANC if c[0] in voulus]
    if not banc:
        raise SystemExit(f"aucune config parmi {[c[0] for c in BANC]}")

    pas = 60 if args.calibrer else args.pas
    val_tous = args.val_tous or max(1, pas // 8)
    echantillons = 0 if args.calibrer else 2

    # Un dossier donné = on complète un banc existant ; sinon on en ouvre un.
    if args.dossier:
        racine = Path(args.dossier)
        if not racine.is_absolute():
            racine = ROOT / racine
        if not racine.is_dir():
            raise SystemExit(f"dossier de banc introuvable : {racine}")
    else:
        racine = ROOT / "runs" / f"echelle-{datetime.now():%Y%m%d-%H%M}"
        racine.mkdir(parents=True, exist_ok=True)

    fichier = racine / "resultats.json"
    anciens = {}
    if fichier.is_file():
        for r in json.loads(fichier.read_text(encoding="utf-8")).get("runs", []):
            anciens[r["nom"]] = r

    journal = racine / "banc.txt"

    def log(m="", fin="\n"):
        print(m, end=fin, flush=True)
        with journal.open("a", encoding="utf-8") as f:
            f.write(m + fin)

    log(f"banc   : {racine.relative_to(ROOT)}")
    if args.partie:
        log(f"partie {args.partie} : {PARTIES[args.partie][1]}")
    log(f"mode   : {'calibration' if args.calibrer else 'mesure'}, {pas} pas, "
        f"validation tous les {val_tous}")
    log(f"corpus : {LOT}x{MAX_LEN} = {LOT*MAX_LEN} tokens par pas, "
        f"{LOT*MAX_LEN*pas/1e6:.1f} M tokens par config")
    if anciens:
        log(f"déjà présent : {', '.join(sorted(anciens))}")
    log()

    debut = time.perf_counter()
    for nom, dim, blocs, heads in banc:
        if nom in anciens and not args.refaire:
            log(f"  {nom:12} déjà mesuré, ignoré (--refaire pour le relancer)")
            continue
        anciens[nom] = lancer(nom, dim, blocs, heads, pas, racine,
                              val_tous, echantillons, log)
        ordre = [c[0] for c in BANC]
        runs = sorted(anciens.values(), key=lambda r: ordre.index(r["nom"]))
        fichier.write_text(json.dumps(
            {"pas": pas, "lot": LOT, "max_len": MAX_LEN, "vocabulaire": ALPH,
             "runs": runs}, ensure_ascii=False, indent=1), encoding="utf-8")

    log(f"\ndurée de cette partie : {(time.perf_counter()-debut)/60:.1f} min")

    ordre = [c[0] for c in BANC]
    runs = sorted(anciens.values(), key=lambda r: ordre.index(r["nom"]))
    log(f"\n{'config':13}{'paramètres':>12}{'ms/pas':>9}{'perte val':>11}")
    for r in runs:
        if "echec" in r:
            log(f"{r['nom']:13}{r['parametres']/1e6:10.2f} M      —  ÉCHEC")
        else:
            v = f"{r['val_finale']:.4f}" if r.get("val_finale") is not None else "—"
            log(f"{r['nom']:13}{r['parametres']/1e6:10.2f} M"
                f"{r['s_par_pas']*1000:9.0f}{v:>11}")

    manque = [c[0] for c in BANC if c[0] not in anciens]
    if manque:
        log(f"\nreste à mesurer : {', '.join(manque)}")
        log(f"  python3 scripts/echelle.py --dossier {racine.relative_to(ROOT)} "
            f"--partie <n> --pas {pas}")
    if args.calibrer:
        log("\nprojection pour le banc complet :")
        for cible in (1000, 2000, 4000):
            total = sum(r.get("s_par_pas", 0) * cible for r in runs) / 60
            log(f"  {cible} pas par config -> {total:.0f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
