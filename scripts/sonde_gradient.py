"""Sonde : distribution de la norme du gradient, pour choisir un seuil d'écrêtage.

Zone verte (tooling) : mesure et tracé. N'implémente aucune logique de modèle et
ne modifie jamais `src/model/transformer.py` — il en dépose une copie dans le
dossier de la sonde et n'y change que trois choses :

  - la taille du modèle et le nombre de pas ;
  - le seuil d'écrêtage, porté à une valeur inatteignable, pour observer la
    distribution **brute** : un seuil actif fausserait la mesure en tronquant
    justement ce qu'on cherche à voir ;
  - la valeur de retour de `clip_grad_norm_`, qui est la norme AVANT écrêtage,
    recueillie dans une liste puis écrite en JSON.

Rien d'autre n'est touché : ni la passe avant, ni la passe arrière, ni la mise à
jour des poids. La sonde lit, elle ne calcule pas.

Usage :
    python3 scripts/sonde_gradient.py --pas 1200
    python3 scripts/sonde_gradient.py --configs "384x6,640x8" --pas 1200
"""

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELE = ROOT / "src" / "model" / "transformer.py"


def source_sonde(dim, blocs, heads, nb_pas, dossier, lr):
    src = MODELE.read_text(encoding="utf-8")
    remplacements = [
        (r"^dim=\d+$",                f"dim={dim}"),
        (r"^num_heads=\d+$",          f"num_heads={heads}"),
        (r"^num_blocs=\d+$",          f"num_blocs={blocs}"),
        (r"^nb_passage=\d+$",         f"nb_passage={nb_pas}"),
        (r"^config_pas=.*$",          f"config_pas=['continu',1000,1000,{lr}]"),
        (r"^chemin_dossier = .*$",    f"chemin_dossier = r'{dossier}'"),
        (r"^t=1$",                    "t=1\n_normes=[]"),
        # seuil rendu inatteignable + capture de la norme d'avant écrêtage
        (r"^    torch\.nn\.utils\.clip_grad_norm_\(params, max_norm=[\d.e+]+\)$",
         "    _normes.append(float(torch.nn.utils.clip_grad_norm_(params, max_norm=1e30)))"),
        (r"^for _ in range\(20\):$",  "for _ in range(0):"),
    ]
    for motif, remplacement in remplacements:
        src, n = re.subn(motif, remplacement, src, flags=re.MULTILINE)
        if n != 1:
            raise SystemExit(f"substitution « {motif} » trouvée {n} fois au lieu d'une.\n"
                             f"La sonde s'arrête : sans elle, la mesure porterait sur "
                             f"autre chose que ce qu'on croit.")
    src += (f"\n\nimport json as _j\n"
            f"_j.dump(_normes, open(r'{dossier}/normes.json','w'))\n"
            f"print('normes écrites :', len(_normes))\n")
    return src


def lancer(nom, dim, blocs, heads, nb_pas, lr, racine, log):
    dossier = racine / nom
    dossier.mkdir(parents=True, exist_ok=True)
    source = dossier / "source.py"
    source.write_text(source_sonde(dim, blocs, heads, nb_pas, str(dossier), lr),
                      encoding="utf-8")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src" / "model") + os.pathsep + env.get("PYTHONPATH", "")
    log(f"  {nom:10} dim={dim} blocs={blocs} lr={lr:g} — {nb_pas} pas…", fin="")
    t = time.perf_counter()
    with (dossier / "log.txt").open("w", encoding="utf-8") as flux:
        code = subprocess.run([sys.executable, "-u", str(source)], cwd=ROOT,
                              stdout=flux, stderr=subprocess.STDOUT, env=env)
    duree = time.perf_counter() - t
    if code.returncode != 0:
        derniere = (dossier / "log.txt").read_text(encoding="utf-8").strip().splitlines()
        log(f"  ÉCHEC — {(derniere or ['(vide)'])[-1][:70]}")
        return None
    normes = json.loads((dossier / "normes.json").read_text(encoding="utf-8"))
    log(f"  {duree/60:.1f} min, {len(normes)} normes")
    for f in dossier.glob("*.pt"):
        f.unlink()
    return normes


def centiles(v, qs=(50, 90, 95, 99, 99.9, 100)):
    s = sorted(v)
    return {q: s[min(len(s) - 1, int(q / 100 * len(s)))] for q in qs}


def tracer(mesures, chemin):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "src" / "tooling"))
    import tracer as T

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)
    T._cadre(axes[0], "Norme du gradient au fil des pas", "échelle logarithmique",
             "pas", "norme du gradient")
    T._cadre(axes[1], "Distribution", "chaque trait horizontal est un centile", "", "")
    axes[0].set_yscale("log"); axes[1].set_yscale("log")
    for i, (nom, v) in enumerate(mesures.items()):
        c = T.SERIES[i % len(T.SERIES)]
        axes[0].plot(range(len(v)), v, color=c, linewidth=0.7, alpha=0.8, label=nom)
        axes[1].boxplot([v], positions=[i], widths=0.5, whis=(1, 99),
                        showfliers=True, flierprops=dict(marker=".", markersize=2,
                                                         markerfacecolor=c, markeredgecolor=c),
                        medianprops=dict(color=c, linewidth=2),
                        boxprops=dict(color=c), whiskerprops=dict(color=c),
                        capprops=dict(color=c))
    axes[1].set_xticks(range(len(mesures)))
    axes[1].set_xticklabels(list(mesures), fontsize=9, color=T.ENCRE_2)
    leg = axes[0].legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(T.ENCRE_2)
    fig.set_facecolor(T.SURFACE); fig.tight_layout()
    fig.savefig(chemin, dpi=150, facecolor=T.SURFACE)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", default="384x6x4,640x8x8",
                   help="dim x blocs x têtes, avec un pas d'apprentissage optionnel "
                        "après @ : « 640x8x8@0.0005 ». Sinon --lr s'applique.")
    p.add_argument("--pas", type=int, default=1200)
    p.add_argument("--lr", type=float, default=0.001)
    args = p.parse_args()

    racine = ROOT / "runs" / f"gradient-{datetime.now():%Y%m%d-%H%M}"
    racine.mkdir(parents=True, exist_ok=True)

    def log(m="", fin="\n"):
        print(m, end=fin, flush=True)

    log(f"sonde  : {racine.relative_to(ROOT)}")
    log(f"seuil d'écrêtage porté à 1e30 — on observe la distribution brute\n")

    mesures = {}
    for c in args.configs.split(","):
        forme, _, lr_txt = c.partition("@")
        lr = float(lr_txt) if lr_txt else args.lr
        dim, blocs, heads = (int(x) for x in forme.split("x"))
        v = lancer(f"dim{dim}", dim, blocs, heads, args.pas, lr, racine, log)
        if v:
            mesures[f"dim {dim} × {blocs} blocs (lr {lr:g})"] = v

    log(f"\n  {'config':22}{'médiane':>10}{'p90':>10}{'p95':>10}{'p99':>10}{'p99.9':>10}{'max':>10}")
    resume = {}
    for nom, v in mesures.items():
        q = centiles(v)
        resume[nom] = q
        log(f"  {nom:22}" + "".join(f"{q[k]:10.3f}" for k in (50, 90, 95, 99, 99.9, 100)))

    (racine / "resume.json").write_text(
        json.dumps({"pas": args.pas, "lr": args.lr,
                    "centiles": {k: {str(q): v for q, v in d.items()} for k, d in resume.items()},
                    "normes": mesures}, ensure_ascii=False, indent=1), encoding="utf-8")
    tracer(mesures, racine / "gradient.png")
    log(f"\nécrit : {(racine/'resume.json').relative_to(ROOT)}, "
        f"{(racine/'gradient.png').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
