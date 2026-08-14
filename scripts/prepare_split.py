"""Découpe le corpus de prénoms en train / validation.

Zone verte (tooling) : préparation de données, aucune logique de modèle.

Le corpus `names.txt` est trié par popularité décroissante. Un découpage par simple
troncature mettrait donc tous les prénoms rares en validation → décalage de distribution.
On mélange avec une graine fixée avant de couper, pour que train et val proviennent de
la même distribution, et que le découpage soit reproductible.

Usage :
    python scripts/prepare_split.py [--val-frac 0.1] [--seed 42]
"""

import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DATA / "names.txt", type=Path)
    parser.add_argument("--val-frac", default=0.1, type=float)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    names = args.input.read_text(encoding="utf-8").splitlines()
    names = [n for n in names if n.strip()]  # écarte d'éventuelles lignes vides

    rng = random.Random(args.seed)
    rng.shuffle(names)

    n_val = round(len(names) * args.val_frac)
    val = names[:n_val]
    train = names[n_val:]

    train_path = DATA / "names.train.txt"
    val_path = DATA / "names.val.txt"
    train_path.write_text("\n".join(train) + "\n", encoding="utf-8")
    val_path.write_text("\n".join(val) + "\n", encoding="utf-8")

    print(f"seed={args.seed}  val_frac={args.val_frac}")
    print(f"total  : {len(names)}")
    print(f"train  : {len(train)}  -> {train_path.relative_to(ROOT)}")
    print(f"val    : {len(val)}  -> {val_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
