#!/usr/bin/env bash
# Run long de l'étape 5 — zone verte (orchestration).
#
# Le livrable manquant de l'étape 5 : un entraînement sur le CORPUS COMPLET
# (497 085 153 tokens d'entraînement), avec le schedule retenu. Tout ce qui a
# tourné jusqu'ici était soit sur l'ancien corpus de 500 Mo (16/08), soit des
# bancs de 78 à 90 minutes.
#
# Dimensionnement — 12 288 tokens par pas (lot 32 x contexte 384) :
#
#   config       forme   params   Chinchilla   pas      durée   époques
#   A2-actuel    384x6   12.4 M     248 M tok   20 166   1.8 h    0.50
#   A3-moyen     512x7   24.4 M     488 M tok   39 681   6.0 h    0.98   <-- retenu
#   B1-large     640x5   27.5 M     550 M tok   44 792   7.7 h    1.11
#   A4-grand     640x8   42.3 M     846 M tok   68 815  17.9 h    1.70
#
# A3-moyen est le seul point où l'optimum de Chinchilla et la taille du corpus
# coïncident : 40 453 pas = 1 époque exacte = 20,4 tokens par paramètre. Aucun
# token n'est vu deux fois, donc la question du surapprentissage ne se pose pas,
# et le run tient dans une nuit.
#
# ATTENTION, valeurs à confirmer avant lancement :
#   --apprentissage : 2e-3 est le meilleur mesuré à dim=384. Le pas maximal
#     stable varie comme l'inverse de la largeur (mesuré le 15/09) ; 384 -> 512
#     fait x1,33, donc 2e-3 pourrait ne pas tenir. D'où la sonde ci-dessous.
#   --fin : à remplacer par le résultat de la nuit du 20-21.
set -u
cd "$(dirname "$0")/../.."

LR=${LR:-0.0015}          # pas de base      (surchargeable : LR=0.002 ./run_long.sh)
FIN=${FIN:-14000}         # pas de décroissance -> à fixer d'après la nuit 6
PAS=${PAS:-40453}         # 1 époque exacte
CONFIG=${CONFIG:-A3-moyen}

J=runs/runlong-$(date +%Y%m%d-%H%M).log
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$J"; }

commun=(--configs "$CONFIG" --apprentissage "$LR" --schedule cos
        --amorce 1000 --clip 1.0)

log "=== sonde de stabilite : 1500 pas, ~14 min ==="
log "    $CONFIG  lr=$LR  cos  clip=1.0"
python3 scripts/echelle.py "${commun[@]}" --pas 1500 --fin 500 \
        --val-tous 250 >>"$J" 2>&1
SONDE=$(ls -d runs/echelle-* | tail -1)

log "--- pertes de validation de la sonde ($SONDE) ---"
grep -A1 'jeu de validation' "$SONDE"/*/log.txt | grep '^i=' | tee -a "$J"

cat <<'TXT' | tee -a "$J"

Lis les pertes ci-dessus AVANT de lancer les six heures.
  - elles doivent DÉCROÎTRE à chaque mesure ;
  - au pas 1500, attendre ~2,3 (c'est ce que donnait 384x6) ;
  - si ça monte ou stagne au-dessus de 3, le pas est trop grand :
    relancer avec LR=0.001 et refaire la sonde.

Le run long ne part PAS tout seul. Quand la sonde est bonne :

    PAS=40453 LR=<le pas validé> FIN=<la nuit 6> ./scripts/nuits/run_long.sh --go

TXT

if [ "${1:-}" != "--go" ]; then
  log "=== sonde seule, run long non lance (ajouter --go) ==="
  exit 0
fi

log "=== RUN LONG : $CONFIG, $PAS pas, lr=$LR, cos, fin=$FIN, clip=1.0 ==="
log "    ~6 h — checkpoints conservés tous les 2000 pas"
python3 scripts/echelle.py "${commun[@]}" --pas "$PAS" --fin "$FIN" \
        --val-tous 2000 --garder-tout >>"$J" 2>&1

log "=== run long termine ==="
tail -20 "$J"
