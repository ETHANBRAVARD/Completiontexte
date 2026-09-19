#!/usr/bin/env bash
# Nuit du 19 au 20/09 — zone verte (orchestration).
#
#   A. sonde de gradient sur quatre tailles, chacune à SON pas d'apprentissage
#      -> la loi d'échelle du seuil d'écrêtage                        ~40 min
#   B. dim 640 à 1e-3 AVEC échauffement et écrêtage
#      -> l'échauffement suffit-il à rattraper la divergence ?        ~90 min
#   C. banc des schedules sur 384x6 : continu / cos / racine x 2 pas
#      -> la question qui bloque l'étape 5                            ~7 h 40
#
# À taille fixée, iso-pas et iso-temps coïncident : on compare donc à nombre
# de pas égal, ce qui est indispensable ici — la forme du cosinus dépend de
# `nb_passage`, elle n'aurait aucun sens avec un arrêt au chronomètre.
set -u
cd "$(dirname "$0")/.."
J=runs/nuit5-$(date +%Y%m%d-%H%M).log
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$J"; }

log "=== A. sonde de gradient, quatre tailles ==="
python3 scripts/sonde_gradient.py --pas 1200 \
  --configs "256x4x4@0.001,384x6x4@0.001,512x7x8@0.0005,640x8x8@0.0005" >>"$J" 2>&1

log "=== B. dim 640 a 1e-3 avec echauffement et ecretage ==="
python3 scripts/echelle.py --configs A4-grand --pas 6000 --val-tous 500 \
  --apprentissage 0.001 --schedule cos --amorce 1000 --fin 1000 --clip 1.0 >>"$J" 2>&1

log "=== C. banc des schedules, 384x6, 15000 pas ==="
for lr in 0.001 0.002; do
  for sched in continu cos racine; do
    log "    schedule=$sched  lr=$lr"
    python3 scripts/echelle.py --configs A2-actuel --pas 15000 --val-tous 1000 \
      --apprentissage "$lr" --schedule "$sched" --amorce 1000 --fin 1000 \
      --clip 1.0 >>"$J" 2>&1
    sleep 65
  done
done
log "=== nuit terminee ==="
