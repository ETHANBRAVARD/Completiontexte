#!/usr/bin/env bash
# Nuit du 20 au 21/09 — zone verte (orchestration).
#
# Dernier étalonnage avant le run long. Tout est gelé aux meilleures valeurs
# du 19-20/09 — cos, lr 2e-3, clip 1.0, 384x6, 15000 pas, échauffement 1000 —
# et on ne fait varier QUE le moment où la décroissance commence.
#
#   `--fin N` = la décroissance occupe les N derniers pas.
#   fin=1000  -> elle couvre 7 % du run   (le réglage du 19/09, val 1.4912)
#   fin=14000 -> elle démarre juste après l'échauffement (le réglage publié)
#
# Un run de témoin à graine différente donne le PLANCHER DE BRUIT : on compare
# ici des écarts de l'ordre de 0.02, et sans ce chiffre on ne saura pas si un
# écart est un effet ou du hasard. C'est la mesure la plus importante de la nuit.
#
# Ordre choisi pour que la nuit reste utile si elle est coupée : l'extrême,
# puis le milieu, puis le témoin de bruit.
#
# ~79 min par run, 6 runs -> ~8 h. Lancé à 20 h, terminé vers 4 h.
set -u
cd "$(dirname "$0")/../.."
J=runs/nuit6-$(date +%Y%m%d-%H%M).log
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$J"; }

commun=(--configs A2-actuel --pas 15000 --val-tous 1000
        --apprentissage 0.002 --schedule cos --amorce 1000 --clip 1.0)

log "=== nuit 6 : ou commence la decroissance ? ==="

for fin in 14000 7000 3750 10500; do
  log "--- decroissance sur les $fin derniers pas ---"
  python3 scripts/echelle.py "${commun[@]}" --fin "$fin" >>"$J" 2>&1
  sleep 65
done

log "--- temoins de bruit : meme config, graine 4242 ---"
for fin in 1000 14000; do
  log "    fin=$fin graine=4242"
  python3 scripts/echelle.py "${commun[@]}" --fin "$fin" --graine 4242 >>"$J" 2>&1
  sleep 65
done

log "=== nuit terminee ==="
grep -h 'A2-actuel' runs/echelle-2026092*/banc.txt 2>/dev/null | tee -a "$J"
