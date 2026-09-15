#!/usr/bin/env bash
# Nuit du 14 au 15/09 — enchaînement automatique, zone verte (orchestration).
#
#   1. attendre la fin du banc de bruit puis du balayage du pas d'apprentissage ;
#   2. banc de bruit sur trois autres tailles : la robustesse dépend-elle de la taille ?
#   3. choisir le pas d'apprentissage qui stabilise dim=640 ;
#   4. relancer B1-large, B2-profond et A4-grand à ce pas, 90 min chacun :
#      la comparaison profondeur contre largeur, restée sans réponse le 08/09.
#
# Garde-fou : si aucun pas ne stabilise dim=640 (perte finale >= 2,5), l'étape 4
# n'est pas lancée — trois runs de 90 min sur des modèles qui divergent ne
# mesureraient rien.
set -u
cd "$(dirname "$0")/.."
J=runs/nuit3-$(date +%Y%m%d-%H%M).log
log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$J"; }

log "attente du banc de bruit"
while pgrep -f "[b]anc_bruit.py" >/dev/null; do sleep 20; done
sleep 30
log "attente du balayage du pas d'apprentissage"
while pgrep -f "[e]chelle.py --lr" >/dev/null; do sleep 60; done

for ckpt in runs/echelle-20260908-2339/A1-petit/*.pt \
            runs/echelle-20260908-2339/A3-moyen/*.pt \
            runs/echelle-20260908-2339/B2-profond/*.pt; do
  log "banc de bruit : $ckpt"
  python3 scripts/banc_bruit.py --checkpoint "$ckpt" >>"$J" 2>&1
  sleep 70   # les dossiers de banc sont horodatés à la minute
done

LR=$(python3 - <<'PY'
import json, glob
meilleur = None
for f in sorted(glob.glob("runs/echelle-*/resultats.json")):
    for r in json.load(open(f)).get("runs", []):
        if r["nom"].startswith("L640") and r.get("val_finale") is not None:
            if meilleur is None or r["val_finale"] < meilleur["val_finale"]:
                meilleur = r
if meilleur and meilleur["val_finale"] < 2.5:
    print(meilleur["apprentissage"])
PY
)
if [ -z "$LR" ]; then
  log "aucun pas ne stabilise dim=640 — étape 4 annulée"
  exit 0
fi
log "pas retenu pour dim=640 : $LR"
python3 scripts/echelle.py --configs "B1-large,B2-profond,A4-grand" \
  --minutes 90 --val-tous 1000 --apprentissage "$LR" >>"$J" 2>&1
log "nuit terminée"
