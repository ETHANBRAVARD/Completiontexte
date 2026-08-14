#!/usr/bin/env bash
# Chaîne complète : corpus 500 Mo -> tokens -> entiers -> entraînement.
# Zone verte : orchestration seule. Chaque étape appelle le code d'Ethan.
# S'arrête à la première erreur plutôt que de laisser la suivante travailler
# sur des données incomplètes.
set -e
cd /home/ethanbravard/projet_perso/Completiontexte

echo "############ 1/3  préparation + encodage des 500 Mo ############"
date '+%H:%M:%S'
python3 -u scripts/encode_gros_corpus.py --mo 500 --taille-morceau 85

echo
echo "############ 2/3  conversion token -> entier ############"
date '+%H:%M:%S'
/usr/bin/time -v python3 -u src/model/rencode.py 2>&1 | grep -E 'Maximum resident|Command exited' || true
python3 - <<'EOF'
import numpy as np, json, os
v = json.load(open('data/tokenizer/bpe_liste.json'))
total = 0
for s, txt in [('train','data/stories.train.txt'),
               ('val','data/stories.val.txt'),
               ('test','data/stories.test.txt')]:
    p = f'data/encode_tok_{s}.npy'
    a = np.load(p); t = open(txt, encoding='utf-8').read(); total += len(a)
    ok = ''.join(v[k] for k in a)[1:] == t
    print("  %-6s %10d tokens | %-7s | %6.1f Mo | 0..%d | ratio %.4f | aller-retour %s"
          % (s, len(a), a.dtype, os.path.getsize(p)/1e6, a.max(), len(t)/len(a), ok))
    assert ok, f"aller-retour rompu sur {s}"
print("  total : %d tokens" % total)
EOF

echo
echo "############ 3/3  entraînement ############"
date '+%H:%M:%S'
python3 -u src/model/transformer.py

echo
echo "############ terminé ############"
date '+%H:%M:%S'