#!/usr/bin/env bash
set -e
cd /home/ethanbravard/projet_perso/Completiontexte
echo "### 1/3 réencodage de train seul ###"; date '+%H:%M:%S'
python3 -u scripts/encode_gros_corpus.py --splits train --sauter-preparation --sans-archivage
echo; echo "### 2/3 conversion token -> entier ###"; date '+%H:%M:%S'
python3 -u src/model/rencode.py
python3 - <<'PY'
import numpy as np, json, os
v=json.load(open('data/tokenizer/bpe_liste.json')); tot=0
for s,txt in [('train','data/stories.train.txt'),('val','data/stories.val.txt'),('test','data/stories.test.txt')]:
    p=f'data/encode_tok_{s}.npy'; a=np.load(p); t=open(txt,encoding='utf-8').read(); tot+=len(a)
    ok=''.join(v[k] for k in a)[1:]==t
    print("  %-6s %10d tokens | %-7s | %6.1f Mo | 0..%d | aller-retour %s"%(s,len(a),a.dtype,os.path.getsize(p)/1e6,a.max(),ok))
    assert ok
print("  total : %d tokens"%tot)
PY
echo; echo "### 3/3 entraînement ###"; date '+%H:%M:%S'
python3 -u src/model/transformer.py
echo; echo "### terminé ###"; date '+%H:%M:%S'
