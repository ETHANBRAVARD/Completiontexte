# Lancer depuis la racine du dépôt : python src/model/<nom_du_fichier>.py
# Les chemins ci-dessous sont relatifs au répertoire courant, pas à l'emplacement du script.

import json
import numpy as np

cheminbpe="data/tokenizer/bpe_liste.json"
chemintrain="data/encode.json"
chemintest='data/encode.test.json'
cheminval='data/encode.val.json'
with open(cheminbpe, 'r', encoding='utf-8') as f:
    bpe = json.load(f)
with open(chemintrain, 'r', encoding='utf-8') as f:
    encodetrain = json.load(f)
with open(cheminval, 'r', encoding='utf-8') as f:
    encodeval = json.load(f)
with open(chemintest, 'r', encoding='utf-8') as f:
    encodetest = json.load(f)
encodelist=[encodetrain, encodeval, encodetest]
i=0
dictbpe={}
for k in bpe:
    dictbpe[k]=i
    i+=1

encode_tok=[]
for encode in encodelist:
    encode_tok.append([])
    for i in range (len(encode['encode'])):
        encode_tok[-1].append(dictbpe[encode['encode'][i]])
for i in range (len(encode_tok)):
    encode_tok[i]=np.array(encode_tok[i],dtype =np.uint16)
np.save("data/encode_tok_train.npy", encode_tok[0])
np.save("data/encode_tok_val.npy", encode_tok[1])
np.save("data/encode_tok_test.npy", encode_tok[2])
