import json
import re
chemin="data/stories.train.txt"
with open(chemin) as stories:
    liste=set([])
    for ligne in stories:
        ligne=ligne.lower()
        liste.update(re.findall(r"[a-zA-Z][a-zA-Z']*", ligne))
liste=sorted(liste)
with open ("data/vocabulaire.json",'w') as f:
    json.dump(liste, f, ensure_ascii=False)