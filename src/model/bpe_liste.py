# Lancer depuis la racine du dépôt : python src/model/<nom_du_fichier>.py
# Les chemins ci-dessous sont relatifs au répertoire courant, pas à l'emplacement du script.

import json
cheminbpe="data/tokenizer/bpe.json"
def bpelist(cheminbpe):
    with open(cheminbpe, 'r', encoding='utf-8') as f:
        bpe = json.load(f)
    retour=bpe['alphabet']+bpe['fusions']
    retour2=[]
    for tok in retour:
        if type(tok) is list:
            retour2.append(tok[0]+tok[1])
        else:
            retour2.append(tok)
    with open("data/tokenizer/bpe_liste.json", "w", encoding="utf-8") as f:
        json.dump(retour2, f,ensure_ascii=False)

if __name__ == "__main__":
    bpelist(cheminbpe)