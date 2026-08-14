# Lancer depuis la racine du dépôt : python src/model/<nom_du_fichier>.py
# Les chemins ci-dessous sont relatifs au répertoire courant, pas à l'emplacement du script.

import json

cheminencode="src/model/encode.json"

def decode(cheminencode):
    with open(cheminencode, 'r', encoding='utf-8') as f:
        encode = json.load(f)
    decode=''
    i=0
    for tok in encode['encode']:
        decode+=tok
    print(decode[1:])

if __name__ == "__main__":
    decode(cheminencode)