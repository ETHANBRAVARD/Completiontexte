# Lancer depuis la racine du dépôt : python src/model/<nom_du_fichier>.py
# Les chemins ci-dessous sont relatifs au répertoire courant, pas à l'emplacement du script.

import json
cheminbpe="data/tokenizer/bpe.json"
chemin="data/stories.train.txt" 
def encode(chemin, cheminbpe):
    with open(cheminbpe, 'r', encoding='utf-8') as f:
        bpe = json.load(f)
    bpeset=set(bpe['alphabet'])
    encode = open(chemin, 'r', encoding='utf-8').read()
    séparateur=['[',']']
    if séparateur[0] in bpe['alphabet'] or séparateur[1] in bpe['alphabet']:
        print('change de séparateur')
        exit()
    encode2=[séparateur[0]+' '+séparateur[1]]
    lettre_non_reconnu=set()
    for lettre in encode:
        if lettre not in bpeset :
            lettre_non_reconnu.add(lettre)
        encode2.append(séparateur[0]+lettre+séparateur[1])
    encode2=''.join(encode2)
    if lettre_non_reconnu:
        print('lettres non reconnues :',lettre_non_reconnu)
        exit()
    for fusion in bpe['fusions']: 
        encode2 =encode2.replace('['+fusion[0]+']'+'['+fusion[1]+']', '['+fusion[0]+fusion[1]+']')
    encode3=[]
    i=0
    for car in encode2:
        if car == '[':
            j=1
            while encode2[i+j] != ']':
                j+=1
            encode3.append(encode2[i+1:i+j])
        i+=1
    retour={'encode':encode3}
    with open("data/encode.json", "w", encoding="utf-8") as f:
        json.dump(retour, f,ensure_ascii=False)

if __name__ == "__main__":
    encode(chemin, cheminbpe)


