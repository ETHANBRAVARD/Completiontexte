# Lancer depuis la racine du dépôt : python src/model/<nom_du_fichier>.py
# Les chemins ci-dessous sont relatifs au répertoire courant, pas à l'emplacement du script.

import json
cheminbpe="data/tokenizer/bpe.json"
chemin="data/stories.train.txt" 
def decouper(chemin):
    with open(chemin, 'r', encoding='utf-8') as f:
        texte = f.read(100)
        mot=''
        while texte:
            for lettre in texte:
                if lettre == ' ' or lettre =='\n':
                    yield mot
                    mot= lettre
                else:
                    mot+=lettre
            texte = f.read(100)
        yield mot


def tokkenisation(mot,bpe,bpeset):
    mot2=[]
    lettre_non_reconnu=set()
    for lettre in mot:
        if lettre not in bpeset :
            lettre_non_reconnu.add(lettre)
        if lettre_non_reconnu:
            print('lettres non reconnues :',lettre_non_reconnu)
            exit()
        mot2.append(lettre)
    for fusion in bpe['fusions']: 
        i=0
        while i < len(mot2):
            car=mot2[i]
            if i<len(mot2)-1:
                if car == fusion[0] and mot2[i+1] == fusion[1]:
                    mot2[i]=fusion[0]+fusion[1]
                    del mot2[i+1]
            i+=1
    return mot2

def encode(chemin, cheminbpe,way= 'data/encode.json'):
    encode2=[]
    dejavue={}
    flag=True
    with open(cheminbpe, 'r', encoding='utf-8') as f:
        bpe = json.load(f)
        bpeset=set(bpe['alphabet'])
        encode=decouper(chemin)
        for mot in encode:
            if flag:
                mot=' '+mot
                flag=False
            if mot in dejavue:
                encode2.extend(dejavue[mot])
            else:
                dejavue[mot]=tokkenisation(mot,bpe,bpeset)
                encode2.extend(dejavue[mot])
    retour={'encode':encode2}
    with open(way, "w", encoding="utf-8") as f:
        json.dump(retour, f,ensure_ascii=False)

if __name__ == "__main__":
    encode(chemin, cheminbpe)


