# Lancer depuis la racine du dépôt : python src/model/<nom_du_fichier>.py
# Les chemins ci-dessous sont relatifs au répertoire courant, pas à l'emplacement du script.

import json
chemin="data/stories.train.txt"
def bpe(chemin,budget):
    mots={'\n':0}
    for ligne in open(chemin):
        i=0
        while i<len(ligne):
            car=ligne[i]
            if car == ' ':
                mot=' '
                i+=1
                car=ligne[i]
            else:
                mot = ''
            while car!=' ' and car!='\n':
                mot+=car
                i+=1
                if i>=len(ligne):
                    break
                car=ligne[i]
            if car=='\n':
                mots['\n']+=1
                i+=1
            if mot in mots:
                mots[mot]+=1
            else:
                if mot != '':
                    mots[mot]=1
            if i>=len(ligne):
                break

    mots_seq={}
    for mot,count in mots.items():
        mots_seq[tuple(mot)]=count
    mots=mots_seq
    bpe={'alphabet':[], 'fusions':[]}
    bpeset=set(bpe['alphabet'])
    for lettre in mots.items():
        for i in range(len(lettre[0])):
            if not lettre[0][i] in bpeset:
                bpe['alphabet'].append(lettre[0][i])
                bpeset.add(lettre[0][i])

    for fusion in range(budget):
        compteur={}
        for seq,count in mots.items():
            for i in range(len(seq)-1):
                paire=(seq[i],seq[i+1])
                compteur[paire]=compteur.get(paire,0)+count
        if not compteur:
            break
        meilleur=max(compteur,key=compteur.get)
        bpe['fusions'].append(meilleur)

        mots2={}
        for seq,count in mots.items():
            nouvelle=[]
            i=0
            while i<len(seq):
                if i+1<len(seq) and seq[i]==meilleur[0] and seq[i+1]==meilleur[1]:
                    nouvelle.append(meilleur[0]+meilleur[1])
                    i+=2
                else:
                    nouvelle.append(seq[i])
                    i+=1
            mots2[tuple(nouvelle)]=mots2.get(tuple(nouvelle),0)+count
        mots=mots2
    with open("data/tokenizer/bpe.json", "w", encoding="utf-8") as f:
        json.dump(bpe, f, ensure_ascii=False)

if __name__ == "__main__":
    bpe(chemin)
