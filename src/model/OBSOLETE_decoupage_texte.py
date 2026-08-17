# Lancer depuis la racine du dépôt : python src/model/<nom_du_fichier>.py
# Les chemins ci-dessous sont relatifs au répertoire courant, pas à l'emplacement du script.

import collections
chemin="data/stories.train.txt"
texte=open(chemin)
L=[]
for ligne in texte:
    for car in ligne:
        L.append(car)
dictpair={}
passe=0
for car in L:
        if passe==0:
            passe=1
            carpre=car
            continue
        duo=carpre+car
        if duo in dictpair:
            dictpair[duo]+=1
        else:
            dictpair[duo]=1
        carpre=car
maxduo=[max(dictpair,key=dictpair.get)]
L2=[]
passe=0
for i in range(len(L)-1):
    if passe!=0:
        passe=0
    elif [L[i]+L[i+1]]==maxduo:
        L2.append(maxduo[0])
        passe=1
    elif passe==0:
        L2.append(L[i])
if passe==0:
    L2.append(L[-1])
print(L2[0:10])

