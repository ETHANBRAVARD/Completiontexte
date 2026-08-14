# Lancer depuis la racine du dépôt : python src/model/<nom_du_fichier>.py
# Les chemins ci-dessous sont relatifs au répertoire courant, pas à l'emplacement du script.

import math
import random
import numpy
def loss(alphabet,name,L):
    p=[]
    dictlettre={}
    tot=0
    som=0
    for i,lettre in enumerate(alphabet):
        dictlettre[lettre]=i
    dictlettre[' ']=len(alphabet)
    lettrepre=' '
    for lettre in name:
        if lettre == '\n':
            lettre=' '
        p.append(L[dictlettre[lettrepre]][dictlettre[lettre]])
        lettrepre=lettre
    for prob in p:
        tot+=1
        som+=math.log(prob)
    som/=tot
    return(-som)


def lossmodel(alphabet,L,chemin):
    som=0
    names=open(chemin)
    i=0
    p=[]
    dictlettre={}
    for i,lettre in enumerate(alphabet):
        dictlettre[lettre]=i
    dictlettre[' ']=len(alphabet)
    lettrepre=' '
    for name in names:
        for lettre in name:
            if lettre == '\n':
                lettre=' '
            p.append(L[dictlettre[lettrepre]][dictlettre[lettre]])
            lettrepre=lettre
    for prob in p:
        som+=math.log(prob)
        i+=1
    return(-som/i)


def tirageproba(alphabet,lettre,L):
    dictinver={}
    j=0
    liste=[]
    for i in range (len(alphabet)):
        dictinver[alphabet[i]]=i
    dictinver[' ']=len(alphabet)
    n=dictinver[lettre]
    sousliste=L[n]
    for a in alphabet:
        liste.append(a)
    liste.append(' ')
    sortie=random.choices(liste,weights=sousliste,k=1)
    return(sortie[0])


def inventeprenom(alphabet,L):
    lettre=' '
    prenom=''
    while True:
        l=tirageproba(alphabet,lettre,L)
        if l == ' ':
            break
        prenom+=l
        lettre=l
    return(prenom)


def frequence(alphabet,chemin):
    L=[[0 for i in range(len(alphabet)+1)]for i in range(len(alphabet)+1)]
    names=open(chemin)
    dictlettre={}
    for i,lettre in enumerate(alphabet):
        dictlettre[lettre]=i
    dictlettre[' ']=len(alphabet)
    tot=0
    i=0
    lettrepre=' '
    for ligne in names.readlines():
         for lettre in ligne:
            if lettre == '\n':
                lettre=' '
            L[dictlettre[lettrepre]][dictlettre[lettre]]+=1
            lettrepre=lettre
    for i,l in enumerate(L):
        for j,l2 in enumerate(L):
            tot+=L[i][j]
        for j,l2 in enumerate(L):
                L[i][j]+=0.001
                L[i][j]/=tot
        tot=0
    return(L)


if __name__=='__main__':
    alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    chemin="data/names.txt"
    names=open(chemin)
    L=frequence(alphabet,chemin)