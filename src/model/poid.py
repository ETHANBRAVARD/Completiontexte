# Lancer depuis la racine du dépôt : python src/model/<nom_du_fichier>.py
# Les chemins ci-dessous sont relatifs au répertoire courant, pas à l'emplacement du script.

import numpy as np
import random
import math
from copy import deepcopy
from collections import deque
alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']


def W(alphabet):
    sortie=np.array([[random.uniform(-0.1,0.1) for i in range (len(alphabet)+1)]for i in range(len(alphabet)+1)])
    return(sortie)

def one_hot(alphabet,lettre):
    a=len(alphabet)+1
    i=0
    for b in alphabet:
        i+=1
        if b == lettre:
            break
        elif lettre == ' ':
            i=27
            break
    one=np.array([0 for i in range(a)])
    one[i-1]=1
    return(one)


def softmax(one,W):
    smax=np.dot(one,W)
    tot=0
    for i in range(len(smax)):
        smax[i]=math.exp(smax[i])
        tot+=smax[i]
    smax/=tot
    return(smax)


def lossmodel(alphabet,chemin,w):
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
            one=one_hot(alphabet,lettrepre)
            p.append(softmax(one,w)[dictlettre[lettre]])
            lettrepre=lettre
    i=0
    for prob in p:
        som+=math.log(prob)
        i+=1
    return(-som/i)


def lossmodelexemple(alphabet,mot,w):
    som=0
    i=0
    p=[]
    dictlettre={}
    for i,lettre in enumerate(alphabet):
        dictlettre[lettre]=i
    dictlettre[' ']=len(alphabet)
    lettrepre=' '
    for name in mot:
        for lettre in name:
            if lettre == '\n':
                lettre=' '
            one=one_hot(alphabet,lettrepre)
            p.append(softmax(one,w)[dictlettre[lettre]])
            lettrepre=lettre
    i=0
    for prob in p:
        som+=math.log(prob)
        i+=1
    return(-som/i)


def derivee (pas,W,chemin,alphabet):
    a=len(alphabet)+1
    W2=deepcopy(W)
    L=[[0 for i in range(a)] for i in range (a)]
    d1=-lossmodel(alphabet,chemin,W)
    for i in range(len(W)):
        for j in range(len(W[i])):
            W2[i][j]+=pas
            d=d1+lossmodel(alphabet,chemin,W2)
            L[i][j]=d/pas
            W2[i][j]-=pas
    return(L)

        
def retour(w,exemple,pas):
    dictlettre={}
    for i,lettre in enumerate(alphabet):
        dictlettre[lettre]=i
    dictlettre[' ']=len(alphabet)
    d=[]
    df=[]
    M=np.array([[0.0 for i in range (len(alphabet)+1)]for i in range(len(alphabet)+1)])
    lettrepre=' '
    for lettre in exemple:
        if lettre == '\n':
            lettre =' '
        one = one_hot(alphabet,lettrepre)
        p=softmax(one,w)
        d.append([0 for i in range(27)])
        d[-1][dictlettre[lettre]]=-1/p[dictlettre[lettre]]
        df.append(deepcopy(p))
        df[-1][dictlettre[lettre]]-=1
        M[dictlettre[lettrepre]]+=df[-1]/len(exemple)
        lettrepre=lettre
    lettrepre=' '
    return(w-pas*M)

    
def jeu_dexemple(chemin,taille):
    names=open(chemin)
    lettrepre=deque([' 'for i in range (taille)])
    lettre=' '
    L=[]
    for name in names.readlines():
        for lettre in name:
            if lettre == '\n':
                lettre = ' '
            lettrepre2=deepcopy(lettrepre)
            L.append([lettrepre2,lettre])
            lettrepre.popleft()
            lettrepre.append(lettre)
        lettrepre=deque([' 'for i in range (taille)])
    return(L)


def Embeding(c,Contexte):
    dictlettre={}
    K=[]
    for i,lettre in enumerate(alphabet):
        dictlettre[lettre]=i
    dictlettre[' ']=len(alphabet)
    for lettre in Contexte:
        num=dictlettre[lettre]
        K=np.concatenate([K,c[num]])
    return(K)
    

def table_embedding(dim):
    c=[[random.uniform(-1,1) for i in range (dim)] for i in range(27)]
    return(np.array(c))


def lineaire(W,C,b,Contexte):
    E = Embeding(C,Contexte)
    return(np.dot(W,E)+b)


def lineaire2(W2,C,b):
    return(softmax2(C,W2,b))


def tanh(K):
    for i in range(len(K)):
        K[i]=math.tanh(K[i])
    return(K)


def softmax2(one,W,b):
    smax=np.dot(one,W)+b
    tot=0
    for i in range(len(smax)):
        smax[i]=math.exp(smax[i])
        tot+=smax[i]
    smax/=tot
    return(smax)


def assemblage(c,W1,W2,b1,b2,taillecaché,names,K,dim):
    dictlettre={}
    loss=0
    lossfinal=0
    for i,lettre in enumerate(alphabet):
        dictlettre[lettre]=i
    dictlettre[' ']=len(alphabet)
    lettrepre=deque([' 'for i in range (K)])
    lettre=' '
    L=[]
    i=0
    for name in names.readlines():
        for lettre in name:
            if lettre == '\n':
                lettre = ' '
            lettrepre2=deepcopy(lettrepre)
            h = tanh(lineaire(W1, c, b1, lettrepre)) 
            p = lineaire2(h, W2, b2)
            loss-=math.log(p[dictlettre[lettre]])
            lettrepre.popleft()
            lettrepre.append(lettre)
        lettrepre=deque([' 'for i in range (K)])
        loss/=len(name)
        lossfinal+=loss
        loss=0
        i+=1
    return(lossfinal/i)



def retropropagation(c,W1,W2,b1,b2,names,pas):
    som=0
    dictlettre={}
    dim=len(W1[0]) // K
    for i,lettre in enumerate(alphabet):
        dictlettre[lettre]=i
    dictlettre[' ']=len(alphabet)
    lettrepre=deque([' 'for i in range (K)])
    lettre=' '
    L=[]
    i=0
    for name in names.readlines():
        for lettre in name:
            if lettre == '\n':
                lettre = ' '
            E=Embeding(c,lettrepre)
            h = tanh(lineaire(W1, c, b1, lettrepre)) 
            p = lineaire2(h, W2, b2)
            db2=deepcopy(p)
            db2[dictlettre[lettre]]-=1
            dW2=np.outer(db2,h)
            r=np.dot(np.transpose(W2),db2)
            dz1=r*(1-h*h)
            dW1 = np.outer(dz1, E)
            dE = np.dot(np.transpose(W1) , dz1)
            db1 = dz1
            dC=np.zeros((27, dim))
            for i in range(K):
                dC[dictlettre[lettrepre[i]]]+=dE[i*dim:(i+1)*dim]
            c-=pas*dC
            W1-=pas*dW1
            W2-=pas*dW2
            b1-=pas*db1
            b2-=pas*db2
            lettrepre.popleft()
            lettrepre.append(lettre)
        lettrepre=deque(' ' for i in range(K))
    return(c,W1,W2,b1,b2)


if __name__=='__main__':
    taillecaché=100
    K=3
    names=open("data/names.train.txt")
    names2=open("data/names.val.txt")
    dim=2
    c=table_embedding(dim)
    W1=np.array([[random.uniform(-0.1,0.1) for i in range (K*dim)]for i in range(taillecaché)])
    W2=np.array([[random.uniform(-0.1,0.1) for i in range (taillecaché)]for i in range(len(alphabet)+1)])
    b1=np.array([0.0 for i in range (taillecaché)])
    b2=np.array([random.uniform(-0.1,0.1) for i in range (len(alphabet)+1)])
    for i in range (10):
        names2.seek(0)
        print(assemblage(c,W1,W2,b1,b2,taillecaché,names2,K,dim))
        names.seek(0)
        retropropagation(c,W1,W2,b1,b2,names,0.01)
    names2.seek(0)
    print(assemblage(c,W1,W2,b1,b2,taillecaché,names2,K,dim))
    