import torch
import random
import numpy as np

def tireur_de_lot (tableau, B, T):
    appareil=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    decalage=[random.randint(0, len(tableau)-T-1) for i in range(B)]
    Lots=[]
    Lots1=[[]for i in range(B)]
    Lots2=[[]for i in range(B)]
    for i in range(B):
        Lots.append(tableau[decalage[i]:decalage[i]+T+1])
    for i in range(len(Lots)):
        for j in range (len(Lots[i])):
            if j!=0 :
                Lots1[i].append(Lots[i][j])
            if j!=len(Lots[i])-1:
                Lots2[i].append(Lots[i][j])
    return  torch.tensor(Lots2,dtype=torch.int32,device=appareil),torch.tensor(Lots1,dtype=torch.int32,device=appareil)

