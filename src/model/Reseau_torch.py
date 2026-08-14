import torch.nn as nn
import torch
from collections import deque
alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

losstot=0
dictlettre={}
for i,lettre in enumerate(alphabet):
    dictlettre[lettre]=i
dictlettre[' ']=len(alphabet)
lettre=' '
names=open("data/names.train.txt")
taillecaché=100
dim=2
Contexte=deque([' ',' ',' '])
cible=12
K=3
pas=0.05
c=(torch.randn(27, dim)*0.1).requires_grad_(True)
E=torch.cat([c[dictlettre[i]] for i in Contexte])
W1=(torch.randn(taillecaché,K*dim)*0.1).requires_grad_(True)
b1=(torch.randn(taillecaché)*0.1).requires_grad_(True)
W2=(torch.randn(27,taillecaché)*0.1).requires_grad_(True)
b2=(torch.randn(27)*0.1).requires_grad_(True)

for i in range (10):
    names.seek(0)
    i=0
    losstot=0
    for name in names:
        for lettre in name:
            if lettre == '\n':
                lettre=' '
            h=W1 @ E + b1
            h2=torch.tanh(h) 
            h3=W2 @ h2 + b2 
            h4=torch.softmax(h3, dim=0)
            loss = -torch.log(h4[dictlettre[lettre]])
            loss.backward()
            with torch.no_grad():
                W1 -=  W1.grad*pas
                b1 -= b1.grad*pas
                W2 -= W2.grad*pas
                b2 -= b2.grad*pas
                c -= c.grad*pas
                W1.grad=None
                b1.grad=None
                W2.grad=None
                b2.grad=None
                c.grad=None
            Contexte.popleft()
            Contexte.append(lettre)
            E=torch.cat([c[dictlettre[i]] for i in Contexte])
            losstot+=loss
            i+=1
        Contexte=deque([' ',' ',' '])
    print((losstot/i).item())
        
        



