# Lancer depuis la racine du dépôt : python src/model/<nom_du_fichier>.py
# Les chemins ci-dessous sont relatifs au répertoire courant, pas à l'emplacement du script.

import torch.nn as nn
import torch

alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
dictlettre={}
for i,lettre in enumerate(alphabet):
    dictlettre[lettre]=i
dictlettre[' ']=len(alphabet)

names=open("data/names.train.txt")
taillecaché=100
dim=2
pas=0.05

c=(torch.randn(27, dim)*0.1).requires_grad_(True)
W_xh=(torch.randn(taillecaché, dim)*0.1).requires_grad_(True)
W_hh=(torch.randn(taillecaché, taillecaché)*0.1).requires_grad_(True)
b_h=(torch.randn(taillecaché)*0.1).requires_grad_(True)
W_hy=(torch.randn(27, taillecaché)*0.1).requires_grad_(True)
b_y=(torch.randn(27)*0.1).requires_grad_(True)

for epoch in range(10):
    names.seek(0)
    i=0
    losstot=0
    for name in names:
        h=torch.zeros(taillecaché)
        prev_lettre=' '
        loss_name=0
        for lettre in name:
            if lettre == '\n':
                lettre=' '
            x=c[dictlettre[prev_lettre]]
            h = torch.tanh(W_xh @ x + W_hh @ h + b_h)
            y = torch.softmax(W_hy @ h + b_y, dim=0)
            loss = -torch.log(y[dictlettre[lettre]])
            loss_name += loss
            prev_lettre = lettre
            i+=1
        loss_name.backward()
        torch.nn.utils.clip_grad_norm_([W_xh, W_hh, b_h, W_hy, b_y, c], max_norm=1.0)
        with torch.no_grad():
            W_xh -= W_xh.grad*pas
            W_hh -= W_hh.grad*pas
            b_h -= b_h.grad*pas
            W_hy -= W_hy.grad*pas
            b_y -= b_y.grad*pas
            c -= c.grad*pas
            W_xh.grad=None
            W_hh.grad=None
            b_h.grad=None
            W_hy.grad=None
            b_y.grad=None
            c.grad=None
        losstot+=loss_name.item()
    print(f"epoch {epoch} loss={losstot/i:.4f}")

def genere_prenom(max_len=20):
    h=torch.zeros(taillecaché)
    prev_lettre=' '
    prenom=''
    for _ in range(max_len):
        x=c[dictlettre[prev_lettre]]
        h = torch.tanh(W_xh @ x + W_hh @ h + b_h)
        y = torch.softmax(W_hy @ h + b_y, dim=0).detach()
        idx = torch.multinomial(y, 1).item()
        lettre = list(dictlettre.keys())[list(dictlettre.values()).index(idx)]
        if lettre == ' ':
            break
        prenom += lettre
        prev_lettre = lettre
    return prenom

print("\nGénération de prénoms :")
for _ in range(20):
    print(genere_prenom())