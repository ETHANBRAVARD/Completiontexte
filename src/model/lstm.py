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

W_ix=(torch.randn(taillecaché, dim)*0.1).requires_grad_(True)
W_ih=(torch.randn(taillecaché, taillecaché)*0.1).requires_grad_(True)
b_i=(torch.randn(taillecaché)*0.1).requires_grad_(True)

W_fx=(torch.randn(taillecaché, dim)*0.1).requires_grad_(True)
W_fh=(torch.randn(taillecaché, taillecaché)*0.1).requires_grad_(True)
b_f=(torch.randn(taillecaché)*0.1).requires_grad_(True)

W_ox=(torch.randn(taillecaché, dim)*0.1).requires_grad_(True)
W_oh=(torch.randn(taillecaché, taillecaché)*0.1).requires_grad_(True)
b_o=(torch.randn(taillecaché)*0.1).requires_grad_(True)

W_cx=(torch.randn(taillecaché, dim)*0.1).requires_grad_(True)
W_ch=(torch.randn(taillecaché, taillecaché)*0.1).requires_grad_(True)
b_c=(torch.randn(taillecaché)*0.1).requires_grad_(True)

W_hy=(torch.randn(27, taillecaché)*0.1).requires_grad_(True)
b_y=(torch.randn(27)*0.1).requires_grad_(True)

params=[W_ix,W_ih,b_i,W_fx,W_fh,b_f,W_ox,W_oh,b_o,W_cx,W_ch,b_c,W_hy,b_y,c]

for epoch in range(10):
    names.seek(0)
    i=0
    losstot=0
    for name in names:
        h=torch.zeros(taillecaché)
        cell=torch.zeros(taillecaché)
        prev_lettre=' '
        loss_name=0
        for lettre in name:
            if lettre == '\n':
                lettre=' '
            x=c[dictlettre[prev_lettre]]
            i_gate=torch.sigmoid(W_ix @ x + W_ih @ h + b_i)
            f_gate=torch.sigmoid(W_fx @ x + W_fh @ h + b_f)
            o_gate=torch.sigmoid(W_ox @ x + W_oh @ h + b_o)
            c_tilde=torch.tanh(W_cx @ x + W_ch @ h + b_c)
            cell=f_gate*cell+i_gate*c_tilde
            h=o_gate*torch.tanh(cell)
            y=torch.softmax(W_hy @ h + b_y, dim=0)
            loss=-torch.log(y[dictlettre[lettre]])
            loss_name+=loss
            prev_lettre=lettre
            i+=1
        loss_name.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        with torch.no_grad():
            for p in params:
                p -= p.grad*pas
                p.grad=None
        losstot+=loss_name.item()
    print(f"epoch {epoch} train_loss={losstot/i:.4f}", end='')

    with torch.no_grad():
        names2=open("data/names.val.txt")
        j=0
        losstot2=0
        for name in names2:
            h=torch.zeros(taillecaché)
            cell=torch.zeros(taillecaché)
            prev_lettre=' '
            loss_name=0
            for lettre in name:
                if lettre == '\n':
                    lettre=' '
                x=c[dictlettre[prev_lettre]]
                i_gate=torch.sigmoid(W_ix @ x + W_ih @ h + b_i)
                f_gate=torch.sigmoid(W_fx @ x + W_fh @ h + b_f)
                o_gate=torch.sigmoid(W_ox @ x + W_oh @ h + b_o)
                c_tilde=torch.tanh(W_cx @ x + W_ch @ h + b_c)
                cell=f_gate*cell+i_gate*c_tilde
                h=o_gate*torch.tanh(cell)
                y=torch.softmax(W_hy @ h + b_y, dim=0)
                loss=-torch.log(y[dictlettre[lettre]])
                loss_name+=loss
                prev_lettre=lettre
                j+=1
            losstot2+=loss_name.item()
        print(f"  val_loss={losstot2/j:.4f}")

def genere_prenom(max_len=20):
    h=torch.zeros(taillecaché)
    cell=torch.zeros(taillecaché)
    prev_lettre=' '
    prenom=''
    for _ in range(max_len):
        x=c[dictlettre[prev_lettre]]
        i_gate = torch.sigmoid(W_ix @ x + W_ih @ h + b_i)
        f_gate = torch.sigmoid(W_fx @ x + W_fh @ h + b_f)
        o_gate = torch.sigmoid(W_ox @ x + W_oh @ h + b_o)
        c_tilde = torch.tanh(W_cx @ x + W_ch @ h + b_c)
        cell = f_gate * cell + i_gate * c_tilde
        h = o_gate * torch.tanh(cell)
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