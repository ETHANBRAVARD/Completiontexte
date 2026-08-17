# Lancer depuis la racine du dépôt : python src/model/<nom_du_fichier>.py
# Les chemins ci-dessous sont relatifs au répertoire courant, pas à l'emplacement du script.

import datetime
import torch
import torch.nn.functional as F
import numpy as np
import json
from tireur_de_lot import tireur_de_lot
import random
import pathlib
import math

appareil=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
chemin="data/tokenizer/bpe_liste.json"
with open(chemin,'r') as f:
    alphabet=json.load(f)
alph=len(alphabet)
tok=np.load("data/encode_tok_train.npy", "r")
tok_val=np.load("data/encode_tok_val.npy", "r")
dim=384
num_heads=4
num_blocs=6
head_dim=dim//num_heads
max_len=384
lot=32
pas=0.001
date=format(datetime.datetime.now(), '%Y%m%d-%H%M')
seed=1337
torch.manual_seed(seed)
random.seed(seed)
chemin_dossier = f'runs/save_runs/{date}' 

c=(torch.randn(alph, dim,device=appareil)*0.1).requires_grad_(True)
pos_emb=(torch.randn(max_len, dim,device=appareil)*0.1).requires_grad_(True)


W_q=[(torch.randn(dim, dim,device=appareil)*1/math.sqrt(dim)).requires_grad_(True) for _ in range(num_blocs)]
W_k=[(torch.randn(dim, dim,device=appareil)*1/math.sqrt(dim)).requires_grad_(True) for _ in range(num_blocs)]
W_v=[(torch.randn(dim, dim,device=appareil)*1/math.sqrt(dim)).requires_grad_(True) for _ in range(num_blocs)]
W_o=[(torch.randn(dim, dim,device=appareil)*1/math.sqrt(dim)).requires_grad_(True) for _ in range(num_blocs)]

W_1=[(torch.randn(dim*4, dim,device=appareil)*1/math.sqrt(dim)).requires_grad_(True) for _ in range(num_blocs)]
b_1=[(torch.randn(dim*4,device=appareil)*0.1).requires_grad_(True) for _ in range(num_blocs)]
W_2=[(torch.randn(dim, dim*4,device=appareil)*1/math.sqrt(4*dim)).requires_grad_(True) for _ in range(num_blocs)]
b_2=[(torch.randn(dim,device=appareil)*0.1).requires_grad_(True) for _ in range(num_blocs)]
ln1_g=[torch.ones(dim,device=appareil).requires_grad_(True) for _ in range(num_blocs)]
ln1_b=[torch.zeros(dim,device=appareil).requires_grad_(True) for _ in range(num_blocs)]
ln2_g=[torch.ones(dim,device=appareil).requires_grad_(True) for _ in range(num_blocs)]
ln2_b=[torch.zeros(dim,device=appareil).requires_grad_(True) for _ in range(num_blocs)]
lnn_g=torch.ones(dim,device=appareil).requires_grad_(True)
lnn_b=torch.zeros(dim,device=appareil).requires_grad_(True)

W_out=(torch.randn(alph, dim,device=appareil)*1/math.sqrt(dim)).requires_grad_(True)
b_out=(torch.randn(alph,device=appareil)*0.1).requires_grad_(True)

params=[c,pos_emb]+W_q+W_k+W_v+W_o+W_1+b_1+W_2+b_2+ln1_g+ln1_b+ln2_g+ln2_b+[W_out,b_out,lnn_b,lnn_g]

m=[torch.zeros_like(p) for p in params]
v=[torch.zeros_like(p) for p in params]
beta1=0.9
beta2=0.999
eps=1e-8
t=1

def layernorm(x, g, b):
    mean=x.mean(dim=-1, keepdim=True)
    var=x.var(dim=-1, keepdim=True, unbiased=False)
    return (x-mean)/torch.sqrt(var+1e-5)*g+b
losstot=0
mask=torch.triu(torch.full((max_len,max_len), float('-inf'),device=appareil), diagonal=1)
pathlib.Path(chemin_dossier).mkdir(parents=True, exist_ok=True)
for i in range(30000):
    tireur=tireur_de_lot(tok, lot, max_len)
    input_indices, target_indices=tireur
    x=c[input_indices]
    x=x+pos_emb[:max_len]
    for bloc in range(num_blocs):
        y=layernorm(x,ln1_g[bloc], ln1_b[bloc])
        Q=y@W_q[bloc].T
        K=y@W_k[bloc].T
        V=y@W_v[bloc].T
        Q=Q.view(lot, max_len, num_heads, head_dim).transpose(1,2)
        K=K.view(lot, max_len, num_heads, head_dim).transpose(1,2)
        V=V.view(lot, max_len, num_heads, head_dim).transpose(1,2)
        attn=Q@K.transpose(-2,-1)/head_dim**0.5
        attn=attn+mask
        attn=torch.softmax(attn, dim=-1)
        out=attn@V
        out=out.transpose(1,2).reshape(lot, max_len, dim)
        out=out@W_o[bloc].T
        x=x+out
        z=layernorm(x, ln2_g[bloc], ln2_b[bloc])
        mlp=F.relu(z@W_1[bloc].T+b_1[bloc])
        mlp=mlp@W_2[bloc].T+b_2[bloc]
        x=x+mlp
    x=layernorm(x, lnn_g, lnn_b)
    logits=x@W_out.T+b_out
    targets=target_indices.to(torch.int64)
    loss=F.cross_entropy(logits.view(lot*max_len,alph), targets.view(lot*max_len,))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, max_norm=10)
    with torch.no_grad():
        for j,p in enumerate(params):
            m[j]=beta1*m[j]+(1-beta1)*p.grad
            v[j]=beta2*v[j]+(1-beta2)*p.grad**2
            m_hat=m[j]/(1-beta1**t)
            v_hat=v[j]/(1-beta2**t)
            p-=pas*m_hat/(v_hat**0.5+eps)
            p.grad=None
        t+=1
    losstot+=loss.item()
    i+=1
    if i%250==0:
        print(f"i={i}, loss={losstot/250:.4f},nombre de tokens vu = {lot*max_len*i}")
        losstot=0
    if i%1500==0:
        with torch.no_grad():
            losstot_val=0
            print("### Loss sur le jeu de validation ###")
            for k in range (10):
                tireur=tireur_de_lot(tok_val, lot, max_len)
                input_indices, target_indices=tireur
                x=c[input_indices]
                x=x+pos_emb[:max_len]
                for bloc in range(num_blocs):
                    y=layernorm(x, ln1_g[bloc], ln1_b[bloc])
                    Q=y@W_q[bloc].T
                    K=y@W_k[bloc].T
                    V=y@W_v[bloc].T
                    Q=Q.view(lot, max_len, num_heads, head_dim).transpose(1,2)
                    K=K.view(lot, max_len, num_heads, head_dim).transpose(1,2)
                    V=V.view(lot, max_len, num_heads, head_dim).transpose(1,2)
                    attn=Q@K.transpose(-2,-1)/head_dim**0.5
                    attn=attn+mask
                    attn=torch.softmax(attn, dim=-1)
                    out=attn@V
                    out=out.transpose(1,2).reshape(lot, max_len, dim)
                    out=out@W_o[bloc].T
                    x=x+out
                    z=layernorm(x, ln2_g[bloc], ln2_b[bloc])
                    mlp=F.relu(z@W_1[bloc].T+b_1[bloc])
                    mlp=mlp@W_2[bloc].T+b_2[bloc]
                    x=x+mlp
                x=layernorm(x, lnn_g, lnn_b)
                logits=x@W_out.T+b_out
                targets=target_indices.to(torch.int64)
                loss=F.cross_entropy(logits.view(lot*max_len,alph), targets.view(lot*max_len,))
                losstot_val+=loss.item()
            print(f"i={i}, loss={losstot_val/10:.4f},nombre de tokens vu = {lot*max_len*i}")
        print('### sauvegarde du model ###')
        sauvegarde = {
            'seed' : seed,
            'nombre de tokens vu' : lot*max_len*i,
            'i' : i,
            'loss' : losstot_val/10,
            'pas' : pas,
            'c': c.detach(), 'pos_emb': pos_emb.detach(),
            'W_q': [p.detach() for p in W_q],
            'W_k': [p.detach() for p in W_k],
            'W_v': [p.detach() for p in W_v],
            'W_o': [p.detach() for p in W_o],
            'W_1': [p.detach() for p in W_1],
            'b_1': [p.detach() for p in b_1],
            'W_2': [p.detach() for p in W_2],
            'b_2': [p.detach() for p in b_2],
            'ln1_g': [p.detach() for p in ln1_g],
            'ln1_b': [p.detach() for p in ln1_b],
            'ln2_g': [p.detach() for p in ln2_g],
            'ln2_b': [p.detach() for p in ln2_b],
            'lnn_g':lnn_g.detach(),
            'lnn_b':lnn_b.detach(),
            'W_out': W_out.detach(), 'b_out': b_out.detach(),
            'm': [p.detach() for p in m],
            'v': [p.detach() for p in v],
            't': t,
            'alph': alph, 'dim': dim, 'num_heads': num_heads,
            'head_dim': head_dim, 'max_len': max_len, 'num_blocs': num_blocs,
        }
        torch.save(sauvegarde, f"{chemin_dossier}/{date}-sauvegarde_{i}.pt")




"""
with torch.no_grad():
    tok2=open('data/encode_tok_val.npy')
    alph2=tireur_de_lot(tok2, 32, 256)
    j=0
    losstot2=0
    for name in tok2:
        input_indices=[]
        target_indices=[]
        prev=' '
        for lettre in name:
            if lettre=='\n':
                lettre=' '
            input_indices.append(alph2(prev))
            target_indices.append(alph2(lettre))
            prev=lettre
        T=len(input_indices)
        if T>max_len:
            continue
        x=c[input_indices]
        x=x+pos_emb[:T]
        for bloc in range(num_blocs):
            y=layernorm(x, ln1_g[bloc], ln1_b[bloc])
            Q=y@W_q[bloc].T
            K=y@W_k[bloc].T
            V=y@W_v[bloc].T
            Q=Q.view(T, num_heads, head_dim).transpose(0,1)
            K=K.view(T, num_heads, head_dim).transpose(0,1)
            V=V.view(T, num_heads, head_dim).transpose(0,1)
            attn=Q@K.transpose(-2,-1)/head_dim**0.5
            mask=torch.triu(torch.full((T,T), float('-inf')), diagonal=1)
            attn=attn+mask
            attn=torch.softmax(attn, dim=-1)
            out=attn@V
            out=out.transpose(0,1).reshape(T, dim)
            out=out@W_o[bloc].T
            x=x+out
            z=layernorm(x, ln2_g[bloc], ln2_b[bloc])
            mlp=F.relu(z@W_1[bloc].T+b_1[bloc])
            mlp=mlp@W_2[bloc].T+b_2[bloc]
            x=x+mlp
        logits=x@W_out.T+b_out
        targets=target_indices.to(torch.int64)
        loss=F.cross_entropy(logits, targets)
        losstot2+=loss.item()
        j+=1
    print(f"  val_loss={losstot2/j:.4f}")
"""
def genere_prenom(max_len=max_len):
    seq=[alphabet.index('\n')]
    for _ in range(max_len):
        T=len(seq)
        x=c[seq]
        x=x+pos_emb[:T]

        for bloc in range(num_blocs):
            y=layernorm(x, ln1_g[bloc], ln1_b[bloc])
            Q=y@W_q[bloc].T
            K=y@W_k[bloc].T
            V=y@W_v[bloc].T
            Q=Q.view(T, num_heads, head_dim).transpose(0,1)
            K=K.view(T, num_heads, head_dim).transpose(0,1)
            V=V.view(T, num_heads, head_dim).transpose(0,1)
            attn=Q@K.transpose(-2,-1)/head_dim**0.5
            mask=torch.triu(torch.full((T,T), float('-inf'),device=appareil), diagonal=1)
            attn=attn+mask
            attn=torch.softmax(attn, dim=-1)
            out=attn@V
            out=out.transpose(0,1).reshape(T, dim)
            out=out@W_o[bloc].T
            x=x+out
            z=layernorm(x, ln2_g[bloc], ln2_b[bloc])
            mlp=F.relu(z@W_1[bloc].T+b_1[bloc])
            mlp=mlp@W_2[bloc].T+b_2[bloc]
            x=x+mlp
        x=layernorm(x, lnn_g, lnn_b)
        logits=x@W_out.T+b_out
        y=torch.softmax(logits[-1], dim=0).detach()
        idx=torch.multinomial(y, 1).item()
        lettre=alphabet[idx]
        if lettre=='\n':
            break
        seq.append(idx)
    return ''.join(alphabet[i] for i in seq[1:])

print("\nGénération de texte :")
for _ in range(20):
    print(genere_prenom())