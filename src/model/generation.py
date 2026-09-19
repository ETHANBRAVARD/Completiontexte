import torch
import torch.nn.functional as F
import json
from pathlib import Path
import datetime
from random import randint
import hashlib
import math
cheminbpe="data/tokenizer/bpe.json"
date=format(datetime.datetime.now(), '%Y%m%d-%H%M%S')

def encode(amorce):
    cheminbpe="data/tokenizer/bpe.json"
    with open(cheminbpe, 'r', encoding='utf-8') as f:
        bpe = json.load(f)
    bpeset=set(bpe['alphabet'])
    encode = amorce
    séparateur=['[',']']
    if séparateur[0] in bpe['alphabet'] or séparateur[1] in bpe['alphabet']:
        print('change de séparateur')
        exit()
    encode2=[séparateur[0]+' '+séparateur[1]]
    lettre_non_reconnu=set()
    for lettre in encode:
        if lettre not in bpeset :
            lettre_non_reconnu.add(lettre)
        encode2.append(séparateur[0]+lettre+séparateur[1])
    encode2=''.join(encode2)
    if lettre_non_reconnu:
        print('lettres non reconnues :',lettre_non_reconnu)
        exit()
    for fusion in bpe['fusions']: 
        encode2 =encode2.replace('['+fusion[0]+']'+'['+fusion[1]+']', '['+fusion[0]+fusion[1]+']')
    encode3=[]
    i=0
    for car in encode2:
        if car == '[':
            j=1
            while encode2[i+j] != ']':
                j+=1
            encode3.append(encode2[i+1:i+j])
        i+=1
    retour={'encode':encode3}
    return(retour)

def rencode(amorce,alphabet):
    encodelist=amorce
    i=0
    dictbpe={}
    for k in alphabet:
        dictbpe[k]=i
        i+=1
    encode_tok=[]
    for i in range (len(amorce['encode'])):
        encode_tok.append(dictbpe[amorce['encode'][i]])
    return(encode_tok)

def layernorm(x, g, b):
    mean=x.mean(dim=-1, keepdim=True)
    var=x.var(dim=-1, keepdim=True, unbiased=False)
    return (x-mean)/torch.sqrt(var+1e-5)*g+b


def genere(checkpoint,nombre_de_car,amorce,mode="topk",temp=1.2,k=5,p=0.9,seed=False): #mode= 'greedy' ou 'topk' ou 'topp'
    if seed:
        torch.manual_seed(seed)
    appareil=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sauvegarde=checkpoint
    seed=1337
    chemin="data/tokenizer/bpe_liste.json"
    with open(chemin,'r') as f:
        alphabet=json.load(f)
    alph=len(alphabet)
    max_len=sauvegarde['max_len']
    c=sauvegarde['c']
    pos_emb=sauvegarde['pos_emb']
    num_heads=sauvegarde['num_heads']
    num_blocs=sauvegarde['num_blocs']
    W_q=sauvegarde['W_q']
    W_k=sauvegarde['W_k']
    W_v=sauvegarde['W_v']
    W_o=sauvegarde['W_o']
    W_1=sauvegarde['W_1']
    b_1=sauvegarde['b_1']
    W_2=sauvegarde['W_2']
    b_2=sauvegarde['b_2']
    ln1_g=sauvegarde['ln1_g']
    ln1_b=sauvegarde['ln1_b']
    ln2_g=sauvegarde['ln2_g']
    ln2_b=sauvegarde['ln2_b']
    lnn_b=sauvegarde['lnn_b']
    lnn_g=sauvegarde['lnn_g']
    W_out=sauvegarde['W_out']
    dim=sauvegarde['dim']
    b_out=sauvegarde['b_out']
    fiche_car={'nombre de tokens vu':sauvegarde['nombre de tokens vu'],'seed':sauvegarde['seed'],
'i':sauvegarde['i'],
'loss':sauvegarde['loss'],
'pas' :sauvegarde['pas']}
    head_dim=dim//num_heads
    amorce=rencode(encode(amorce),alphabet)
    if nombre_de_car>max_len:
        nombre_de_car=max_len
    with torch.no_grad():
        seq=amorce
        for _ in range(nombre_de_car):
            T=len(seq)
            x=c[seq]
            x=x+pos_emb[:T]
            mask=torch.triu(torch.full((T,T), float('-inf'),device=appareil), diagonal=1)
            for bloc in range(num_blocs):
                y=layernorm(x, ln1_g[bloc], ln1_b[bloc])
                Q=y@W_q[bloc].T
                K=y@W_k[bloc].T
                V=y@W_v[bloc].T
                Q=Q.view(T, num_heads, head_dim).transpose(0,1)
                K=K.view(T, num_heads, head_dim).transpose(0,1)
                V=V.view(T, num_heads, head_dim).transpose(0,1)
                attn=Q@K.transpose(-2,-1)/head_dim**0.5
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
            logits/=temp
            y=torch.softmax(logits[-1], dim=0).detach()
            y2=torch.sort(y,descending=True)
            if mode =='greedy':
                idx=torch.argmax(y).item()
            if mode == 'topk':
                y=y[y2[1][:k]] /torch.sum(y[y2[1][:k]] ).detach()
                z=torch.multinomial(y, 1)
                idx=y2[1][z].item()
            if mode == 'topp':
                ybis=torch.cumsum(y2[0],0)
                i=0
                while ybis[i]<=p:
                    i+=1
                y = y[y2[1][:i+1]] / torch.sum(y[y2[1][:i+1]])  
                z = torch.multinomial(y, 1)
                idx = y2[1][z].item()
            lettre=alphabet[idx]
            if lettre=='\n':
                break
            seq.append(idx)
        print(f"i:{fiche_car['i']},loss:{fiche_car['loss']},seed:{fiche_car['seed']}")
        return ''.join(alphabet[i] for i in seq)

def graine(seed,cpt):
    hash=f'{seed}'+'_'+f'{cpt}'
    return(int(hashlib.sha256(hash.encode()).hexdigest()[:16],16))
if __name__ == '__main__':
    cpt=0
    topk=[]
    topp=[]
    greedy=[]
    dossier = max((d for d in Path('runs/save_runs').iterdir() if d.is_dir()),
              key=lambda d: d.name)
    saveway = str(max(dossier.glob('*-sauvegarde_*.pt'),
                  key=lambda f: int(f.stem.split('_')[-1])))
    print('checkpoint :', saveway)
    nombre_de_car=100
    p = Path(saveway)
    checkpoint=torch.load(saveway)
    numero_i=checkpoint['i']
    nom=f"runs/generation/generation{p.parent.name}_{date}_{numero_i}.json"
    amorceliste = [
    'Once upon a time',
    'One time in a castle',
    'One day i will',
    'Once ',
    'The',
    'Suddenly,',
    'Tim was',
    'Lucy and her mom',
    'and then she',
    'but the little boy did not',
    'He wanted to',
    'It was a very hot day and',
    'Why did the dog',
    'What is inside the',
    '"Hello," said',
    '"I am scared," she',
    "\"That's mine!\" shouted",
    'One rainy morning, the little girl found a',
    'In the big forest, there was a tiny',
    'The moral of the story is',
]
    seed=randint(0,10**10)
    taille=len(amorceliste)
    nb=25 * taille
    grille_k = [3, 5, 10, 20, 40, 80]
    grille_p = [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]
    grille_temperature = [j * 0.4 for j in range(1, 5)]

    with open(nom, "x", encoding="utf-8") as f:
        for e in range (nb):
            topk.append([])
            topp.append([])
            amorce=amorceliste[e%taille]
            for i in grille_k:
                for j in grille_temperature:
                    temp=j
                    cpt+=1
                    seed2=graine(seed,cpt)
                    topk[-1].append({'texte':genere(checkpoint,nombre_de_car,amorce,mode='topk',temp=temp,k=i,seed=seed2),'amorce':amorce,'k':i,'temperature':temp,'cpt':cpt})
            for i in grille_p:
                    for j in grille_temperature:
                        cpt+=1
                        seed2=graine(seed,cpt)
                        temp=j
                        topp[-1].append({'texte':genere(checkpoint,nombre_de_car,amorce,mode='topp',temp=temp,p=i,seed=seed2),'amorce':amorce,'p':i,'temperature':temp,'cpt':cpt})
        for i in range(taille):
            cpt+=1
            seed2=graine(seed,cpt)
            greedy.append({'texte':genere(checkpoint,nombre_de_car,amorceliste[i],mode='greedy',temp=1.0,seed=seed2),'amorce':amorceliste[i],'cpt':cpt})
        POIDS = ['c', 'pos_emb', 'lnn_g', 'lnn_b', 'W_out', 'b_out',
                 'W_q', 'W_k', 'W_v', 'W_o', 'W_1', 'b_1', 'W_2', 'b_2',
                 'ln1_g', 'ln1_b', 'ln2_g', 'ln2_b']
        parametres = sum(t.numel() for k in POIDS
                         for t in (checkpoint[k] if isinstance(checkpoint[k], list)
                                   else [checkpoint[k]]))

        provenance = {
            'date_generation'      : date,
            'checkpoint'           : saveway,
            'pas_entrainement'     : checkpoint['i'],
            'loss_validation'      : checkpoint['loss'],
            'perplexite_validation': math.exp(checkpoint['loss']),          
            'parametres'           : parametres,   
            'tokens_vus'           : checkpoint['nombre de tokens vu'],
            'config_pas'   : checkpoint['pas'],
            'seed_entrainement'    : checkpoint['seed'],
            'dim'                  : checkpoint['dim'],
            'num_blocs'            : checkpoint['num_blocs'],
            'num_heads'            : checkpoint['num_heads'],
            'max_len'              : checkpoint['max_len'],
            'vocabulaire'          : checkpoint['alph'],
            'tokenizer'            : 'data/tokenizer/bpe_liste.json',
            'amorces'              : amorceliste,
            'textes_par_amorce'    : nb // taille,
            'nombre_de_car'        : nombre_de_car,
            'grille_k'             : grille_k,
            'grille_p'             : grille_p,
            'grille_temperature'   : grille_temperature,
            'seed_base'            : seed,
            'regle_graine'         : 'sha256("<seed_base>_<cpt>") hexdigest[:16] en base 16',
            'appels'               : cpt,
        }
        retour = {'topk': topk, 'topp': topp, 'greedy': greedy, 'provenance': provenance}
        json.dump(retour, f, ensure_ascii=False)


