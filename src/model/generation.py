import torch
import torch.nn.functional as F
import json
from pathlib import Path
cheminbpe="data/tokenizer/bpe.json"

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


def genere(saveway,nombre_de_car,mode,temp,amorce,k=3,p=0.9,seed=False): #mode= 'greedy' ou 'topk' ou 'topp'
    if seed:
        torch.manual_seed(seed)
    appareil=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sauvegarde=torch.load(saveway)
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
    head_dim=sauvegarde['head_dim']
    dim=sauvegarde['dim']
    b_out=sauvegarde['b_out']
    fiche_car={'nombre de tokens vu':sauvegarde['nombre de tokens vu'],'seed':sauvegarde['seed'],
'm':sauvegarde['m'],
'v':sauvegarde['v'],
't':sauvegarde['t'],
'i':sauvegarde['i'],
'loss':sauvegarde['loss'],
'pas' :sauvegarde['pas']}
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
        return ''.join(alphabet[i] for i in seq[1:])

if __name__ == '__main__':
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
    nb=100
    for e in range (nb):
        topk.append([])
        topp.append([])
        if e<nb/4:
            amorce='Once upon a time'
        elif e<nb/2:
            amorce='One time in a castle'
        elif e<3*nb/4:
            amorce='One day i will'
        else:
            amorce = 'Once '
        for i in [3, 5, 10, 20, 40, 80]:
            for j in range (1,5):
                temp=j*0.4
                topk[-1].append({'texte':genere(saveway,nombre_de_car,'topk',temp,amorce,i),'amorce':amorce,'k':i,'temperature':temp})
        for i in [0.2, 0.4, 0.5, 0.6, 0.8, 0.9]:
                for j in range (1,5):
                    temp=j*0.4
                    topp[-1].append({'texte':genere(saveway,nombre_de_car,'topp',temp, amorce ,i,i),'amorce':amorce,'p':i,'temperature':temp})
        if e%(nb/4)==0:
            greedy.append({'texte':genere(saveway,nombre_de_car,'greedy',temp,amorce),'amorce':amorce})
    retour={'topk':topk,'topp':topp,'greedy':greedy}
    with open(f"runs/generation/generation{p.parent.name}.json", "w", encoding="utf-8") as f:
          json.dump(retour, f,ensure_ascii=False)
