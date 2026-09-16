import random
import torch
import torch.nn.functional as F
from tireur_de_lot import tireur_de_lot


def layernorm(x, g, b):
    mean=x.mean(dim=-1, keepdim=True)
    var=x.var(dim=-1, keepdim=True, unbiased=False)
    return (x-mean)/torch.sqrt(var+1e-5)*g+b



def loss_validation(model,lot,tok_val,appareil):
    max_len=model['max_len']
    num_blocs=model['num_blocs']
    num_heads=model['num_heads']
    head_dim=model['head_dim']
    dim=model['dim']
    alph=model['alph']
    c=model['c']
    pos_emb=model['pos_emb']
    ln1_g=model['ln1_g']
    ln1_b=model['ln1_b']
    ln2_g=model['ln2_g']
    ln2_b=model['ln2_b']
    lnn_g=model['lnn_g']
    lnn_b=model['lnn_b']
    W_q=model['W_q']
    W_k=model['W_k']
    W_v=model['W_v']
    W_o=model['W_o']
    W_1=model['W_1']
    b_1=model['b_1']
    W_2=model['W_2']
    b_2=model['b_2']
    W_out=model['W_out']
    b_out=model['b_out']
    mask=torch.triu(torch.full((max_len,max_len), float('-inf'),device=appareil), diagonal=1)
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
    return(losstot_val/10)

def bruit_mult(model,alpha):
    model2=model.copy()
    model2['W_q']=[]
    model2['W_k']=[]
    model2['W_o']=[]
    model2['W_1']=[]
    model2['W_2']=[]
    model2['W_v']=[]
    for i in range(len(model['W_q'])):
        model2['W_q'].append(model['W_q'][i]*torch.randn_like(model['W_q'][i])*alpha+model['W_q'][i])
    for i in range(len(model['W_k'])):
        model2['W_k'].append(model['W_k'][i]*torch.randn_like(model['W_k'][i])*alpha+model['W_k'][i])
    for i in range(len(model['W_v'])):
        model2['W_v'].append(model['W_v'][i]*torch.randn_like(model['W_v'][i])*alpha+model['W_v'][i])
    for i in range(len(model['W_o'])):
        model2['W_o'].append(model['W_o'][i]*torch.randn_like(model['W_o'][i])*alpha+model['W_o'][i])
    for i in range(len(model['W_1'])):
        model2['W_1'].append(model['W_1'][i]*torch.randn_like(model['W_1'][i])*alpha+model['W_1'][i])
    for i in range(len(model['W_2'])):
        model2['W_2'].append(model['W_2'][i]*torch.randn_like(model['W_2'][i])*alpha+model['W_2'][i])
    model2['W_out']=model['W_out']*torch.randn_like(model['W_out'])*alpha+model2['W_out']
    return(model2)

def bruit_add(model,alpha):
    model2=model.copy()
    model2['W_q']=[]
    model2['W_k']=[]
    model2['W_o']=[]
    model2['W_1']=[]
    model2['W_2']=[]
    model2['W_v']=[]
    for i in range(len(model['W_q'])):
        maximum=model['W_q'][i].abs().max()
        model2['W_q'].append(torch.randn_like(model['W_q'][i])*alpha*maximum+model['W_q'][i])
    for i in range(len(model['W_k'])):
        maximum=model['W_k'][i].abs().max()
        model2['W_k'].append(torch.randn_like(model['W_k'][i])*alpha*maximum+model['W_k'][i])
    for i in range(len(model['W_v'])):
        maximum=model['W_v'][i].abs().max()
        model2['W_v'].append(torch.randn_like(model['W_v'][i])*alpha*maximum+model['W_v'][i])
    for i in range(len(model['W_o'])):
        maximum=model['W_o'][i].abs().max()
        model2['W_o'].append(torch.randn_like(model['W_o'][i])*alpha*maximum+model['W_o'][i])
    for i in range(len(model['W_1'])):
        maximum=model['W_1'][i].abs().max()
        model2['W_1'].append(torch.randn_like(model['W_1'][i])*alpha*maximum+model['W_1'][i])
    for i in range(len(model['W_2'])):
        maximum=model['W_2'][i].abs().max()
        model2['W_2'].append(torch.randn_like(model['W_2'][i])*alpha*maximum+model['W_2'][i])
    maximum=model['W_out'].abs().max()
    model2['W_out']=model2['W_out']+torch.randn_like(model['W_out'])*alpha*maximum
    return(model2)

def bruit_mult_unique_list (W,alpha):
    W2=[]
    for i in range(len(W)):
        W2.append(W[i]*torch.randn_like(W[i])*alpha+W[i])
    return(W2)

def bruit_add_unique_list(W,alpha):
    maximum=0
    W2=[]
    for i in range(len(W)):
        maximum=W[i].abs().max()
        W2.append(torch.randn_like(W[i])*alpha*maximum+W[i])
    return(W2)

def bruit_mult_unique (W,alpha):
    W2=[]
    W2.append(W*torch.randn_like(W)*alpha+W)
    return(W2)

def bruit_add_unique(W,alpha):
    maximum=0
    W2=[]
    maximum=W.abs().max()
    W2.append(torch.randn_like(W)*alpha*maximum+W)
    return(W2)