import json
import re
from collections import defaultdict,Counter
import sys
sys.path.insert(0, 'src/tooling')
from tracer import taux, table, sauver, tracer_grille, tracer_carte, tracer_compromis
from importlib.metadata import version
from pathlib import Path

VERSIONS = {paquet: version(paquet) for paquet in ('torch', 'numpy', 'matplotlib')}




cheminvocab = "data/vocabulaire.json"
chemin   = max(Path('runs/generation').glob('generation*.json'), key=lambda f: f.name)
date_run = chemin.stem.removeprefix('generation')
print('analyse de :', chemin.name)
with open(chemin, 'r', encoding='utf-8') as f:
    gen=json.load(f)
with open(cheminvocab, 'r', encoding='utf-8') as f:
    vocab=set(json.load(f))
liste_mot=[]
cpt=defaultdict(int)
erreur=defaultdict(list)
cptot=defaultdict(int)
for liste in gen['topk']:
    for sous_liste in liste:
        k=sous_liste['k']
        temp=sous_liste['temperature']
        texte=sous_liste['texte']
        temp = round(sous_liste['temperature'], 1)
        liste_mot=re.findall(r"[a-zA-Z][a-zA-Z']*", texte.lower())
        cptot[k,temp]+=len(liste_mot)
        for mot in liste_mot:
            if mot not in vocab:
                cpt[k,temp]+=1
                erreur[k,temp].append(mot)
nom       = f'inexistants_topk_{date_run}'
inexistants_k = taux(cpt, cptot)

table(inexistants_k, 'k')
sauver(inexistants_k, f'runs/generation/mesures_{nom}.json',
       metrique='mots inexistants',
       checkpoint=chemin.name,
       versions=VERSIONS)

sous_titre = 'top-k · 100 répétitions · 4 amorces · 8,6 M paramètres · perplexité 5,02'
tracer_carte(inexistants_k,   f'runs/generation/carte_{nom}.png',   'k', sous_titre=sous_titre)
tracer_grille(inexistants_k,  f'runs/generation/courbes_{nom}.png', 'k', sous_titre=sous_titre)
liste_mot_p=[]
cpt_p=defaultdict(int)
erreur_p=defaultdict(list)
cptot_p=defaultdict(int)
for liste in gen['topp']:
    for sous_liste in liste:
        p=sous_liste['p']
        texte=sous_liste['texte']
        temp = round(sous_liste['temperature'], 1)
        liste_mot_p=re.findall(r"[a-zA-Z][a-zA-Z']*", texte.lower())
        cptot_p[p,temp]+=len(liste_mot_p)
        for mot in liste_mot_p:
            if mot not in vocab:
                cpt_p[p,temp]+=1
                erreur_p[p,temp].append(mot)
nom_p     = f'inexistants_topp_{date_run}'
inexistants_p = taux(cpt_p, cptot_p)

table(inexistants_p, 'p')
sauver(inexistants_p, f'runs/generation/mesures_{nom_p}.json',
       metrique='mots inexistants',
       checkpoint=chemin.name,
       versions=VERSIONS)
sous_titre_p = 'top-p · 100 répétitions · 4 amorces · 8,6 M paramètres · perplexité 5,02'
tracer_carte(inexistants_p,   f'runs/generation/carte_{nom_p}.png',   'p', sous_titre=sous_titre_p)
tracer_grille(inexistants_p,  f'runs/generation/courbes_{nom_p}.png', 'p', sous_titre=sous_titre_p)

rep=defaultdict(int)
tot=defaultdict(int)
for liste in gen['topk']:
    for sous_liste in liste:
        m=re.findall(r"[a-zA-Z][a-zA-Z']*", sous_liste['texte'].lower())
        grams=[tuple(m[i:i+4]) for i in range(len(m)-3)]
        c=Counter(grams)
        cle=(sous_liste['k'], round(sous_liste['temperature'],1))
        tot[cle]+=len(grams)
        rep[cle]+=sum(n-1 for n in c.values() if n>1)

rep2=defaultdict(int)
tot2=defaultdict(int)
for liste in gen['topp']:
    for sous_liste in liste:
        m=re.findall(r"[a-zA-Z][a-zA-Z']*", sous_liste['texte'].lower())
        grams=[tuple(m[i:i+4]) for i in range(len(m)-3)]
        c=Counter(grams)
        cle=(sous_liste['p'], round(sous_liste['temperature'],1))
        tot2[cle]+=len(grams)
        rep2[cle]+=sum(n-1 for n in c.values() if n>1)

repetition_k=taux(rep,tot)
repetition_p=taux(rep2,tot2)
print(' répétition - top-k')
table(repetition_k,nom_parametre="k")
print(' répétition - top-p')
table(repetition_p,nom_parametre="p")
nom_rep   = f'repetition_topk_{date_run}'
nom_rep_p = f'repetition_topp_{date_run}'

sauver(repetition_k, f'runs/generation/mesures_{nom_rep}.json',
       metrique='répétition de 4-grammes',
       checkpoint=chemin.name,
       versions=VERSIONS)
sauver(repetition_p, f'runs/generation/mesures_{nom_rep_p}.json',
       metrique='répétition de 4-grammes',
       checkpoint=chemin.name,
       versions=VERSIONS)

titre_rep = 'Taux de répétition de 4-grammes'

tracer_carte(repetition_k,  f'runs/generation/carte_{nom_rep}.png',   'k',
             titre=titre_rep, sous_titre=sous_titre)
tracer_grille(repetition_k, f'runs/generation/courbes_{nom_rep}.png', 'k',
              titre=titre_rep, sous_titre=sous_titre, y_label='répétition (%)')

tracer_carte(repetition_p,  f'runs/generation/carte_{nom_rep_p}.png',   'p',
             titre=titre_rep, sous_titre=sous_titre_p)
tracer_grille(repetition_p, f'runs/generation/courbes_{nom_rep_p}.png', 'p',
              titre=titre_rep, sous_titre=sous_titre_p, y_label='répétition (%)')

tracer_compromis(repetition_k, inexistants_k,
                 f'runs/generation/compromis_topk_{date_run}.png', 'k',
                 sous_titre=sous_titre)
tracer_compromis(repetition_p, inexistants_p,
                 f'runs/generation/compromis_topp_{date_run}.png', 'p',
                 sous_titre=sous_titre_p)

sacs = defaultdict(Counter)

for liste in gen['topk']:
    for sous_liste in liste:
        amorce=sous_liste['amorce']
        k=sous_liste['k']
        m=re.findall(r"[a-zA-Z][a-zA-Z']*", sous_liste['texte'].lower())
        grams=[tuple(m[i:i+4]) for i in range(len(m)-3)]
        sacs[amorce,k,round(sous_liste['temperature'],1)].update(grams)
distincts={}
total={}
paquets = defaultdict(list)
for cle in sacs:
    distincts[cle]=(len(sacs[cle]))
    total[cle]=sum(sacs[cle].values())
tau=taux(distincts,total)
for (etiquette, p1, p2), valeur in tau.items():
    paquets[p1, p2].append(valeur)
diversite_k = {cle: sum(v) / len(v) for cle, v in paquets.items()}

sacs = defaultdict(Counter)
for liste in gen['topp']:
    for sous_liste in liste:
        amorce=sous_liste['amorce']
        p=sous_liste['p']
        m=re.findall(r"[a-zA-Z][a-zA-Z']*", sous_liste['texte'].lower())
        grams=[tuple(m[i:i+4]) for i in range(len(m)-3)]
        sacs[amorce,p,round(sous_liste['temperature'],1)].update(grams)
distincts={}
total={}
paquets = defaultdict(list)
for cle in sacs:
    distincts[cle]=(len(sacs[cle]))
    total[cle]=sum(sacs[cle].values())
tau=taux(distincts,total)
for (etiquette, p1, p2), valeur in tau.items():
    paquets[p1, p2].append(valeur)
diversite_p = {cle: sum(v) / len(v) for cle, v in paquets.items()}

nom_div   = f'diversite_topk_{date_run}'
nom_div_p = f'diversite_topp_{date_run}'

sauver(diversite_k, f'runs/generation/mesures_{nom_div}.json',
       metrique='diversité de 4-grammes entre textes de même amorce',
       amorces=4, textes_par_groupe=25,
       checkpoint=chemin.name,
       versions=VERSIONS)

sauver(diversite_p, f'runs/generation/mesures_{nom_div_p}.json',
       metrique='diversité de 4-grammes entre textes de même amorce',
       amorces=4, textes_par_groupe=25,
       checkpoint=chemin.name,
       versions=VERSIONS)


    
    
