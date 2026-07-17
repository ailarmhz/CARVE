#!/usr/bin/env python3
"""Integrated Table 1: single-seed vs seed-ensemble, z=40 vs z=240, AUROC + F1.
Reads the results_matrix_* dirs (real varying seeds)."""
import torch, numpy as np, glob, os, csv as _csv

def auroc(y,s):
    npos=int((y==1).sum());nneg=len(y)-npos
    if npos==0 or nneg==0:return np.nan
    o=np.argsort(s);ss=s[o];r=np.empty(len(s));i=0;rk=1.0
    while i<len(ss):
        j=i+1
        while j<len(ss) and ss[j]==ss[i]:j+=1
        r[o[i:j]]=(rk+rk+(j-i)-1)/2;rk+=j-i;i=j
    return (r[y==1].sum()-npos*(npos+1)/2)/(npos*nneg)
def mac(pr,y):return float(np.nanmean([auroc(y[:,j],pr[:,j]) for j in range(y.shape[1])]))
def f1(pr,y,t=0.5):
    fs=[]
    for j in range(y.shape[1]):
        p=(pr[:,j]>t).astype(int);yy=y[:,j]
        tp=((p==1)&(yy==1)).sum();fp=((p==1)&(yy==0)).sum();fn=((p==0)&(yy==1)).sum()
        d=2*tp+fp+fn;fs.append(2*tp/d if d>0 else np.nan)
    return float(np.nanmean(fs))
def cell(d,ds,m):
    ps=[p for p in sorted(glob.glob(f"{d}/{ds}_{m}_seed*.pt")) if "_sh" not in os.path.basename(p)]
    if not ps:return None
    prs=[];y=None
    for p in ps:
        o=torch.load(p,map_location="cpu",weights_only=False);prs.append(np.asarray(o["probs"],float));y=np.asarray(o["y_true"],int)
    p0=prs[0]; pe=np.mean(prs,0)
    return dict(A1=mac(p0,y),Ae=mac(pe,y),F1=f1(p0,y),Fe=f1(pe,y),n=len(ps),sd=float(np.mean([mac(p,y) for p in prs])-mac(p0,y)))

SETTINGS=[
 ("RAD-ChestCT (16)","radchest",[(v,"results_matrix_radchest_VARIANT_zDEPTH") for v in ("zeroshot","vocabfine","classfine")]),
 ("CC-CCII (3)","ccii",[("zeroshot","results_matrix_ccii_VARIANT_zDEPTH"),
                        ("vocabfine","results_matrix_ccii_VARIANT_zDEPTH"),
                        ("classfine","results_matrix_ccii_VARIANT_zDEPTH")]),
 ("LUNA16 (binary)","luna",[("zeroshot","results_matrix_luna_VARIANT_zDEPTH"),
                            ("vocabfine","results_matrix_luna_VARIANT_zDEPTH"),
                            ("classfine","results_matrix_luna_VARIANT_zDEPTH")]),
]
METH=["zeroshot","tent","ml_tta","bem","carve_xview","carve_xview_gate"]
rows=[]
def dirfor(tmpl,variant,depth): return tmpl.replace("VARIANT",variant).replace("DEPTH",str(depth))
for sname,ds,variants in SETTINGS:
    for variant,tmpl in variants:
        print(f"\n############ {sname}  [{variant}] ############")
        print(f"{'method':16}| AUROC z40 s/ens | AUROC z240 s/ens | F1 z40 s/ens | F1 z240 s/ens | n40/n240")
        for m in METH:
            c40=cell(dirfor(tmpl,variant,40),ds,m); c240=cell(dirfor(tmpl,variant,240),ds,m)
            def g(c,k): return f"{c[k]:.4f}" if c else "  -   "
            print(f"{m:16}| {g(c40,'A1')}/{g(c40,'Ae')} | {g(c240,'A1')}/{g(c240,'Ae')} | "
                  f"{g(c40,'F1')}/{g(c40,'Fe')} | {g(c240,'F1')}/{g(c240,'Fe')} | "
                  f"{c40['n'] if c40 else 0}/{c240['n'] if c240 else 0}")
            rows.append(dict(setting=sname,variant=variant,method=m,
                AUROC_z40_single=g(c40,'A1'),AUROC_z40_ens=g(c40,'Ae'),
                AUROC_z240_single=g(c240,'A1'),AUROC_z240_ens=g(c240,'Ae'),
                F1_z40_single=g(c40,'F1'),F1_z40_ens=g(c40,'Fe'),
                F1_z240_single=g(c240,'F1'),F1_z240_ens=g(c240,'Fe'),
                n40=c40['n'] if c40 else 0,n240=c240['n'] if c240 else 0))
# fVLM
print(f"\n############ RAD-ChestCT (fVLM) ############")
print(f"{'method':16}| AUROC z240 s/ens | F1 z240 s/ens | n")
for m in METH:
    c=cell("results_matrix_fvlm_radchest_z240","radchest",m)
    def g(c,k): return f"{c[k]:.4f}" if c else "  -   "
    print(f"{m:16}| {g(c,'A1')}/{g(c,'Ae')} | {g(c,'F1')}/{g(c,'Fe')} | {c['n'] if c else 0}")
    rows.append(dict(setting="RAD-ChestCT (fVLM)",variant="zeroshot",method=m,
        AUROC_z40_single="",AUROC_z40_ens="",AUROC_z240_single=g(c,'A1'),AUROC_z240_ens=g(c,'Ae'),
        F1_z40_single="",F1_z40_ens="",F1_z240_single=g(c,'F1'),F1_z240_ens=g(c,'Fe'),n40=0,n240=c['n'] if c else 0))
if rows:
    os.makedirs("results",exist_ok=True)
    with open("results/integrated_table.csv","w",newline="") as f:
        w=_csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
    print(f"\nwrote results/integrated_table.csv ({len(rows)} rows)")
