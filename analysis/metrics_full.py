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
def auprc(y,s):
    npos=int((y==1).sum())
    if npos==0:return np.nan
    o=np.argsort(-s);yy=y[o];tp=np.cumsum(yy);fp=np.cumsum(1-yy)
    prec=tp/np.maximum(tp+fp,1);rec=tp/npos
    rprev=np.concatenate([[0],rec[:-1]])
    return float(np.sum((rec-rprev)*prec))
def perlabel(pr,y,fn):return float(np.nanmean([fn(y[:,j],pr[:,j]) for j in range(y.shape[1])]))
def prf(pr,y,t=0.5):
    P=[];R=[];F=[]
    for j in range(y.shape[1]):
        p=(pr[:,j]>t).astype(int);yy=y[:,j]
        tp=((p==1)&(yy==1)).sum();fp=((p==1)&(yy==0)).sum();fn=((p==0)&(yy==1)).sum()
        prec=tp/(tp+fp) if tp+fp>0 else np.nan
        rec=tp/(tp+fn) if tp+fn>0 else np.nan
        f1=2*tp/(2*tp+fp+fn) if (2*tp+fp+fn)>0 else np.nan
        P.append(prec);R.append(rec);F.append(f1)
    return float(np.nanmean(P)),float(np.nanmean(R)),float(np.nanmean(F))
def allm(pr,y):
    P,R,F=prf(pr,y)
    return dict(AUROC=perlabel(pr,y,auroc),AUPRC=perlabel(pr,y,auprc),F1=F,Prec=P,Rec=R,
                Acc=float(((pr>0.5).astype(int)==y).mean()))
def cell(d,ds,m):
    ps=[p for p in sorted(glob.glob(f"{d}/{ds}_{m}_seed*.pt")) if "_sh" not in os.path.basename(p)]
    if not ps:return None
    prs=[];y=None
    for p in ps:
        o=torch.load(p,map_location="cpu",weights_only=False);prs.append(np.asarray(o["probs"],float));y=np.asarray(o["y_true"],int)
    return dict(single=allm(prs[0],y),ens=allm(np.mean(prs,0),y),n=len(ps))
SET=[("CT-RATE-int(CT-CLIP)","ctrate",[(v,"results_matrix_ctrate_VARIANT_zD") for v in ("zeroshot","vocabfine","classfine")]),
     ("RAD-ChestCT(16)","radchest",[(v,"results_matrix_radchest_VARIANT_zD") for v in ("zeroshot","vocabfine","classfine")]),
     ("CC-CCII(3)","ccii",[(v,"results_matrix_ccii_VARIANT_zD") for v in ("zeroshot","vocabfine","classfine")]),
     ("LUNA16(bin)","luna",[(v,"results_matrix_luna_VARIANT_zD") for v in ("zeroshot","vocabfine","classfine")])]
METH=["zeroshot","tent","ml_tta","bem","carve_xview","carve_xview_gate"]
METR=["AUROC","AUPRC","F1","Prec","Rec","Acc"]
rows=[]
def D(t,v,d):return t.replace("VARIANT",v).replace("D",str(d))
for sname,ds,vs in SET:
    for variant,tmpl in vs:
        for m in METH:
            r={}
            for depth in (40,240):
                c=cell(D(tmpl,variant,depth),ds,m); r[depth]=c
            row=dict(setting=sname,variant=variant,method=m)
            for depth in (40,240):
                c=r[depth]
                for agg in ("single","ens"):
                    for mt in METR:
                        row[f"{mt}_z{depth}_{agg}"]=round(c[agg][mt],4) if c else ""
            rows.append(row)
# fvlm (single depth) — RAD-ChestCT and CT-RATE(external to fVLM)
FVLM=[("RAD-ChestCT(fVLM)","radchest","results_matrix_fvlm_radchest_z240"),
      ("CT-RATE(fVLM-ext)","ctrate","results_matrix_fvlm_ctrate_z240")]
for fsetting,fds,fdir in FVLM:
  for m in METH:
    c=cell(fdir,fds,m)
    row=dict(setting=fsetting,variant="zeroshot",method=m)
    for depth in (40,240):
        for agg in ("single","ens"):
            for mt in METR:
                row[f"{mt}_z{depth}_{agg}"]=(round(c[agg][mt],4) if (c and depth==240) else "n/a")
    rows.append(row)
os.makedirs("results",exist_ok=True)
with open("results/full_metrics_table.csv","w",newline="") as f:
    w=_csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
# print ensemble, paper's 4 metrics + AUPRC/Acc, both depths
for sname,ds,vs in SET:
    for variant,_ in vs:
        print(f"\n#### {sname} [{variant}] — seed-ENSEMBLE ####")
        print(f"{'method':16}|        z=40 (AUROC/AUPRC/F1/P/R/Acc)        |       z=240 (AUROC/AUPRC/F1/P/R/Acc)")
        for r in [x for x in rows if x['setting']==sname and x['variant']==variant]:
            a=" ".join(f"{r[f'{mt}_z40_ens']}" for mt in METR)
            b=" ".join(f"{r[f'{mt}_z240_ens']}" for mt in METR)
            print(f"{r['method']:16}| {a} | {b}")
for fs in ["RAD-ChestCT(fVLM)","CT-RATE(fVLM-ext)"]:
    print(f"\n#### {fs} z=240 ENSEMBLE (AUROC/AUPRC/F1/P/R/Acc) ####")
    for r in [x for x in rows if x['setting']==fs]:
        print(f"{r['method']:16}| "+" ".join(f"{r[f'{mt}_z240_ens']}" for mt in METR))
print("\nwrote results/full_metrics_table.csv (single+ens x z40+z240 x 6 metrics)")
