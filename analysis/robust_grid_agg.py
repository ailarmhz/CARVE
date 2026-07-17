#!/usr/bin/env python3
"""Honest matched-budget robustness grid.

For each (dataset, budget) cell, ALL methods run at the SAME budget (no per-method
handicap). We report, vs each cell's own zero-shot:
  - dAUROC, dAccuracy per method
  - the gate's abstention rate (fraction of scans where carve_xview_gate reverted to
    zero-shot because sigma_pred <= tau)
and the WORST-CASE (min) dAUROC / dAcc per method across budgets -> the bounded-harm
claim for the gate vs the always-on methods.
"""
import torch, numpy as np, glob, os
import csv as _csv

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
def acc(pr,y):return float(((pr>0.5).astype(int)==y).mean())

def ens(d,ds,m):
    ps=[p for p in sorted(glob.glob(f"{d}/{ds}_{m}_seed*.pt")) if "_sh" not in os.path.basename(p)]
    prs=[];y=None;gate=[]
    for p in ps:
        o=torch.load(p,map_location="cpu",weights_only=False)
        prs.append(np.asarray(o["probs"],float));y=np.asarray(o["y_true"],int)
        if "gate_applied" in o: gate.append(np.asarray(o["gate_applied"]).astype(float))
    if not prs: return None
    g=float(np.mean([gg.mean() for gg in gate])) if gate else np.nan
    return np.mean(prs,0),y,g,len(ps)

# (dataset_key, dataset_id_in_files, budget_label, results_dir)
CELLS = [
    ("internal","ctrate","B0_safe(lr1e-5 s2 norm)",       "results_endtask_internal_zeroshot"),
    ("internal","ctrate","B1_norm(lr1e-3 s2 norm)",       "results_methodsep_internal_lr3"),
    ("internal","ctrate","B2_drift(finalblk lr1e-3 s2)",  "results_robust_internal_finalblk_lr3s2"),
    ("radchest","radchest","B0_safe(lr1e-5 s2 norm)",     "results_endtask_external_zeroshot"),
    ("radchest","radchest","B2_drift(finalblk lr1e-3 s2)","results_robust_radchest_finalblk_lr3s2"),
    ("ccii","ccii","B0_safe(lr1e-5 s2 norm)",             "results_carve_xview_z240"),
    ("ccii","ccii","B2_drift(finalblk lr1e-3 s2)",        "results_robust_ccii_finalblk_lr3s2"),
]
METHODS=["tent","ml_tta","bem","carve_xview","carve_xview_gate"]

def main():
    rows=[]; worst={m:{"dA":[], "dAcc":[]} for m in METHODS}
    cur=None
    for dk,did,budget,d in CELLS:
        if not os.path.isdir(d):
            print(f"[skip absent] {dk} {budget} ({d})"); continue
        z=ens(d,did,"zeroshot")
        if z is None:
            # fall back to the dataset's B0 zeroshot as the common reference
            zb={"internal":"results_endtask_internal_zeroshot","radchest":"results_endtask_external_zeroshot",
                "ccii":"results_carve_xview_z240"}[dk]
            z=ens(zb,did,"zeroshot")
        zpr,zy,_,_=z; Az=mac(zpr,zy); Cz=acc(zpr,zy)
        print(f"\n### {dk:9} {budget:30} (zeroshot AUROC={Az:.4f} acc={Cz:.4f}) ###")
        for m in METHODS:
            r=ens(d,did,m)
            if r is None: print(f"   {m:18} (absent)"); continue
            pr,y,g,ns=r
            dA=mac(pr,y)-Az; dC=acc(pr,y)-Cz
            gate_s=f"  abstain={g:.0%}" if (m=="carve_xview_gate" and not np.isnan(g)) else ""
            print(f"   {m:18} dAUROC={dA:+.4f}  dAcc={dC:+.4f}{gate_s}  (sd={ns})")
            rows.append(dict(dataset=dk,budget=budget,method=m,dAUROC=round(dA,4),dAcc=round(dC,4),
                             gate_abstain=round(g,3) if not np.isnan(g) else "",nseeds=ns))
            # worst-case only over DRIFT budgets (B1/B2) where harm can occur
            if "B0" not in budget:
                worst[m]["dA"].append(dA); worst[m]["dAcc"].append(dC)
    print("\n================ WORST-CASE over drift budgets (min dAUROC / min dAcc) ================")
    print(f"{'method':18} {'worst dAUROC':>13} {'worst dAcc':>11}")
    for m in METHODS:
        wa=min(worst[m]["dA"]) if worst[m]["dA"] else float("nan")
        wc=min(worst[m]["dAcc"]) if worst[m]["dAcc"] else float("nan")
        print(f"{m:18} {wa:>13.4f} {wc:>11.4f}")
    if rows:
        os.makedirs("results",exist_ok=True)
        with open("results/robustness_grid.csv","w",newline="") as f:
            w=_csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)
        print(f"\nwrote results/robustness_grid.csv ({len(rows)} rows)")

if __name__=="__main__":
    main()
