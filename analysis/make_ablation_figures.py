#!/usr/bin/env python3
"""Ablation figures (measured): CARVE component ladder, fig:hparam, fig:augsweep."""
import torch, numpy as np, glob, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                     "figure.dpi":140,"savefig.bbox":"tight","font.family":"DejaVu Sans"})
OUT="figures/honest"; os.makedirs(OUT,exist_ok=True)
BLUE="#2E5A88"; GRAY="#B0B0B0"; CARVECOL="#8172B3"; GATECOL="#937860"

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
def ens(d,ds,m,want_vdis=False):
    ps=[p for p in sorted(glob.glob(f"{d}/{ds}_{m}_seed*.pt")) if "_sh" not in os.path.basename(p)]
    if not ps:return None
    prs=[];y=None;vd=[]
    for p in ps:
        o=torch.load(p,map_location="cpu",weights_only=False);prs.append(np.asarray(o["probs"],float));y=np.asarray(o["y_true"],int)
        if want_vdis and "view_disagreement" in o and len(o["view_disagreement"]): vd.append(float(np.asarray(o["view_disagreement"]).mean()))
    L=min(len(a) for a in prs); prs=[a[:L] for a in prs]; y=y[:L]
    r={"A":mac(np.mean(prs,0),y),"n":L,"sd":len(ps)}
    if want_vdis and vd: r["vdis"]=float(np.mean(vd))
    return r

# ---------- Fig: CARVE component ladder ----------
def fig_ladder():
    stages=[("No-TTA","results_matrix_ctrate_zeroshot_z240","results_matrix_radchest_zeroshot_z240","zeroshot"),
            ("w/o multi-view","results_ablate_noMV_ctrate_z240","results_ablate_noMV_radchest_z240","carve_xview"),
            ("w/o selection","results_ablate_noSel_ctrate_z240","results_ablate_noSel_radchest_z240","carve_xview"),
            ("CARVE","results_matrix_ctrate_zeroshot_z240","results_matrix_radchest_zeroshot_z240","carve_xview"),
            ("CARVE\n+gate","results_matrix_ctrate_zeroshot_z240","results_matrix_radchest_zeroshot_z240","carve_xview_gate")]
    names=[s[0] for s in stages]
    inte=[ens(s[1],"ctrate",s[3])["A"] for s in stages]
    exte=[ens(s[2],"radchest",s[3])["A"] for s in stages]
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.0))
    cols=[GRAY,BLUE,BLUE,CARVECOL,GATECOL]
    for ax,(vals,ttl,lo,hi) in zip(axes,[(inte,"CT-RATE (internal)",0.70,0.72),
                                          (exte,"RAD-ChestCT (external)",0.49,0.52)]):
        ax.bar(range(len(vals)),vals,0.62,color=cols,edgecolor="k",lw=.6)
        ax.set_xticks(range(len(vals))); ax.set_xticklabels(names,fontsize=8.5)
        ax.set_ylim(lo,hi); ax.set_ylabel("Macro AUROC"); ax.set_title(ttl,fontsize=10.5)
        for xi,v in enumerate(vals): ax.text(xi,v+(hi-lo)*0.012,f"{v:.4f}",ha="center",va="bottom",fontsize=7.8)
        ax.text(0.02,0.97,f"spread = {max(vals)-min(vals):.4f}",transform=ax.transAxes,fontsize=8.5,va="top",
                bbox=dict(boxstyle="round,pad=0.3",fc="#f5f5f5",ec="gray"))
    fig.suptitle("CARVE component ablation (z=240): removing multi-view or low-entropy selection\n"
                 "barely changes AUROC — the components do not separate CARVE from No-TTA",fontsize=10.5,y=1.05)
    for e in ("pdf","png"): fig.savefig(f"{OUT}/fig_carve_ladder.{e}")
    plt.close(fig); print("wrote fig_carve_ladder")

# ---------- Fig: hparam sensitivity ----------
def fig_hparam():
    refA=ens("results_matrix_ctrate_zeroshot_z240","ctrate","carve_xview")["A"]  # V8 rho0.25 lam0.8
    def gv(d): r=ens(d,"ctrate","carve_xview"); return r["A"] if r else np.nan
    V=[(1,gv("results_sweep_hp_V1")),(2,gv("results_sweep_hp_V2")),(4,gv("results_sweep_hp_V4")),
       (8,refA),(16,gv("results_sweep_hp_V16"))]
    RHO=[(0.1,gv("results_sweep_hp_rho0p1")),(0.25,refA),(0.5,gv("results_sweep_hp_rho0p5")),(1.0,gv("results_sweep_hp_rho1p0"))]
    LAM=[(0.3,gv("results_sweep_hp_lam0p3")),(0.5,gv("results_sweep_hp_lam0p5")),(0.8,refA),(1.0,gv("results_sweep_hp_lam1p0"))]
    fig,axes=plt.subplots(1,3,figsize=(11,3.6))
    for ax,(data,xlab,refx) in zip(axes,[(V,"V (number of views)",8),(RHO,r"$\rho$ (retained fraction)",0.25),(LAM,r"$\lambda_{neg}$",0.8)]):
        xs=[a for a,_ in data]; ys=[b for _,b in data]
        ax.plot(xs,ys,"o-",color=BLUE,lw=1.8,ms=7)
        ri=xs.index(refx); ax.plot(refx,ys[ri],"o",ms=13,mfc="none",mec="red",mew=2,label="chosen")
        ax.set_xlabel(xlab); ax.set_ylabel("Macro AUROC (internal)")
        fin=[y for y in ys if y==y]
        ax.set_ylim(min(fin)-0.01,max(fin)+0.01)
        ax.text(0.04,0.05,f"range = {max(fin)-min(fin):.4f}",transform=ax.transAxes,fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.3",fc="#f5f5f5",ec="gray"))
        ax.legend(frameon=False,fontsize=8,loc="upper right")
    fig.suptitle("Hyperparameter sensitivity (CARVE, internal CT-RATE, z=240): AUROC is flat across "
                 "$V$, $\\rho$, $\\lambda_{neg}$;\nthe chosen values sit on a plateau (no fine tuning)",fontsize=10,y=1.06)
    for e in ("pdf","png"): fig.savefig(f"{OUT}/fig_hparam.{e}")
    plt.close(fig); print("wrote fig_hparam")

# ---------- Fig: augmentation strength sweep ----------
def fig_augsweep():
    levels=[(0.0,"results_sweep_aug_s0"),(0.5,"results_sweep_aug_s0p5"),(1.0,"results_sweep_aug_s1"),
            (2.0,"results_sweep_aug_s2"),(4.0,"results_sweep_aug_s4")]
    xs=[];dis=[];au=[]
    for s,d in levels:
        r=ens(d,"ctrate","carve_xview",want_vdis=True)
        if r: xs.append(s); au.append(r["A"]); dis.append(r.get("vdis",np.nan))
    fig,ax=plt.subplots(figsize=(6.6,4.2)); ax2=ax.twinx()
    l1=ax.plot(xs,dis,"o-",color="#C44E52",lw=2,ms=7,label="cross-view disagreement")
    l2=ax2.plot(xs,au,"s--",color=BLUE,lw=2,ms=7,label="macro AUROC")
    ax.set_xlabel("augmentation strength (× reference HU jitter 0.02 / shift 0.03)")
    ax.set_ylabel("mean cross-view disagreement",color="#C44E52")
    ax2.set_ylabel("macro AUROC (internal)",color=BLUE)
    ax.tick_params(axis="y",colors="#C44E52"); ax2.tick_params(axis="y",colors=BLUE)
    fin=[a for a in au if a==a]; ax2.set_ylim(min(fin)-0.01,max(fin)+0.01)
    ax.axvline(1.0,ls=":",c="gray"); ax.text(1.04,ax.get_ylim()[1]*0.9,"chosen",fontsize=8,color="gray")
    ax.legend(handles=l1+l2,frameon=False,loc="center right",fontsize=9)
    ax.set_title("Stronger weak-view augmentation raises cross-view disagreement\n"
                 "without improving AUROC — multi-view is reliability filtering, not augmentation",fontsize=10)
    for e in ("pdf","png"): fig.savefig(f"{OUT}/fig_augsweep.{e}")
    plt.close(fig); print("wrote fig_augsweep")

import sys
which=sys.argv[1] if len(sys.argv)>1 else "all"
if which in("all","ladder"): fig_ladder()
if which in("all","hparam"): fig_hparam()
if which in("all","aug"): fig_augsweep()
print(f"figures in {OUT}/")
