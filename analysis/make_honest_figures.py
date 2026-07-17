#!/usr/bin/env python3
"""Honest figures from the MEASURED predictions (real-seed matrix).
Produces: fig_depth_precondition, fig_base_vs_tta, fig_objective_gate_ablation.
"""
import torch, numpy as np, glob, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 140, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
OUT = "figures/honest"; os.makedirs(OUT, exist_ok=True)
C = dict(zs="#4C72B0", tent="#DD8452", mltta="#55A868", bem="#C44E52", carve="#8172B3",
         gate="#937860", z40="#B0B0B0", z240="#2E5A88", ctclip="#C44E52", fvlm="#55A868")

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
    if not ps:return None
    prs=[];y=None
    for p in ps:
        o=torch.load(p,map_location="cpu",weights_only=False);prs.append(np.asarray(o["probs"],float));y=np.asarray(o["y_true"],int)
    pe=np.mean(prs,0); return dict(A=mac(pe,y),Acc=acc(pe,y))

# ---------------- Fig 1: depth precondition ----------------
def fig_depth():
    cks=["zeroshot","vocabfine","classfine"]; lbl=["Zero-shot","Vocab-FT","Class-FT"]
    intern40=[ens(f"results_matrix_ctrate_{c}_z40","ctrate","zeroshot")["A"] for c in cks]
    intern240=[ens(f"results_matrix_ctrate_{c}_z240","ctrate","zeroshot")["A"] for c in cks]
    ext40=[ens(f"results_matrix_radchest_{c}_z40","radchest","zeroshot")["A"] for c in cks]
    ext240=[ens(f"results_matrix_radchest_{c}_z240","radchest","zeroshot")["A"] for c in cks]
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.9),sharey=True)
    x=np.arange(3); w=0.36
    for k,(ax,(a40,a240,title)) in enumerate(zip(axes,[(intern40,intern240,"CT-RATE (internal / in-distribution)"),
                                          (ext40,ext240,"RAD-ChestCT (external / shifted)")])):
        b1=ax.bar(x-w/2,a40,w,label="z=40 (reduced depth)",color=C["z40"],edgecolor="k",lw=.6)
        b2=ax.bar(x+w/2,a240,w,label="z=240 (native depth)",color=C["z240"],edgecolor="k",lw=.6)
        ax.axhline(0.5,ls=":",c="gray",lw=1); ax.set_xticks(x); ax.set_xticklabels(lbl)
        ax.set_title(title,fontsize=10.5); ax.set_ylim(0.45,0.88)
        for b in list(b1)+list(b2):
            ax.text(b.get_x()+b.get_width()/2,b.get_height()+.005,f"{b.get_height():.3f}",
                    ha="center",va="bottom",fontsize=7.5)
        for xi,(lo,hi) in enumerate(zip(a40,a240)):
            if hi-lo>0.03:
                ax.annotate(f"+{hi-lo:.3f}",xy=(xi,max(lo,hi)+0.045),fontsize=8.5,
                            ha="center",color=C["z240"],fontweight="bold")
    axes[0].set_ylabel("Macro AUROC")
    axes[1].legend(frameon=False,loc="upper right",fontsize=9)
    axes[1].text(2.42,0.502,"chance",fontsize=7.5,color="gray",va="bottom",ha="right")
    fig.suptitle("Input depth is a precondition: native depth recovers the in-distribution base,\n"
                 "but the external deficit is genuine shift (unchanged by depth)",fontsize=10.5,y=1.05)
    for ext in ("pdf","png"): fig.savefig(f"{OUT}/fig_depth_precondition.{ext}")
    plt.close(fig); print("wrote fig_depth_precondition")

# ---------------- Fig 2: base model >> TTA ----------------
def fig_base():
    meths=["zeroshot","tent","ml_tta","bem","carve_xview"]; ml=["No-TTA","Tent","ML-TTA","BEM","CARVE"]
    ct=[ens("results_matrix_radchest_zeroshot_z240","radchest",m)["A"] for m in meths]
    fv=[ens("results_matrix_fvlm_radchest_z240","radchest",m)["A"] for m in meths]
    fig,ax=plt.subplots(figsize=(7.2,4.2))
    x=np.arange(len(meths))
    ax.plot(x,ct,"o-",color=C["ctclip"],lw=2,ms=8,label="CT-CLIP (global mean-pool)")
    ax.plot(x,fv,"s-",color=C["fvlm"],lw=2,ms=8,label="fVLM (anatomy-aware)")
    ax.axhline(0.5,ls=":",c="gray",lw=1)
    ax.annotate("",xy=(0,fv[0]),xytext=(0,ct[0]),arrowprops=dict(arrowstyle="<->",color="k",lw=1.3))
    ax.text(0.12,(ct[0]+fv[0])/2,f"base-model gap\n+{fv[0]-ct[0]:.3f} AUROC",fontsize=9,va="center")
    ax.text(2.0,ct[0]-0.012,f"TTA spread within CT-CLIP: {max(ct)-min(ct):.3f}",fontsize=8.5,color=C["ctclip"],ha="center")
    ax.set_xticks(x); ax.set_xticklabels(ml); ax.set_ylabel("Macro AUROC (external RAD-ChestCT)")
    ax.set_ylim(0.46,0.66); ax.legend(frameon=False,loc="center right")
    ax.set_title("Base architecture, not the TTA objective, governs external robustness\n"
                 "(all objectives coincide within each backbone)",fontsize=10.5)
    for ext in ("pdf","png"): fig.savefig(f"{OUT}/fig_base_vs_tta.{ext}")
    plt.close(fig); print("wrote fig_base_vs_tta")

# ---------------- Fig 3: objectives coincide at the usable budget ----------------
def fig_coincide():
    meths=["zeroshot","tent","ml_tta","bem","carve_xview"]; ml=["No-TTA","Tent","ML-TTA","BEM","CARVE"]
    cols=[C["zs"],C["tent"],C["mltta"],C["bem"],C["carve"]]
    intern=[ens("results_matrix_ctrate_zeroshot_z240","ctrate",m)["A"] for m in meths]
    extern=[ens("results_matrix_radchest_zeroshot_z240","radchest",m)["A"] for m in meths]
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.9))
    x=np.arange(len(meths))
    for ax,(vals,ttl,lo,hi) in zip(axes,[(intern,"CT-RATE (internal)",0.70,0.72),
                                          (extern,"RAD-ChestCT (external)",0.49,0.515)]):
        ax.bar(x,vals,0.6,color=cols,edgecolor="k",lw=.6)
        ax.set_xticks(x); ax.set_xticklabels(ml,rotation=18,ha="right")
        ax.set_title(ttl,fontsize=10.5); ax.set_ylim(lo,hi); ax.set_ylabel("Macro AUROC")
        for xi,v in enumerate(vals): ax.text(xi,v+(hi-lo)*0.01,f"{v:.4f}",ha="center",va="bottom",fontsize=7.5)
        spread=max(vals)-min(vals)
        ax.text(0.02,0.97,f"spread = {spread:.4f}",transform=ax.transAxes,fontsize=8.5,va="top",
                bbox=dict(boxstyle="round,pad=0.3",fc="#f5f5f5",ec="gray"))
    fig.suptitle("At the usable budget (lr 1e-5, 2 steps, 8-view selection), all five objectives coincide —\n"
                 "multi-view selection and the cardinality loss do not separate from No-TTA",fontsize=10.5,y=1.05)
    for ext in ("pdf","png"): fig.savefig(f"{OUT}/fig_objective_coincidence.{ext}")
    plt.close(fig); print("wrote fig_objective_coincidence")

# ---------------- Fig 4: worst-case harm across budgets (the gate's role) ----------------
def fig_worstcase():
    meths=["tent","ml_tta","bem","carve_xview","carve_xview_gate"]
    ml=["Tent","ML-TTA","BEM","CARVE","CARVE-\nGated"]; cols=[C["tent"],C["mltta"],C["bem"],C["carve"],C["gate"]]
    # worst-case (min) dAUROC / dAcc vs No-TTA over drift cells we ran (internal, radchest, ccii @ final-block)
    cells=[("results_robust_internal_finalblk_lr3s2","ctrate"),
           ("results_robust_radchest_finalblk_lr3s2","radchest"),
           ("results_robust_ccii_finalblk_lr3s2","ccii")]
    wA={m:[] for m in meths}; wC={m:[] for m in meths}
    for d,ds in cells:
        z=ens(d,ds,"zeroshot")
        for m in meths:
            r=ens(d,ds,m)
            if r: wA[m].append(r["A"]-z["A"]); wC[m].append(r["Acc"]-z["Acc"])
    wcA=[min(wA[m]) for m in meths]; wcC=[min(wC[m]) for m in meths]
    fig,axes=plt.subplots(1,2,figsize=(9.6,4.0))
    x=np.arange(len(meths))
    for ax,(vals,ttl,unit) in zip(axes,[(wcA,"Worst-case ΔAUROC","ΔAUROC"),
                                        (wcC,"Worst-case ΔAccuracy","ΔAccuracy")]):
        ax.bar(x,vals,0.6,color=cols,edgecolor="k",lw=.6)
        ax.axhline(0,c="k",lw=.8); ax.set_xticks(x); ax.set_xticklabels(ml,fontsize=9)
        ax.set_title(ttl,fontsize=10.5); ax.set_ylabel(unit)
        for xi,v in enumerate(vals): ax.text(xi,v-0.002 if v<0 else v+0.002,f"{v:.3f}",
                                             ha="center",va="top" if v<0 else "bottom",fontsize=8)
        ax.margins(y=0.18)
    axes[0].annotate("CARVE's loss is the\nmost harmful when pushed",xy=(3,wcA[3]),xytext=(0.3,wcA[3]*0.55),
                     fontsize=8.2,color=C["carve"],arrowprops=dict(arrowstyle="->",color=C["carve"]))
    axes[1].annotate("gate abstains →\nsmallest worst-case harm",xy=(4,wcC[4]),xytext=(1.4,wcC[4]*0.6),
                     fontsize=8.2,color=C["gate"],arrowprops=dict(arrowstyle="->",color=C["gate"]))
    fig.suptitle("Pushed to a high-drift budget, the objective only harms; the dispersion GATE is the one\n"
                 "component with a real benefit — it bounds worst-case operating-point harm by abstaining",
                 fontsize=10.5,y=1.06)
    for ext in ("pdf","png"): fig.savefig(f"{OUT}/fig_gate_worstcase.{ext}")
    plt.close(fig); print("wrote fig_gate_worstcase")

fig_depth(); fig_base(); fig_coincide(); fig_worstcase()
print(f"\nFigures in {OUT}/  (pdf+png)")
