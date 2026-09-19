#!/usr/bin/env python3
"""
selector_math_experiments.py

Reproducible diagnostics for DeepTokenForest concept selection.

Experiments:
1) Real-run offline re-selection from model_structure.json.
2) Synthetic redundancy/multiplicity experiment.
3) Synthetic scale-sensitivity experiment for residual-guided directions.

No teacher logits, distillation, or validation labels are used.
"""

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss

def flatten(node):
    if node["isleaf"]:
        return []
    return [node] + flatten(node["left"]) + flatten(node["right"])

def normalize_rows(W):
    W=np.asarray(W,dtype=float)
    return W/np.maximum(np.linalg.norm(W,axis=1,keepdims=True),1e-12)

def direction_stats(sel):
    W=normalize_rows([x["w"] for x in sel])
    C=np.abs(W@W.T)
    vals=C[np.triu_indices(len(W),1)]
    sv=np.linalg.svd(W,compute_uv=False)
    p=sv**2/max(np.sum(sv**2),1e-30)
    er=float(np.exp(-np.sum(p*np.log(np.maximum(p,1e-15)))))
    return dict(
        mean_abs_cos=float(vals.mean()),
        max_abs_cos=float(vals.max()),
        effective_rank=er,
    )

def selector_summary(sel,N):
    d=direction_stats(sel)
    d.update(
        root_count=int(sum(x["depth"]==0 for x in sel)),
        root_fraction=float(np.mean([x["depth"]==0 for x in sel])),
        mean_depth=float(np.mean([x["depth"] for x in sel])),
        median_coverage=float(np.median([x["n_samples"]/N for x in sel])),
        mean_gain_per_sample=float(np.mean([
            x["effective_gain"]/max(x["n_samples"],1) for x in sel
        ])),
        total_effective_gain=float(sum(x["effective_gain"] for x in sel)),
    )
    return d

def real_offline(model_path):
    model=json.load(open(model_path))
    N=108000
    stages=[]
    for st in model["stages"][:-1]:
        nodes=[]
        for ti,tr in enumerate(st["trees"]):
            for n in flatten(tr["root"]):
                x=dict(n); x["tree"]=ti
                nodes.append(x)
        stages.append(nodes)

    def legacy(n):
        return n["effective_gain"]*math.sqrt(n["n_samples"]/N)

    def normalized(n,beta=.5):
        c=n["n_samples"]/N
        return n["effective_gain"]/max(n["n_samples"],1)*c**beta

    def pick(nodes,score,k=6,mincov=0.0):
        p=[n for n in nodes if n["n_samples"]/N>=mincov]
        return sorted(p,key=score,reverse=True)[:k]

    rows=[]
    allsel={}
    configs=[
        ("legacy", lambda ns:pick(ns,legacy)),
        ("gain_only", lambda ns:pick(ns,lambda n:n["effective_gain"])),
        ("normalized_b05_cov10", lambda ns:pick(ns,lambda n:normalized(n,.5),mincov=.1)),
        ("normalized_b08_cov10", lambda ns:pick(ns,lambda n:normalized(n,.8),mincov=.1)),
    ]
    for name,fn in configs:
        allsel[name]=[]
        for si,nodes in enumerate(stages):
            s=fn(nodes); allsel[name]+=s
            r=selector_summary(s,N); r.update(method=name,stage=si+1)
            rows.append(r)
    df=pd.DataFrame(rows)

    aggregate=[]
    for name,ss in allsel.items():
        # Across stages, direction dimensions differ, so aggregate only scalar metrics.
        aggregate.append(dict(
            method=name,
            selected=len(ss),
            roots=int(sum(x["depth"]==0 for x in ss)),
            root_fraction=float(np.mean([x["depth"]==0 for x in ss])),
            mean_depth=float(np.mean([x["depth"] for x in ss])),
            median_coverage=float(np.median([x["n_samples"]/N for x in ss])),
            mean_gain_per_sample=float(np.mean([
                x["effective_gain"]/x["n_samples"] for x in ss
            ])),
            total_effective_gain=float(sum(x["effective_gain"] for x in ss)),
            mean_stage_effective_rank=float(df[df.method==name].effective_rank.mean()),
            mean_stage_max_abs_cos=float(df[df.method==name].max_abs_cos.mean()),
            mean_stage_mean_abs_cos=float(df[df.method==name].mean_abs_cos.mean()),
        ))
    return pd.DataFrame(aggregate),df

def hbin(pos,total):
    if total<=0:return 0.0
    p=pos/total
    if p<=0 or p>=1:return 0.0
    return -(p*math.log(p)+(1-p)*math.log(1-p))

def stump_gain(x,y,min_leaf=20):
    order=np.argsort(x,kind="mergesort")
    xs=x[order]; ys=y[order]
    n=len(y); total_pos=int(ys.sum()); H=hbin(total_pos,n)
    cpos=np.cumsum(ys); best=0.0
    for pos in range(min_leaf,n-min_leaf+1):
        if xs[pos-1]>=xs[pos]: continue
        lp=int(cpos[pos-1]); rp=total_pos-lp
        cond=(pos/n)*hbin(lp,pos)+((n-pos)/n)*hbin(rp,n-pos)
        best=max(best,H-cond)
    return best

def synthetic_redundancy(seeds=200):
    out=[]
    for seed in range(seeds):
        rng=np.random.default_rng(seed)
        Xtr=rng.normal(size=(1200,8)); Xte=rng.normal(size=(5000,8))
        def label(X):
            bits=np.stack([X[:,j]>.55 for j in range(4)],axis=1)
            return (bits.sum(axis=1)>=2).astype(int)
        ytr=label(Xtr); yte=label(Xte)

        cand=[]; tau=.5
        for j in range(4):
            for b in np.linspace(.35,.75,7):
                cand.append((j,b,
                    np.tanh((Xtr[:,j]-b)/tau),
                    np.tanh((Xte[:,j]-b)/tau)))
        for j in range(4,8):
            for b in (-.2,.2):
                cand.append((j,b,
                    np.tanh((Xtr[:,j]-b)/tau),
                    np.tanh((Xte[:,j]-b)/tau)))

        scores=np.array([stump_gain(c[2],ytr) for c in cand])
        top=np.argsort(-scores,kind="stable")[:6]

        A=np.column_stack([c[2] for c in cand])
        C=np.corrcoef(A,rowvar=False)
        selected=[]; remaining=list(range(len(cand)))
        while len(selected)<6:
            best=None; bestv=-1
            for i in remaining:
                red=max((abs(C[i,j]) for j in selected),default=0.0)
                val=scores[i]*(1-red**2)
                if val>bestv:
                    bestv=val; best=i
            selected.append(best); remaining.remove(best)

        for method,idx in [("top6",top),("corr_diverse",np.array(selected))]:
            Ftr=np.column_stack([cand[i][2] for i in idx])
            Fte=np.column_stack([cand[i][3] for i in idx])
            clf=DecisionTreeClassifier(
                max_depth=5,min_samples_leaf=15,random_state=seed
            )
            clf.fit(Ftr,ytr)
            p=np.clip(clf.predict_proba(Fte)[:,1],1e-6,1-1e-6)
            pred=(p>=.5).astype(int)
            maxcorr=max(
                abs(C[i,j])
                for ii,i in enumerate(idx)
                for j in idx[:ii]
            )
            out.append(dict(
                seed=seed,method=method,
                accuracy=accuracy_score(yte,pred),
                logloss=log_loss(yte,np.c_[1-p,p]),
                distinct_signal_dims=len(set(
                    cand[i][0] for i in idx if cand[i][0]<4
                )),
                max_abs_activation_corr=maxcorr,
            ))
    return pd.DataFrame(out)

def synthetic_scale(seeds=500,a=10.0):
    out=[]
    true=np.array([1.,1.]); true/=np.linalg.norm(true)
    for seed in range(seeds):
        rng=np.random.default_rng(seed)
        Z=rng.normal(size=(2000,2))
        r=Z[:,0]+Z[:,1]+.3*rng.normal(size=2000)

        w=Z.T@r; w/=np.linalg.norm(w)
        p=Z@w

        scale=np.array([a,1.])
        Zs=Z*scale
        ws=Zs.T@r; ws/=np.linalg.norm(ws)
        ps=Zs@ws
        eff=scale*ws; eff/=np.linalg.norm(eff)

        Zz=(Zs-Zs.mean(0))/Zs.std(0)
        wz=Zz.T@r; wz/=np.linalg.norm(wz)
        pz=Zz@wz

        out.append(dict(
            seed=seed,
            corr_unscaled=np.corrcoef(p,r)[0,1],
            corr_scaled=np.corrcoef(ps,r)[0,1],
            corr_standardized=np.corrcoef(pz,r)[0,1],
            cosine_scaled_to_true=abs(float(eff@true)),
            effective_weight_ratio=abs(float(eff[0]/eff[1])),
        ))
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model-structure",required=True)
    ap.add_argument("--out-dir",default="./selector_experiment_results")
    ap.add_argument("--synthetic-seeds",type=int,default=200)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)

    agg,stage=real_offline(args.model_structure)
    syn=synthetic_redundancy(args.synthetic_seeds)
    scale=synthetic_scale()

    agg.to_csv(out/"real_offline_selector_aggregate.csv",index=False)
    stage.to_csv(out/"real_offline_selector_by_stage.csv",index=False)
    syn.to_csv(out/"synthetic_redundancy.csv",index=False)
    scale.to_csv(out/"synthetic_scale.csv",index=False)

    summary={
        "real_offline":agg.to_dict(orient="records"),
        "synthetic_redundancy":{
            m:g[["accuracy","logloss","distinct_signal_dims","max_abs_activation_corr"]]
              .mean().to_dict()
            for m,g in syn.groupby("method")
        },
        "synthetic_scale":scale.mean(numeric_only=True).to_dict(),
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
