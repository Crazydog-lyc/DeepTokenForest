"""
analyze_deep_token_forest_run.py

Post-hoc diagnostics for one Deep Token Forest run.
No training is performed.

Produces:
- stage transfer summary,
- late-vs-early transfer behavior,
- split gain quantiles and honest/search agreement,
- confusion-pair concentration,
- concept class-separation ranking,
- compact JSON report.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def qstats(values):
    x=np.asarray([v for v in values if v is not None and np.isfinite(v)],dtype=float)
    if len(x)==0:
        return {"n":0}
    return {
        "n":int(len(x)),
        "min":float(np.min(x)),
        "q10":float(np.quantile(x,0.10)),
        "q25":float(np.quantile(x,0.25)),
        "median":float(np.median(x)),
        "q75":float(np.quantile(x,0.75)),
        "q90":float(np.quantile(x,0.90)),
        "max":float(np.max(x)),
        "mean":float(np.mean(x)),
    }


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--output",default=None)
    return p.parse_args()


def main():
    args=parse_args()
    run_dir=Path(args.run_dir)
    metrics=json.loads(
        (run_dir/"metrics_deep_token_forest.json").read_text(encoding="utf-8")
    )
    diag=run_dir/"diagnostics"
    history=metrics.get("history",[])

    by_stage={}
    for r in history:
        by_stage.setdefault(int(r["stage"]),[]).append(r)

    stage_summary={}
    for s,rows in sorted(by_stage.items()):
        tr=float(sum(r.get("train_improvement",0.0) for r in rows))
        va=float(sum(r.get("eval_improvement",0.0) or 0.0 for r in rows))
        stage_summary[str(s)]={
            "trees":len(rows),
            "representation_dim":rows[0].get("representation_dim"),
            "train_improvement":tr,
            "val_improvement":va,
            "transfer":va/tr if abs(tr)>1e-15 else None,
            "tree_transfer_quantiles":qstats(
                [r.get("transfer_ratio") for r in rows]
            ),
            "effective_gain_quantiles":qstats(
                [r.get("mean_effective_gain") for r in rows]
            ),
        }

    n=len(history)
    chunk=max(1,n//5)
    temporal=[]
    for start in range(0,n,chunk):
        rows=history[start:min(n,start+chunk)]
        tr=float(sum(r.get("train_improvement",0.0) for r in rows))
        va=float(sum(r.get("eval_improvement",0.0) or 0.0 for r in rows))
        temporal.append(
            {
                "trees":[int(rows[0]["global_tree"]),int(rows[-1]["global_tree"])],
                "train_improvement":tr,
                "val_improvement":va,
                "transfer":va/tr if abs(tr)>1e-15 else None,
            }
        )

    tree_files=sorted(diag.glob("tree_s*_t*.json"))
    search_gains=[]
    honest_gains=[]
    effective_gains=[]
    honest_positive=0
    honest_total=0
    node_depths=[]
    node_sizes=[]
    for path in tree_files:
        obj=json.loads(path.read_text(encoding="utf-8"))
        stack=[obj["root"]]
        while stack:
            node=stack.pop()
            if node["isleaf"]:
                continue
            search_gains.append(node.get("search_gain"))
            hg=node.get("honest_gain")
            if hg is not None and np.isfinite(hg):
                honest_total+=1
                honest_positive+=int(hg>0)
                honest_gains.append(hg)
            effective_gains.append(node.get("effective_gain"))
            node_depths.append(node.get("depth"))
            node_sizes.append(node.get("n_samples"))
            stack.append(node["left"]); stack.append(node["right"])

    cm=np.asarray(metrics["validation"]["confusion_matrix"],dtype=int)
    pairs=[]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i!=j and cm[i,j]>0:
                pairs.append({"true":i,"pred":j,"count":int(cm[i,j])})
    pairs.sort(key=lambda x:x["count"],reverse=True)
    total_errors=int(cm.sum()-np.trace(cm))
    for p in pairs:
        p["fraction_of_errors"]=p["count"]/max(total_errors,1)

    concept_rank=[]
    for path in sorted(diag.glob("concept_stats_val_stage_*.json")):
        stage=int(path.stem.split("_")[-1])
        items=json.loads(path.read_text(encoding="utf-8"))
        for idx,item in enumerate(items):
            vals=[
                float(v)
                for v in item.get("class_doc_fraction_mean",{}).values()
            ]
            separation=(max(vals)-min(vals)) if vals else 0.0
            concept_rank.append(
                {
                    "stage":stage,
                    "concept_index":idx,
                    "source_tree":item.get("tree"),
                    "source_node_id":item.get("node_id"),
                    "effective_gain":item.get("effective_gain"),
                    "class_fraction_range":float(separation),
                    "class_doc_fraction_mean":item.get("class_doc_fraction_mean"),
                }
            )
    concept_rank.sort(
        key=lambda x:x["class_fraction_range"],reverse=True
    )

    report={
        "validation":metrics.get("validation"),
        "test":metrics.get("test"),
        "representation":metrics.get("representation"),
        "stage_summary":stage_summary,
        "temporal_transfer":temporal,
        "split_diagnostics":{
            "tree_files":len(tree_files),
            "search_gain":qstats(search_gains),
            "honest_gain":qstats(honest_gains),
            "effective_gain":qstats(effective_gains),
            "honest_positive_fraction":(
                honest_positive/honest_total if honest_total else None
            ),
            "node_depth":qstats(node_depths),
            "node_size":qstats(node_sizes),
        },
        "validation_error_pairs":pairs,
        "top_concepts_by_class_separation":concept_rank[:30],
    }

    out=Path(args.output) if args.output else run_dir/"posthoc_analysis.json"
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")

    print("Validation:",metrics.get("validation"))
    print("Test:",metrics.get("test"))
    print("\nStage transfer:")
    for k,v in stage_summary.items():
        print(k,v)
    print("\nTemporal transfer:")
    for row in temporal:
        print(row)
    print("\nTop confusion pairs:")
    for row in pairs[:10]:
        print(row)
    print("\nTop concepts:")
    for row in concept_rank[:10]:
        print(row)
    print("\nSaved",out)


if __name__=="__main__":
    main()
