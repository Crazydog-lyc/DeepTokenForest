"""
run_agnews_ablation.py

Launch controlled AG News ablations and collect one summary table.

The important principle is "one mechanism at a time":
- fixed_distribution: remove Top-K but keep representation fixed;
- +honest: isolate candidate-selection regularization;
- +growth: isolate token concept feature growth;
- +prefix: isolate causal token mixing;
- PCA64/128/256/full: isolate information loss due to PCA.

All Deep Token Forest variants use the same number of trees unless overridden.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--dataset-dir",default="./data/ag_news")
    p.add_argument("--checkpoint",default="./checkpoints/pythia-70m")
    p.add_argument("--output-root",default="./runs/deep_token_forest_ablation")
    p.add_argument("--feature-cache-dir",default="./feature_cache/agnews_deep_token_forest")
    p.add_argument("--python",default=sys.executable)
    p.add_argument("--n-stages",type=int,default=4)
    p.add_argument("--trees-per-stage",type=int,default=25)
    p.add_argument("--max-length",type=int,default=64)
    p.add_argument("--hidden-state-index",type=int,default=2)
    p.add_argument("--max-train-samples",type=int,default=0)
    p.add_argument("--max-val-samples",type=int,default=0)
    p.add_argument("--max-test-samples",type=int,default=0)
    p.add_argument("--random-state",type=int,default=42)
    p.add_argument(
        "--suite",
        choices=["core","pca","search","all"],
        default="core",
    )
    p.add_argument("--continue-on-error",action="store_true")
    p.add_argument("--include-topk-baseline",action="store_true")
    return p.parse_args()


def deep_variant(name, extra):
    return {"name":name,"kind":"deep","extra":list(extra)}


def variants_for_suite(suite):
    core=[
        deep_variant(
            "A_fixed_distribution_no_honest",
            ["--token-proj-dim","64","--disable-feature-growth","--honest-fraction","0"],
        ),
        deep_variant(
            "B_fixed_distribution_honest",
            ["--token-proj-dim","64","--disable-feature-growth","--honest-fraction","0.15"],
        ),
        deep_variant(
            "C_growth_distribution",
            ["--token-proj-dim","64","--honest-fraction","0.15"],
        ),
        deep_variant(
            "D_growth_distribution_prefix",
            ["--token-proj-dim","64","--honest-fraction","0.15","--prefix-mixing"],
        ),
    ]
    pca=[
        deep_variant("PCA64_growth",["--token-proj-dim","64","--honest-fraction","0.15"]),
        deep_variant("PCA128_growth",["--token-proj-dim","128","--honest-fraction","0.15"]),
        deep_variant("PCA256_growth",["--token-proj-dim","256","--honest-fraction","0.15"]),
        deep_variant("PCAfull_growth",["--token-proj-dim","0","--honest-fraction","0.15"]),
    ]
    search=[
        deep_variant(
            "S_narrow_search",
            [
                "--token-proj-dim","64",
                "--n-random-directions","2",
                "--n-token-prototype-directions","2",
                "--n-local-perturbations","1",
                "--beam-width","2",
                "--honest-fraction","0",
            ],
        ),
        deep_variant(
            "S_default_search_no_honest",
            ["--token-proj-dim","64","--honest-fraction","0"],
        ),
        deep_variant(
            "S_default_search_honest",
            ["--token-proj-dim","64","--honest-fraction","0.15"],
        ),
    ]
    if suite=="core":
        return core
    if suite=="pca":
        return pca
    if suite=="search":
        return search
    # Deduplicate names for all.
    out=[]
    seen=set()
    for v in core+pca+search:
        if v["name"] not in seen:
            out.append(v); seen.add(v["name"])
    return out


def load_deep_metrics(run_dir):
    path=run_dir/"metrics_deep_token_forest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_topk_metrics(run_dir):
    path=run_dir/"metrics_token_topk_tanh.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def row_from_report(name,kind,report):
    if report is None:
        return {"name":name,"kind":kind,"status":"missing_metrics"}
    rep=report.get("representation",{})
    val=report.get("validation",{})
    test=report.get("test",{})
    transfer=report.get("transfer_summary",{})
    last_stage=None
    if transfer:
        last_stage=transfer[sorted(transfer.keys(),key=int)[-1]]
    return {
        "name":name,
        "kind":kind,
        "status":"ok",
        "base_dim":rep.get("base_token_dim",rep.get("token_dim")),
        "final_dim":rep.get("final_token_dim",rep.get("token_dim")),
        "pca_explained_variance":rep.get("pca_explained_variance_ratio"),
        "feature_growth":rep.get("feature_growth"),
        "prefix_mixing":rep.get("prefix_mixing"),
        "val_logloss":val.get("logloss"),
        "val_accuracy":val.get("accuracy"),
        "val_macro_f1":val.get("macro_f1"),
        "test_logloss":test.get("logloss"),
        "test_accuracy":test.get("accuracy"),
        "test_macro_f1":test.get("macro_f1"),
        "last_stage_transfer":(
            None if last_stage is None else last_stage.get("val_over_train_transfer")
        ),
        "best_iteration":report.get("best_iteration"),
    }


def main():
    args=parse_args()
    here=Path(__file__).resolve().parent
    train_script=here/"train_agnews_deep_token_forest.py"
    topk_script=here/"train_agnews_token_topk_tanh.py"
    out_root=Path(args.output_root)
    out_root.mkdir(parents=True,exist_ok=True)

    variants=variants_for_suite(args.suite)
    if args.include_topk_baseline:
        variants.insert(
            0,
            {
                "name":"TOPK_original_control",
                "kind":"topk",
                "extra":[],
            },
        )

    common=[
        "--dataset-dir",args.dataset_dir,
        "--checkpoint",args.checkpoint,
        "--feature-cache-dir",args.feature_cache_dir,
        "--hidden-state-index",str(args.hidden_state_index),
        "--max-length",str(args.max_length),
        "--max-train-samples",str(args.max_train_samples),
        "--max-val-samples",str(args.max_val_samples),
        "--max-test-samples",str(args.max_test_samples),
        "--random-state",str(args.random_state),
    ]

    summary=[]
    for v in variants:
        run_dir=out_root/v["name"]
        print("\n"+"="*100)
        print("RUN",v["name"])
        print("="*100)
        if v["kind"]=="deep":
            cmd=[
                args.python,str(train_script),
                *common,
                "--output-dir",str(run_dir),
                "--n-stages",str(args.n_stages),
                "--trees-per-stage",str(args.trees_per_stage),
                *v["extra"],
            ]
        else:
            if not topk_script.exists():
                print("Skipping Top-K control: train_agnews_token_topk_tanh.py not beside runner.")
                summary.append({"name":v["name"],"kind":"topk","status":"script_missing"})
                continue
            total_trees=args.n_stages*args.trees_per_stage
            cmd=[
                args.python,str(topk_script),
                *common,
                "--output-dir",str(run_dir),
                "--n-estimators",str(total_trees),
                "--token-proj-dim","64",
                "--top-k","4",
            ]

        print(" ".join(cmd))
        try:
            subprocess.run(cmd,check=True)
            report=(
                load_deep_metrics(run_dir)
                if v["kind"]=="deep"
                else load_topk_metrics(run_dir)
            )
            summary.append(row_from_report(v["name"],v["kind"],report))
        except subprocess.CalledProcessError as e:
            summary.append(
                {
                    "name":v["name"],
                    "kind":v["kind"],
                    "status":f"failed:{e.returncode}",
                }
            )
            if not args.continue_on_error:
                raise

        (out_root/"ablation_summary.json").write_text(
            json.dumps(summary,indent=2),
            encoding="utf-8",
        )

    columns=sorted({k for row in summary for k in row.keys()})
    with (out_root/"ablation_summary.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(summary)

    print("\nSUMMARY")
    for row in summary:
        print(row)
    print("Saved",out_root/"ablation_summary.json")
    print("Saved",out_root/"ablation_summary.csv")


if __name__=="__main__":
    main()
