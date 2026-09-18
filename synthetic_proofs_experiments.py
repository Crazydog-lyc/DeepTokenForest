"""
synthetic_proofs_experiments.py

Small, reproducible experiments corresponding to the mathematical claims behind
the Deep Token Forest redesign.  They are deliberately independent of AG News
and do not require a language model checkpoint.

Experiments
-----------
1. Top-K information loss:
   Two token bags have identical top/bottom K responses for every 1-D direction,
   while a middle token carries the label. Top-K is provably blind; an empirical
   survival/CDF statistic separates perfectly.

2. Strict feature-class expansion:
   A single raw linear threshold cannot represent the two-tail set
       {x < -1} union {x > 1}.
   After adding two learned half-space concept channels, one linear threshold in
   the augmented feature space represents the union.

3. Prefix mixing breaks permutation invariance:
   Bag/distribution statistics cannot distinguish AB from BA when the token
   multiset is identical. A causal prefix concept can.

4. Candidate multiplicity / winner's curse:
   With one weak true direction and many noise directions, selecting the largest
   search-set correlation increasingly chooses noise. Rechecking a shortlist on
   an independent in-training honest subset improves true-direction selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def best_threshold_accuracy(train_score, train_y, test_score, test_y):
    train_score = np.asarray(train_score, dtype=np.float64)
    test_score = np.asarray(test_score, dtype=np.float64)
    train_y = np.asarray(train_y, dtype=np.int64)
    test_y = np.asarray(test_y, dtype=np.int64)

    vals = np.unique(train_score)
    if len(vals) == 1:
        pred = np.full_like(test_y, int(np.mean(train_y) >= 0.5))
        return float(np.mean(pred == test_y)), None, 1

    thresholds = np.concatenate(
        [
            [vals[0] - 1e-8],
            (vals[:-1] + vals[1:]) * 0.5,
            [vals[-1] + 1e-8],
        ]
    )
    best = (-1.0, None, 1)
    for threshold in thresholds:
        for orientation in (1, -1):
            pred = (
                orientation * train_score
                >= orientation * threshold
            ).astype(np.int64)
            acc = float(np.mean(pred == train_y))
            if acc > best[0]:
                best = (acc, float(threshold), orientation)

    _, threshold, orientation = best
    pred_test = (
        orientation * test_score
        >= orientation * threshold
    ).astype(np.int64)
    return float(np.mean(pred_test == test_y)), threshold, orientation


def experiment_topk_information_loss(seed=0):
    rng = np.random.default_rng(seed)
    T = 9
    K = 4
    base = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], dtype=np.float64)

    def make(n):
        y = rng.integers(0, 2, n)
        X = np.tile(base, (n, 1))
        # Only the middle token carries the label and never enters top/bottom 4.
        X[:, 4] = (2 * y - 1) * 0.5 + rng.normal(0, 0.08, n)
        X[:, :4] += rng.normal(0, 0.03, (n, 4))
        X[:, 5:] += rng.normal(0, 0.03, (n, 4))
        return X, y

    Xtr, ytr = make(5000)
    Xte, yte = make(5000)

    topk_results = []
    for w in (+1.0, -1.0):
        qtr = Xtr * w
        qte = Xte * w
        str_ = np.tanh(
            np.partition(qtr, T - K, axis=1)[:, T - K:]
        ).mean(axis=1)
        ste = np.tanh(
            np.partition(qte, T - K, axis=1)[:, T - K:]
        ).mean(axis=1)
        acc, thr, orient = best_threshold_accuracy(
            str_, ytr, ste, yte
        )
        topk_results.append(
            {
                "w": w,
                "test_accuracy": acc,
                "threshold": thr,
                "orientation": orient,
            }
        )

    # One point of the projected empirical survival function.
    ctr = (Xtr >= 0.0).mean(axis=1)
    cte = (Xte >= 0.0).mean(axis=1)
    cdf_acc, cdf_thr, cdf_orient = best_threshold_accuracy(
        ctr, ytr, cte, yte
    )

    return {
        "name": "topk_information_loss",
        "topk_best_test_accuracy": max(
            x["test_accuracy"] for x in topk_results
        ),
        "topk_all": topk_results,
        "projection_survival_test_accuracy": cdf_acc,
        "projection_survival_threshold": cdf_thr,
        "projection_survival_orientation": cdf_orient,
        "interpretation": (
            "The label lives in a middle order statistic. Top-K and bottom-K "
            "are unchanged, whereas the empirical survival function at b=0 "
            "changes by exactly 1/T."
        ),
    }


def experiment_feature_growth_strict_expansion(seed=1):
    rng = np.random.default_rng(seed)

    def make(n):
        y = rng.integers(0, 2, n)
        x = np.empty(n, dtype=np.float64)
        neg = np.flatnonzero(y == 0)
        pos = np.flatnonzero(y == 1)
        x[neg] = rng.normal(0.0, 0.25, len(neg))
        signs = rng.choice([-1.0, 1.0], size=len(pos))
        x[pos] = signs * (2.0 + rng.normal(0.0, 0.25, len(pos)))
        return x, y

    xtr, ytr = make(10000)
    xte, yte = make(10000)

    raw_acc, raw_thr, raw_orient = best_threshold_accuracy(
        xtr, ytr, xte, yte
    )

    # Two first-layer tree concepts.
    gtr = np.column_stack([xtr > 1.0, xtr < -1.0]).astype(np.float64)
    gte = np.column_stack([xte > 1.0, xte < -1.0]).astype(np.float64)

    # One later oblique direction [1,1] on learned concept features.
    str_ = gtr.sum(axis=1)
    ste = gte.sum(axis=1)
    aug_acc, aug_thr, aug_orient = best_threshold_accuracy(
        str_, ytr, ste, yte
    )

    return {
        "name": "feature_growth_strict_expansion",
        "raw_one_threshold_test_accuracy": raw_acc,
        "raw_threshold": raw_thr,
        "raw_orientation": raw_orient,
        "augmented_one_threshold_test_accuracy": aug_acc,
        "augmented_threshold": aug_thr,
        "augmented_orientation": aug_orient,
        "interpretation": (
            "A raw half-line cannot represent two disjoint tails. After "
            "g1=1[x>1] and g2=1[x<-1] are appended, the single later split "
            "g1+g2>=0.5 represents their union."
        ),
    }


def experiment_prefix_breaks_permutation_invariance():
    # All 24 permutations of A,B,C,D. Label = A appears before B.
    import itertools

    tokens = ("A", "B", "C", "D")
    seqs = list(itertools.permutations(tokens))
    y = np.asarray(
        [int(s.index("A") < s.index("B")) for s in seqs],
        dtype=np.int64,
    )

    # Any bag statistic is identical for all permutations, so best constant=50%.
    bag_acc = float(max(np.mean(y == 0), np.mean(y == 1)))

    # First-layer token concept g_A(t)=1[token=A].
    # Prefix mixer p_A(t)=whether A has appeared up to t.
    # At token B, p_A is exactly the target.
    pred = []
    for s in seqs:
        seen_a = False
        value_at_b = 0
        for tok in s:
            if tok == "A":
                seen_a = True
            if tok == "B":
                value_at_b = int(seen_a)
        pred.append(value_at_b)
    prefix_acc = float(np.mean(np.asarray(pred) == y))

    return {
        "name": "prefix_breaks_permutation_invariance",
        "bag_best_accuracy": bag_acc,
        "prefix_concept_accuracy": prefix_acc,
        "n_permutations": len(seqs),
        "interpretation": (
            "Permutation-invariant pooling cannot distinguish order when the "
            "multiset is fixed. A causal prefix concept distinguishes A-before-B."
        ),
    }


def experiment_candidate_multiplicity(
    seed_offset=789,
    seeds=200,
    n_train=2000,
    n_test=5000,
    mu=0.10,
    honest_fraction=0.30,
    shortlist=16,
):
    """
    Candidate 0 is a weak true direction:
        score = mu*y + noise.
    All other candidates are pure noise.

    Search-only chooses max |correlation| on the search subset.
    Honest mode takes the best search shortlist, fixes each direction's sign
    using the search subset, and ranks candidates by
        min(search correlation, honest correlation).
    """
    results = []
    for m in (8, 32, 128, 512):
        rows = []
        for seed in range(seeds):
            rng = np.random.default_rng(seed_offset + seed)
            ytr = rng.choice([-1.0, 1.0], n_train)
            yte = rng.choice([-1.0, 1.0], n_test)

            Str = rng.normal(size=(n_train, m))
            Ste = rng.normal(size=(n_test, m))
            Str[:, 0] += mu * ytr
            Ste[:, 0] += mu * yte

            perm = rng.permutation(n_train)
            nh = int(honest_fraction * n_train)
            honest_idx = perm[:nh]
            search_idx = perm[nh:]

            corr = (
                ytr[search_idx, None] * Str[search_idx]
            ).mean(axis=0)

            # Search-only.
            j = int(np.argmax(np.abs(corr)))
            orient = 1.0 if corr[j] >= 0 else -1.0
            test_pred = np.sign(orient * Ste[:, j])
            search_test_acc = float(np.mean(test_pred == yte))

            # Honest shortlist.
            top = np.argsort(np.abs(corr))[-min(shortlist, m):]
            orientations = np.where(corr[top] >= 0, 1.0, -1.0)
            search_strength = np.abs(corr[top])
            honest_strength = np.asarray(
                [
                    np.mean(
                        ytr[honest_idx]
                        * Str[honest_idx, cand]
                        * orientation
                    )
                    for cand, orientation in zip(top, orientations)
                ]
            )
            objective = np.minimum(search_strength, honest_strength)
            jj = int(np.argmax(objective))
            jh = int(top[jj])
            oh = float(orientations[jj])
            honest_pred = np.sign(oh * Ste[:, jh])
            honest_test_acc = float(np.mean(honest_pred == yte))

            rows.append(
                {
                    "search_selected_true": int(j == 0),
                    "search_test_accuracy": search_test_acc,
                    "honest_selected_true": int(jh == 0),
                    "honest_test_accuracy": honest_test_acc,
                }
            )

        results.append(
            {
                "candidate_count": m,
                "search_true_direction_rate": float(
                    np.mean([r["search_selected_true"] for r in rows])
                ),
                "honest_true_direction_rate": float(
                    np.mean([r["honest_selected_true"] for r in rows])
                ),
                "search_test_accuracy": float(
                    np.mean([r["search_test_accuracy"] for r in rows])
                ),
                "honest_test_accuracy": float(
                    np.mean([r["honest_test_accuracy"] for r in rows])
                ),
            }
        )

    return {
        "name": "candidate_multiplicity",
        "settings": {
            "seeds": seeds,
            "n_train": n_train,
            "n_test": n_test,
            "signal_mu": mu,
            "honest_fraction": honest_fraction,
            "shortlist": shortlist,
        },
        "results": results,
        "interpretation": (
            "As candidate multiplicity grows, the maximum search-set statistic "
            "is increasingly likely to be a noise winner. Honest rechecking does "
            "not create signal, but reduces this selection bias."
        ),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output",
        default="./synthetic_proof_results.json",
    )
    p.add_argument("--seeds", type=int, default=200)
    args = p.parse_args()

    report = {
        "topk_information_loss": experiment_topk_information_loss(),
        "feature_growth_strict_expansion": (
            experiment_feature_growth_strict_expansion()
        ),
        "prefix_breaks_permutation_invariance": (
            experiment_prefix_breaks_permutation_invariance()
        ),
        "candidate_multiplicity": experiment_candidate_multiplicity(
            seeds=args.seeds
        ),
    }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for key, value in report.items():
        print("\n==", key, "==")
        if key == "candidate_multiplicity":
            for row in value["results"]:
                print(row)
        else:
            for k, v in value.items():
                if k not in ("interpretation", "topk_all"):
                    print(k, "=", v)
            print(value["interpretation"])
    print("\nSaved", path)


if __name__ == "__main__":
    main()
