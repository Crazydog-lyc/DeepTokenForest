# Patch notes

Place these files in the root of the current `tokentopk` branch, beside
`FrozenLMHiddenState.py`:

```text
deep_token_forest.py
train_agnews_deep_token_forest.py
synthetic_proofs_experiments.py
run_agnews_ablation.py
analyze_deep_token_forest_run.py
README_DEEP_TOKEN_FOREST.md
```

No change to `FrozenLMHiddenState.py` is required.

For an automated Top-K baseline inside `run_agnews_ablation.py`, keep the
existing repository file `train_agnews_token_topk_tanh.py` beside the runner and
pass `--include-topk-baseline`.

Recommended first command:

```bash
python synthetic_proofs_experiments.py

python train_agnews_deep_token_forest.py \
  --max-train-samples 4000 \
  --max-val-samples 1000 \
  --max-test-samples 1000 \
  --token-proj-dim 64 \
  --n-stages 2 \
  --trees-per-stage 3 \
  --max-leaves 8 \
  --max-depth 4 \
  --concepts-per-stage 8 \
  --overwrite-output
```

Then run the controlled core ablation described in README_DEEP_TOKEN_FOREST.md.
