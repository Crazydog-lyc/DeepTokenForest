# Selector V2 — evidence-driven concept selection

This variant keeps the original filenames.

New CLI:
- `--concept-selector legacy|gain_only|activation_logdet`
- `--concept-min-coverage 0.10`
- `--concept-selector-sample-tokens 4096`
- `--concept-logdet-gamma 3.0`

Recommended causal ablation:
1. Keep the already-run 10x10 legacy result as baseline.
2. Re-run exactly the same 10x10 command with only:
   `--concept-selector activation_logdet`
3. Do not change stage count, tree count, PCA, search budget, seed, or prefix mixing.
4. Only after that compare 5x20 vs 10x10.

The activation-logdet selector:
- uses TRAIN features only;
- uses no teacher, distillation, validation labels, or backprop;
- filters tiny-coverage nodes;
- computes candidate tanh activations on a fixed train-token sample;
- residualizes candidate activations against the current H linear span;
- selects high-gain, nonredundant residual activations by weighted log-det.

`legacy` exactly preserves the old selector for reproducibility.
