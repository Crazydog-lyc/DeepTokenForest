# Deep Token Forest — no distillation, no backpropagation

This bundle is a replacement/extension for the `tokentopk` AG News experiment.

It does **not** use teacher logits, soft labels, attention maps, hidden targets from
later layers, or any other form of distillation. The only supervised target is
the original AG News label. The frozen Pythia prefix is used only to produce the
starting token representation.

## Files

- `deep_token_forest.py` — model implementation.
- `train_agnews_deep_token_forest.py` — full AG News training/caching/logging.
- `synthetic_proofs_experiments.py` — reproducible structural counterexamples.
- `synthetic_proof_results.json` — results already produced by that script.
- `run_agnews_ablation.py` — controlled AG News ablation launcher.
- `analyze_deep_token_forest_run.py` — post-hoc diagnostics.
- `PATCH_NOTES.md` — how to place these files into the current repository.

The existing `FrozenLMHiddenState.py` from the `tokentopk` branch is reused
unchanged.

---

# 1. New node: projection distribution instead of Top-K

For document `i`, token `t`, representation `h_it`, and a learned direction `w`:

```text
q_it = h_it^T w
```

A node now asks for one point of the empirical projected survival function:

```text
c_i(w,b) = (1/T_i) sum_t 1[q_it >= b]
```

and routes by

```text
right iff c_i(w,b) >= rho.
```

So an internal node learns three objects:

```text
w    token-space direction
b    token-level threshold
rho  document-level fraction threshold
```

The implementation can also use a smooth survival statistic, but `hard` is the
default because it has the cleanest interpretation.

## Why this strictly contains more information than Top-K in the ideal limit

Fix `w`. Let the projected token multiset be

```text
Q_w(H) = {h_1^T w, ..., h_T^T w}.
```

Its empirical CDF

```text
F_w(b) = (1/T) sum_t 1[h_t^T w <= b]
```

for all `b` uniquely determines the sorted projected values, hence every order
statistic and therefore every Top-K average. Thus an *entire* projected CDF can
recover Top-K statistics.

The converse is false: Top-K does not determine the middle or lower order
statistics. The synthetic counterexample in this bundle makes the middle token
carry the label while the largest and smallest four projected values are the
same in both classes.

Important nuance: the code evaluates a **finite grid of token thresholds**, not
the continuum of all `b`. Therefore the implemented finite model is an
approximation to the full projected-distribution argument. Increase
`--token-threshold-quantiles` if this discretization is the bottleneck.

### Executed synthetic result

```text
Top-K best test accuracy: 0.5144
Projection survival test accuracy: 1.0000
```

This is a structural counterexample, not an AG News benchmark.

---

# 2. No-BP feature transformation

After one stage has learned trees, high-value internal nodes become reusable
token concepts.

For a selected node `(w_j,b_j)`:

```text
g_j(h_t) = tanh((w_j^T h_t - b_j) / tau)
```

The next-stage token representation is

```text
H^(s+1) = concat(H^(s), g_1(H^(s)), ..., g_M(H^(s))).
```

The concept channels are standardized using training-token statistics before
they are appended.

## Capacity cannot decrease under concatenation

Let `Phi(H) = [H, G(H)]`. Every old oblique direction `w` can be embedded into
the new space as

```text
w' = [w, 0].
```

Therefore every node available before growth remains available after growth:

```text
F_s subseteq F_(s+1).
```

This is a representational statement, not a promise that greedy optimization or
test error must improve.

## The inclusion can be strict

If `g(H)` is nonlinear, a linear/oblique split in `[H,g(H)]` corresponds to a
piecewise nonlinear decision surface in the original `H` space.

A concrete 1-D example is

```text
positive iff x < -1 OR x > 1.
```

One raw threshold is a half-line, so it cannot represent two disjoint tails.
Create

```text
g1 = 1[x > 1]
g2 = 1[x < -1]
```

and the next-stage single split

```text
g1 + g2 >= 0.5
```

represents the union.

### Executed synthetic result

```text
Raw one-threshold test accuracy: 0.7526
One threshold after concept growth: 0.9999
```

Again, this proves strict function-class expansion in a controlled case; it does
not prove that AG News validation loss must improve.

---

# 3. Optional causal prefix mixing

For future language-model/generation work, token-local nonlinear features alone
are not sufficient: a generic sequence model must be able to transform a token
using information from other positions.

With `--prefix-mixing`, every learned concept also produces

```text
p_tj = mean_(s<=t) g_j(h_s).
```

The next representation appends both token-local concept features and causal
prefix means:

```text
h_t^(s+1) = [
    h_t^(s),
    g_1(h_t), ..., g_M(h_t),
    p_t1, ..., p_tM
].
```

This is a simple, causal, tree-created token mixer. It is not claimed to be a
replacement for attention; it is an experimentally testable first step.

## Why it changes the function class

Any aggregation depending only on the unordered token multiset is permutation
invariant. It cannot distinguish two sequences with the same tokens in
different orders.

For all permutations of `A,B,C,D`, define

```text
y = 1 iff A occurs before B.
```

All bag statistics are identical. The prefix feature “has A appeared yet?”,
evaluated at token `B`, exactly determines the label.

### Executed synthetic result

```text
Best permutation-invariant bag accuracy: 0.5000
Causal prefix concept accuracy: 1.0000
```

---

# 4. Supervision and Newton boosting are still pure label supervision

At global tree iteration `m`, current logits are

```text
F_m(x).
```

For multiclass cross entropy:

```text
g_ic = p_ic - 1[y_i=c]
h_ic = p_ic (1-p_ic)
```

using the same diagonal multiclass Hessian approximation as the original code.

A leaf value is

```text
v_c = -G_c / (H_c + lambda).
```

For the second-order regularized surrogate, the optimum leaf value reduces the
surrogate by

```text
0.5 * sum_c G_c^2/(H_c+lambda).
```

A split is accepted according to the left-plus-right Newton score minus the
parent score and `gamma`.

Nothing is propagated backward through previous stages. A later stage simply
receives a new explicit representation produced by already-learned tree
concepts.

---

# 5. Honest split selection

The old failure mode we specifically want to test is candidate multiplicity:

```text
weak late residual
+ many adaptively searched directions/thresholds
-> training winner may be a noise winner.
```

The new tree can split the node's training rows into:

```text
search rows
honest rows
```

Candidate directions and thresholds are learned on `search rows`. Only the top
few proposals are evaluated on `honest rows` using exactly the already-fixed
`(w,b,rho)`. The validation set is never used for this process.

The robust priority is proportional to the weaker of search and honest
per-sample Newton gains. Set

```bash
--honest-fraction 0
```

to disable it.

### Executed weak-signal simulation

For one real weak direction plus many noise directions:

| candidates | search selects true | honest selects true | search test acc | honest test acc |
|---:|---:|---:|---:|---:|
| 8 | 0.950 | 0.925 | 0.5380 | 0.5368 |
| 32 | 0.875 | 0.925 | 0.5340 | 0.5364 |
| 128 | 0.825 | 0.835 | 0.5333 | 0.5336 |
| 512 | 0.660 | 0.755 | 0.5266 | 0.5300 |


The important observation is not that honest selection magically creates
signal—it does not. The point is that the false-winner problem becomes worse as
the candidate bank grows, and independent in-training checking can reduce it.

---

# 6. PCA is treated as a separate hypothesis

No later transform can reconstruct information destroyed by PCA.

If `P` is the PCA projection and two original representations satisfy

```text
P H1 = P H2,
```

then any later deterministic tree representation `Phi` also satisfies

```text
Phi(P H1) = Phi(P H2).
```

Therefore feature growth and PCA information loss must be tested separately.

The provided ablation suite includes:

```text
64 -> 128 -> 256 -> full width
```

with the rest of the model fixed.

---

# 7. Recommended AG News experiments

## First smoke test

From the repository root, after copying these files beside
`FrozenLMHiddenState.py`:

```bash
python train_agnews_deep_token_forest.py \
  --dataset-dir ./data/ag_news \
  --checkpoint ./checkpoints/pythia-70m \
  --output-dir ./runs/dtf_smoke \
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

## Controlled 100-tree comparison to the current experiment

### A. Fixed representation; distribution node only

```bash
python train_agnews_deep_token_forest.py \
  --output-dir ./runs/dtf_A_fixed \
  --token-proj-dim 64 \
  --n-stages 4 \
  --trees-per-stage 25 \
  --disable-feature-growth \
  --honest-fraction 0 \
  --overwrite-output
```

This asks only: **is Top-K itself the bottleneck?**

### B. Same fixed representation + honest candidate checking

```bash
python train_agnews_deep_token_forest.py \
  --output-dir ./runs/dtf_B_honest \
  --token-proj-dim 64 \
  --n-stages 4 \
  --trees-per-stage 25 \
  --disable-feature-growth \
  --honest-fraction 0.15 \
  --overwrite-output
```

This asks: **how much of the late train/validation divergence is split-selection
overfitting?**

### C. Token concept feature growth

```bash
python train_agnews_deep_token_forest.py \
  --output-dir ./runs/dtf_C_growth \
  --token-proj-dim 64 \
  --n-stages 4 \
  --trees-per-stage 25 \
  --concepts-per-stage 16 \
  --honest-fraction 0.15 \
  --overwrite-output
```

This is the clean test of the **fixed-H bottleneck**.

### D. Add causal prefix mixing

```bash
python train_agnews_deep_token_forest.py \
  --output-dir ./runs/dtf_D_prefix \
  --token-proj-dim 64 \
  --n-stages 4 \
  --trees-per-stage 25 \
  --concepts-per-stage 16 \
  --honest-fraction 0.15 \
  --prefix-mixing \
  --overwrite-output
```

This tests whether an explicit tree-created token mixer adds anything on AG
News. Because the starting Pythia hidden state is already contextualized, it is
entirely possible that this gives little AG News benefit while still being
important for future replacement of more LM layers.

## Run the automated core ablation

```bash
python run_agnews_ablation.py \
  --suite core \
  --dataset-dir ./data/ag_news \
  --checkpoint ./checkpoints/pythia-70m \
  --output-root ./runs/dtf_ablation_core
```

## PCA ablation

```bash
python run_agnews_ablation.py \
  --suite pca \
  --dataset-dir ./data/ag_news \
  --checkpoint ./checkpoints/pythia-70m \
  --output-root ./runs/dtf_ablation_pca
```

The `full` PCA variant is computationally expensive because every token retains
the entire Pythia hidden width.

---

# 8. What would count as evidence for each hypothesis?

Do not judge only by final validation loss.

### H1 — Top-K bottleneck

Supported if fixed-H projection-distribution trees beat the original Top-K model
under comparable tree/leaf/search budgets.

Most convincing signature:

```text
same H
same number of trees
distribution model:
  lower val loss
  better Business <-> Sci/Tech confusion
  no larger or smaller late transfer collapse
```

### H2 — fixed representation bottleneck

Supported if model C (feature growth) beats model B (fixed H) with the same base
features, tree count and node type.

Stronger evidence is:

```text
stage 2/3:
  validation loss improves after new channels appear
  late val/train transfer ratio rises
  tree gain remains useful on honest rows
```

A lower train loss alone is **not** enough.

### H3 — selection overfitting

Supported if honest checking gives:

```text
train loss slightly worse or similar
validation loss better
late transfer ratio better
fraction of positive honest gains higher among accepted nodes
```

### H4 — PCA bottleneck

Supported if increasing 64 -> 128 -> 256/full improves both train and validation
and especially reduces the stable Business/Sci-Tech confusion.

If train improves while validation does not, extra retained PCA information is
not by itself solving the generalization problem.

### H5 — sequence-mixing bottleneck

Supported on AG News only if prefix mixing improves over identical token-local
growth. More importantly, the supplied order counterexample proves that some
form of token mixing is necessary if the architecture is eventually expected to
represent general order-sensitive sequence functions without relying on the
remaining neural layers.

---

# 9. Diagnostics written during training

Every run contains:

```text
metrics_deep_token_forest.json
experiment_config.json
deep_token_forest_model.pkl
predictions_deep_token_forest.npz

diagnostics/
  training_history.jsonl
  model_structure.json
  tree_s00_t000.json
  tree_s00_t001.json
  ...
  concepts_stage_00.json
  concept_stats_train_stage_00.json
  concept_stats_val_stage_00.json
  concept_top_tokens.json
```

Each internal tree node records:

```text
node_id
depth
n_samples
w                       full direction vector
token_threshold b
document_threshold rho
search_gain
honest_gain
effective_gain
left/right subtree
```

Each training-history row records:

```text
global tree index
stage
representation dimension
train/validation loss
train/validation improvement
validation/train transfer ratio
depth/leaves/node count
mean and summed split gains
```

Each selected feature records:

```text
source stage/tree/node
w
b
gain
training and validation activation statistics
class-conditional document activation fraction
decoded high-activation tokenizer tokens
```

Run:

```bash
python analyze_deep_token_forest_run.py ./runs/dtf_C_growth
```

to create `posthoc_analysis.json`.

---

# 10. Important limitations of this version

1. **No theorem guarantees lower AG News test loss.** The proofs establish
   information preservation / strict function-class expansion. Generalization is
   empirical.

2. **Greedy supervised feature growth can miss latent features with zero
   immediate label gain.** This is a genuine difference from end-to-end BP. If a
   future task requires concepts that are useful only after several compositions,
   an additional self-supervised/reconstruction or diversity objective may be
   needed. That should be introduced as a separate ablation, not hidden inside
   the AG News model.

3. **Finite projection thresholds approximate the full projected distribution.**
   Seven quantiles are the default for compute reasons.

4. **Diagonal multiclass Hessian remains an approximation.** This redesign does
   not claim that it is the optimal multiclass Newton geometry.

5. **Prefix mean is only a first sequence mixer.** It proves the architecture can
   construct order-sensitive features without BP; it is not yet a general
   substitute for attention.

---

# 11. Relation to prior tree representation learning

The architecture is conceptually related to, but not copied from:

- Zhou & Feng, *Deep Forest: Towards an Alternative to Deep Neural Networks*,
  IJCAI 2017.
- Feng, Yu & Zhou, *Multi-Layered Gradient Boosting Decision Trees*,
  NeurIPS 2018.

Those works demonstrate that non-differentiable tree ensembles can be organized
into multi-layer representation-learning systems without ordinary neural
backpropagation. Here the representation is specifically **token-dependent** and
the stage features are generated from projection-distribution tree concepts.
