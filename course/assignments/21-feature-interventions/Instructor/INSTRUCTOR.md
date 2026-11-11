# Assignment 21 — Test features through residual interventions

This follows sparse reconstruction with causal tests of a student-selected
feature. It does not assess semantic labels or promise factual story steering.
Three independent native base/autoencoder pairs supported this separate
assignment: each had several features meeting all bounded-effect/control gates.

## Construction and interpretation

Students retain `sparse.json` unchanged and construct `zero.json`,
`amplified.json` and `ablated.json`. The three copies freeze every Param, restore
the original vocabulary Cross entropy Loss and keep the encoder attached to the
original residual. At the final second-block MLP residual, before output Layer
norm, the copies respectively use `r + 0 v_j`, `r + alpha v_j` and `r - h_j v_j`.
All input positions are intervened on; all numerical effects score the final
position. The selected direction must come from a Slice of the actual learned
decoder; ablation must use the corresponding encoder feature. Only the residual
feeding final output Layer norm is replaced. The reconstruction error is retained.

Keep scalar alpha in (0, 1], starting with 1. The same feature index is required
across the three intervention files, but no particular index is prescribed.
The original and sparse model's learned values must match across related files.
No retraining, supplied model, written response or external quiz is required.
Optional clamping or position masks are not additional assessed work.

## Deterministic assessment

The evaluator uses all 765 original narrative validation windows. It sorts their
final-position feature activations, with canonical-order tie breaking, to select
the highest and lowest 32 contexts. It independently reconstructs distributions
and calculates total variation `0.5 * sum(abs(p - q))` over all vocabulary entries.
There is no free-running generation acceptance test.

| Check | Bound |
|---|---|
| Sparse prerequisite | Generator CE ≤0.60/0.65 training/validation; SAE MSE ≤0.20 both; validation active ≤12 and inactive ≤16 |
| Selected feature | max h_j ≤4 over all 765 ×32 input positions |
| Zero restoration | maximum absolute probability difference ≤1e-7 |
| Amplification | 0<alpha≤1 and mean final-position TV ≥0.005 |
| Ablation high contexts | mean TV ≥0.05 |
| Ablation low controls | mean TV ≤0.01 |
| Collateral changes in either nonzero intervention | all-window mean TV ≤0.10 and final-position CE increase ≤0.30 |

Points: sparse prerequisite 20; zero form/restoration 10/10; amplification
form/effect 10/10; ablation form/effect-and-controls 15/15; unchanged learned values
and matching selected feature 10. Valid independent components earn partial
credit. Grader messages report measured effects and control changes.

Shape-only matching is insufficient: the grader anchors the original residual to
the preserved transformer vocabulary path, checks actual parameter values and
connections, and rejects feedback into the encoder. A vocabulary-wide bias,
disconnected branch, copied unrelated direction, extra Params or altered dataset
cannot satisfy the construction. All execution uses a bounded NumPy evaluator;
submitted code and cached claims are never executed or trusted.

## Evidence and limits

The exact native sparse models yielded 6, 8 and 8 candidate features meeting all
gates. For the chosen private references, high-context ablation TV was
0.232–0.435, low-context TV was zero, amplification all-window mean TV was
0.025–0.052, and ablation all-window mean TV was 0.018–0.032. Selected-feature maxima
were 3.07–3.68. The largest final-position CE increase was below 0.057.
These results justify bounded causal feature tests; they do not demonstrate
reproducible named concepts or beneficial factual steering. Validation contexts
used to choose a feature are not an untouched test set.

Pilots used app `d856af3da2fc411eb58252138fd51117b4bb97b1` with extension edits,
recorded per-run status and diff hashes. All three sets of saved intervention
graphs matched independently evaluated NumPy probabilities: Trace error below
6e-15 and Tensor CPU below 1.5e-6 on checked prompts. Eight-token generation
reevaluated the graph at each token, with zero restoring the baseline sequence.
See [extension validation](../../../docs/INTERPRETABILITY-EXTENSION.md).

Private fixtures and tests live in NeuralCanvas-Coursework:

```sh
.venv/bin/python -m unittest discover -s assignments/21-feature-interventions/source -p test_grader.py -v
```

Rebuild both public packages with
`python3.12 scripts/assignments/package-interpretability.py` from `course/`, then
run the public course validator. Use `../description.html` in Canvas and
`Gradescope/gradescope-21-feature-interventions.zip` in a 100-point Gradescope
Programming Assignment (Ubuntu 24.04, Python 3.12, ≥4 GB memory, 20-minute
external timeout). The bounded runner imposes additional per-worker limits.
Test saved instructor models on hosted Gradescope before issuing. A repository
push and local ZIP execution do not establish hosted or deployed-app behavior.
