# Sparse features and residual interventions

Two assignments append the existing course: [sparse reconstruction](../assignments/20-sparse-autoencoder/description.md)
and [feature interventions](../assignments/21-feature-interventions/description.md).
Assignments 1–19, their datasets, numerical targets and release packages remain
unchanged. Students use their own preserved narrative `multihead.json`, not the
QA-finetuned checkpoint. No starter model, written interpretation or external
quiz is required. The dataset browser reuses the existing narrative/all-position
preset; no new dataset or changed split was needed.

## Choice and student workflow

The attachment point is the width-16 residual after the second block's final
MLP addition, before output Layer norm. A 32-feature ReLU autoencoder adds 1,072
trainable values. Decoder rows have norm one. From the same personally built,
untrained autoencoder, students fit a zero-penalty reconstruction and a sparse
variant with activation L1 strength 0.03. Both use ten epochs, Adam 0.003, batch
32, shuffling, clipping 1, report every epoch, no early stopping and Tensor CPU.
Only training examples fit the autoencoder. Validation informs setting choices.

Students submit the unchanged adequate baseline, dense reconstruction and sparse
model. An independently constructed `h_j v_j` branch isolates a chosen feature's
contribution. The second assignment freezes all learned values, restores the
single vocabulary Loss, and constructs zero, bounded additive and feature-ablation
interventions with visible Slice, Input, Arithmetic and Matrix product blocks.
The original residual continues feeding the encoder. Each position receives the
intervention; grading measures the final-position next-token distribution.

## Pilot evidence

The private screen evaluated three independently trained transformer bases and
three autoencoder initializations per selected configuration, comparing narrative
and QA bases, 2d/4d feature widths and λ=0/0.01/0.03/0.1: 48 exploratory NumPy runs.

| Base / feature width, λ=0.03 | Validation MSE range | Active features/token |
|---|---:|---:|
| Narrative /32 | 0.0769–0.0941 | 9.12–9.52 |
| Narrative /64 | 0.0556–0.0628 | 14.03–14.81 |
| QA /32 | 0.1001–0.1115 | 11.15–11.61 |
| QA /64 | 0.0772–0.0879 | 16.21–16.60 |

The distributions and datasets differ, so these are feasibility comparisons, not
claims that one base is intrinsically better. The narrative 32-feature recipe
offers a compact construction, varied next-token contexts and fewer training
windows. Width 64 improves reconstruction but increases the feature bank and
inspection burden. The selected recipe was then trained using the actual app
runtime across all three independent base/encoder initializations.

| Exact app recipe | Validation MSE | Active features/token | Inactive features |
|---|---:|---:|---:|
| λ=0 | 0.0249–0.0280 | 19.00–19.20 | 0 |
| λ=0.03 | 0.0789–0.0884 | 8.83–8.93 | 0 |

All six runs preserved every frozen value exactly; largest decoder row-norm error
was below 8.6e-8. Observed wall times were roughly 4–6 minutes per run while three
jobs shared five cloud vCPUs. They are not laptop or hardware-GPU guarantees.

All three sparse models had multiple bounded intervention candidates (6/8/8).
Chosen private references gave high-context ablation total variation 0.232–0.435,
low-context TV zero, additive all-window mean TV 0.025–0.052, ablation mean TV
0.018–0.032, maximum feature activation 3.07–3.68, and final-position CE increase
below 0.057. This supports a separate causal-intervention assignment. It does
not establish reproducible named concepts or beneficial factual steering.

## Grading targets

The preserved generator must satisfy the existing 0.60/0.65 train/validation CE
limits. Autoencoder MSE must be ≤0.10 dense and ≤0.20 sparse on both splits.
Sparse validation activity must be ≤12, at most 75% of the dense activity, with
≤16 inactive features. Active means exact nonzero ReLU output, tolerance zero.
The unit-row-norm tolerance is 5e-5; accepted sparse λ lies in [0.01,0.10].

Interventions use all 765 original validation windows. The 32 highest/lowest
final-position activations define feature-specific effect/control contexts, with
canonical-order ties. Zero restoration tolerance is 1e-7 in probability. Additive
strength lies in (0,1] and mean TV is ≥0.005. Ablation requires max activation ≤4
over all positions, high-context mean TV ≥0.05 and low-context mean TV ≤0.01.
Both nonzero interventions require all-window mean TV ≤0.10 and final-position
CE increase ≤0.30. Feature indices are not standardized between students.

Bounded graders independently evaluate canonical data, actual operations,
parameter values and ancestry. No submitted code executes. They reject altered
datasets, cached-score claims, trainable targets, scaling loopholes, shape-only
residual aliases, feedback into the encoder, disconnected interventions and
unrelated directions. Valid components earn independent partial credit.

## App semantics and validation scope

Saved Params have trainable/frozen controls. Freezing skips initialization,
SGD/Adam/AdamW and decay without blocking gradients through an operation. Stable
Dropout is separate. Internal targets detach their gradients and require frozen,
deterministic ancestry. Mean reconstruction and activation L1 share valid-token
masking; padding is excluded, zero subgradients are defined, and empty masks are
rejected. Decoder directions project after initialization and updates, with safe
zero/large-value handling. Save/import preserves values and settings. Reports
separate data loss, activation penalty and total objective, and expose exact-zero
activity diagnostics. Feature inspection scans bounded examples with cancellation.

Pilots identify app `d856af3da2fc411eb58252138fd51117b4bb97b1` plus the extension's
uncommitted changes; private reports retain status and diff hashes. Later
validation preflight adds a first-epoch forward pass for malformed-objective
safety. Final numerical checks use the updated engine; no new convergence claim
is inferred from this safety change.

- Trace/Tensor finite differences, target detachment, masks, optimizer updates,
  frozen initialization, norm extremes, persistence and objective-reference
  copying have focused regression coverage.
- All three final intervention sets passed independent NumPy/Trace/Tensor CPU
  probability checks and eight-token generation; maximum checked discrepancies
  were below 6e-15 Trace and 1.5e-6 Tensor. Zero restored baseline generation.
- Actual Chromium verified freezing/randomizing, Loss/Param controls, one full
  CPU training epoch, inference, feature inspection, save/reimport, visible
  feature-contribution wiring and deterministic intervention generation. Complete
  transformer/SAE graphs were imported private fixtures; the browser check does
  not claim a complete model was built from a blank canvas. New handouts were
  checked at 375px and 1280px widths.
- Chromium CPU versus SwiftShader WebGL checks covered masked SAE losses,
  gradients, SGD/AdamW updates, norms and exact frozen values on toy and actual
  reference batches. Prediction differences were ≤2.21e-6 and update differences
  ≤5.36e-7. This is software-WebGL compatibility, not physical GPU performance.
- Actual PyTorch execution checks cover export semantics. Exported training
  programs require Trace/SGD settings; Tensor optimizer exports remain explicitly
  unsupported. Experimental WebGPU was not validated for this recipe.
- All three reference sets earn 100/100 on both graders. The 22 private grader
  regressions include full-credit and malformed/partial extracted-ZIP execution.
  The public course validator checks all 21 releases and 26 dataset references.
  Actual browser-saved submissions, including the visibly rewired contribution,
  also earn 100/100 in both extracted packages.

The final consolidated app suite passes 1,179 tests across 116 files, with 34
tests and one file skipped. Separate actual PyTorch execution checks cover the
export runtime. Build, lint and detailed results are recorded in private validation
evidence. Hosted Gradescope and the deployed student site were not tested; no
Canvas or Gradescope upload was performed. A Git push is not evidence of either.

## Reproduction and artifacts

Public handouts, guides, fixtures and current ZIPs are in the two assignment
folders. From `course/`, rebuild using
`python3.12 scripts/assignments/package-interpretability.py`, then run
`python3.12 scripts/validate-course.py`. Handouts remain canonical HTML; generate
Markdown and student pages with `python3.12 course/scripts/sync-handouts.py` from
the app root. Use Ubuntu 24.04 with Python 3.12 for a new hosted grader.

Private `pilots/interpretability/` contains screen/native reports, hashes,
trained references, engine/browser checks and grading evidence. Reproduction
scripts are in `scripts/assignments/interpretability/`; solution-dependent tests
are in the two private assignment folders. Private Git history and trained values
are never copied into public artifacts.
