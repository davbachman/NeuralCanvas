# Interpretability after reward adaptation

The revised course ends with a connected pair: students first build an SAE on
their own final preference-adapted transformer, then investigate and steer a
feature in a supplied compatible checkpoint. The preceding chain is instruction
tuning → verifiable rewards → preference rewards. The base model's learned
values are frozen during both interpretability lessons.

## Task and checkpoint continuity

`rlhf.json` retains the original answer-only QA task. Students preserve that
file, select the canonical all-position narrative corpus in a copy, and replace
the final-token output Slice with full vocabulary logits for Loss and Softmax.
This exposes 32-position narrative contexts while preserving every trained
parameter. It does not retrain, restore pretraining weights or imply a return to
pretraining performance. A value-sensitive operation fingerprint strips the QA
output Slice and verifies the same underlying computation in both forms.

Assignment 27 bundles the reward evaluator and canonical reward-task reference.
The separately submitted final policy must have a valid fitted preference reward
model, an actual policy update, independently recomputed objective improvement,
and adequate QA behavior. All SAE constructions must preserve its base-model
fingerprint. Runtime labels, cached scores and claimed training history are not
accepted as evidence. Reconstruction and sparsity are measured on the canonical
narrative split; narrative cross entropy is diagnostic after post-training.

Assignment 28 uses a class-wide supplied checkpoint so a specified feature index
has a reproducible meaning. Its SAE is trained on that exact final transformer's
residuals. The grader verifies both transformer and SAE fingerprints and the
checkpoint download hash. The student's own SAE is retained for optional
comparison, not transplanted onto the supplied model.

The fixed engineering reference's reward comparisons are explicitly synthetic,
derived from the exact story checker. They exercise the preference fitting and
policy optimization mechanism but do not constitute collected human feedback.
Students make actual comparisons in the preceding preference assignment. This
provenance distinction is preserved in the private pilot reports and instructor
guides.

## Interpretation and interventions

SAE width is 32, over a width-16 residual after the second block's feed-forward
residual addition and before final output Layer norm. Decoder rows are
constrained to unit norm. The dense and sparse runs share their starting
parameters; only activation L1 strength changes. Every required investigation
has an automatically graded connected construction.

For guided discovery, feature selection jointly considers input association and
causal output changes. The input hypothesis is a vocabulary One-hot indicator;
its on-word and off-word activation branches are checked on both splits at the
same final position. Amplification and ablation are tested on all matching
validation contexts, with separately ranked high/low activation controls and
bounds on collateral distribution and loss changes. Validation participates in
feature selection and is not described as an untouched test set.

The earlier feature-5 “now → the” findings in INTERPRETABILITY-EXTENSION.md apply
to the old narrative-only checkpoint. Current feature indices, thresholds and
measurements must come from the post-training pilots below.

## Reproduction

Run private solution-producing commands in NeuralCanvas-Coursework:

```sh
for seed in 137 211 307; do
  SAE_SEEDS="$seed" SAE_SOURCE="pilots/rewards/native/$seed/rlhf.json" \
    node scripts/assignments/interpretability/posttraining-pilot.mjs
done
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/interpretability/posttraining-screen.py
.venv/bin/python scripts/assignments/interpretability/finalize-posttraining.py
.venv/bin/python scripts/assignments/interpretability/build-posttraining-steering.py
.venv/bin/python scripts/assignments/interpretability/update-posttraining-handout.py
```

The loop uses each independently trained final policy and corresponding SAE
initialization; both sparsity settings use the same initialization within each
pair. App training, independent NumPy recomputation, actual Chromium
inspection and extracted-package execution are separate checks. Local tests do
not establish hosted Gradescope or public deployment behavior.

## Current pilot evidence

Six complete app Tensor CPU SAE runs used three independently trained final
policies and SAE initializations, with matched initial values within each dense/
sparse pair. Each run used the exact ten-epoch student recipe. The final-policy
optimization checkpoints came from the independently implemented native reward
pilot, checked against the app's objective and gradients; this is distinct from
claiming six full browser reward-training runs. Frozen transformer values stayed
exactly unchanged during SAE training. No SAE feature was inactive on validation.

| Base / SAE seed family | Dense validation MSE | Sparse validation MSE | Dense active/token | Sparse active/token |
| --- | ---: | ---: | ---: | ---: |
| 137 | 0.01881 | 0.15066 | 25.459 | 13.787 |
| 211 | 0.02042 | 0.11937 | 24.917 | 13.400 |
| 307 | 0.01791 | 0.15084 | 26.092 | 14.904 |

The post-training sparse activity limit is **16**, replacing the earlier
narrative-only limit of 12. Dense MSE ≤ 0.10, sparse MSE ≤ 0.20 and sparse activity
≤ 75% of the matched dense baseline remain supported by these runs. These are
empirical feasibility checks, not convergence guarantees for every initialization.

The supplied final-model checkpoint uses **feature 26**. Its activation
associates with final-position **entered** (training/validation AUROC
0.98189/0.98670). On all **80** matching validation contexts, mean P(**office**)
is **0.71711** unchanged, **0.78851** with amplification at α=1, and **0.47289**
with ablation. This is a context association plus a causal output effect, not
proof that the feature uniquely represents entering or factual room tracking.

High-32 ablation TV is 0.23012; low-32 TV is zero. Mean TV across all windows is
0.05265 for amplification and 0.04443 for ablation. Maximum selected activation
across every validation position is 4.54212. The bounded collateral-loss limit
is calibrated to the observed amplification increase of 0.32974; narrative loss
is diagnostic for this QA-adapted model. The full measured selection and all
source hashes are retained in private posttraining/screen.json.

The app frozen-residual cache was checked against the same uncached one-epoch
preflight: displayed objectives, losses and activity metrics were identical.
The two elapsed times on this shared cloud host were 50.47s and 241.77s; those
measurements do not establish student-device or GPU performance.

The revised graders passed 31 tests, including full-credit extracted packages,
missing-file partial credit, final-policy lineage, malformed constructions,
supplied-value checks and causal output-direction requirements. Actual Chromium
imported the deployed-format supplied checkpoint, evaluated the training split,
scanned feature contexts, built a connected complementary activation branch,
and saved/reimported all four assessment files without changing trained values.
For the recorded example prompt, P(office) was 88.92% unchanged, 92.61% amplified
and 54.99% ablated; these single-prompt values are distinct from the all-context
means above. Both handouts fit 375px and 1280px viewports, and no page errors
were recorded. The browser check built one new branch from an otherwise
completed private fixture; it did not rebuild the entire transformer or every
intervention through the UI. The four browser-saved submissions then earned
100/100 from the extracted Assignment 28 package.
