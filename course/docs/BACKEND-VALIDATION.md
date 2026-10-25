The current whole-story Assignments 9 and 10 use the five-epoch Tensor CPU

Numbering note: this dated report retains its original assignment numbers. See
[SAMPLING-SEQUENCE.md](SAMPLING-SEQUENCE.md) for the current sequence after moving sampling to Assignment 12.
recipe validated in [WHOLE-STORY-REVISION.md](WHOLE-STORY-REVISION.md). The
30-epoch structured-fact measurements below concern the preceding architecture.

Historical structured-football update (2026-10-20): Assignments 9 and 10 also completed their full 30-epoch recipes on WebGL, in about five minutes in concurrent software-rendered browser checks. Auto is allowed with Tensor CPU as a tested fallback; see the current football report.

Current football capstone update (2026-10-20): the supplied Assignment 20 start completes the full ten-epoch recipe on Tensor CPU and WebGL, with 100% on both football-order halves and independent full-credit saved models. Auto is allowed. See [current football evidence](FOOTBALL-SEQUENCE-REVISION.md); software WebGL timing is not a physical-GPU benchmark.

# Coursework backend validation

Current football update (2026-10-20): [FOOTBALL-SEQUENCE-REVISION.md](FOOTBALL-SEQUENCE-REVISION.md) describes the implemented Assignments 7–10 and 20. Their earlier dataset/backend statements below are historical. The generation evidence for Assignments 11–19 is retained.

For the current synthetic-narrative Assignments 11–20, see [NARRATIVE-REVISION.md](NARRATIVE-REVISION.md). The measurements and earlier sequence below are historical evidence; they do not validate the replacement dataset.

This report preserves the sequence and evidence from before the stacking revision. For current numbering, releases and fresh two-block validation, see [Stacking revision](STACKING-REVISION.md). Historical measurements are not new two-block results.

The new app defaults are Tensor engine, Adam, Auto, early-stopping patience 0 and minimum validation improvement 0; gradient norm limit remains 1. Assignment-specific learning rates, batch sizes, epochs and initializers still apply. Assignments 7–8 require clipping off (0); later trained text models use 1.

## Guidance across the course

| Assignments | Execution and optimizer | Backend | Gradient norm limit |
|---|---|---|---:|
| 1–5 | Change to Trace engine · SGD | Not applicable in Trace mode | Not applicable in Trace mode |
| 6 | Tensor engine; change Adam to SGD | Change Auto to Tensor CPU | Change 1 to 0 |
| 7–8 | Tensor engine; Adam | Auto, or Tensor CPU | Change 1 to 0 |
| 9–20, where new training is required | Tensor engine; Adam | Auto, or Tensor CPU | Keep 1 |

Keep early-stopping patience and minimum validation improvement at 0 for Tensor training. Assignments 11, 19 and 20 reuse learned weights, and Assignment 17's comparison copy must also retain its weights; those steps must not be randomized or retrained. Assignment 8 uses the same backend for both ten-epoch comparisons.

Assignments 7–10 and 12–18 include an optional speed tip: try Tensor CPU when Auto feels slow, keeping the other training settings. This follows the instructor’s report that Assignment 9 ran about twice as fast with Tensor CPU on their computer. It is not a hardware benchmark for the other assignments or a universal speed guarantee; both backends already have convergence evidence. If Assignment 8 uses CPU for either model, both ten-epoch comparisons use CPU.

The early handouts now explicitly identify the overrides from the new defaults. Six production-browser checks verified their controls and File → Save settings on the Tensor-default app revision below, with no page errors and no learned parameter changes. These checks do not retrain Assignments 1–6: their issued recipes and original training evidence are preserved. See `early-training-defaults-browser.json`.

All twenty handouts render without horizontal overflow at 390- and 960-pixel widths (`backend-handout-checks.json`). Updated handout hashes are recorded in the assignment manifests. Publishing this update requires replacing the changed Canvas descriptions; the existing Gradescope ZIPs remain byte-for-byte unchanged.

## Scope and environment

Fresh browser training uses clean app `a0b69068c566c1fe372a7df4600d90073c6443ca`, Chromium 151, and Auto selecting WebGL. Production UI checks use the subsequent Tensor-default commit `51e00c7c64d6b68624a7d02a246479530c88b09c`; its numerical training code is unchanged. The cloud renderer is ANGLE/SwiftShader, a software WebGL implementation. This validates the WebGL calculation and training path, not physical-GPU acceleration, laptop runtime, or every browser/driver. WebGPU is experimental and is not selected by Auto or validated here.

The issued CSV/prepared datasets, graph architectures, optimizer, learning rates, batch size, initialization methods, reporting intervals and epoch budgets are retained. No grade thresholds, rubrics, sources or upload ZIPs are changed. Existing CPU reference pilots remain intact. Saved projects can override new defaults, so handouts continue to name required settings.

## Completed training coverage

Thirty full-budget WebGL training runs cover all thirteen trained model forms in Assignments 7–20. Both parts of Assignments 7 and 8 have three seeds; their four production UI runs additionally verify the new defaults, actual training buttons, held-out inference and File → Save. The larger transformer variants use the full WebGL runs listed below plus their existing three-seed CPU evidence; no claim of three new WebGL runs for every architecture is made.

| Assignment | Model | Epochs per run | WebGL seeds | Held-out loss |
|---|---|---:|---|---:|
| 7 | reviews | 5 | 137, 211, 307 | 0.5150–0.5252 |
| 7 | stories | 100 | 137, 211, 307 | 0.7244–0.7249 |
| 8 | embedding | 10 | 137, 211, 307 | 0.0005–0.0009 |
| 8 | onehot | 10 | 137, 211, 307 | 0.7113–0.9213 |
| 9 | baseline | 40 | 137, 211, 307 | 5.5763–5.9419 |
| 10 | attention | 40 | 137, 211, 307 | 0.0055–0.0555 |
| 12 | bigram | 5 | 137, 211, 307 | 2.3219–2.3258 |
| 13 | context | 5 | 137, 211, 307 | 1.9738–1.9947 |
| 14 | attention | 5 | 137 | 2.2315 |
| 15 | residual | 5 | 137, 211 | 2.0848–2.0982 |
| 16 | transformer | 5 | 137 | 2.1366 |
| 17 | all_positions | 5 | 137 | 2.1221 |
| 18 | multihead | 5 | 137 | 2.0980 |

All these saved models receive full credit from their independent graders, which agree with browser losses within 0.00002. Assignment 9's high held-out loss remains informational and is part of the intended comparison with attention. Assignment 7's reviews achieve 74.5–75.5% accuracy and stories 50%; Assignment 8's one-hot models achieve 45.4–64.6% and embedding models 100% after ten epochs.

The existing ZIPs for all fourteen assignments are also exercised with complete submissions. Assignments 11, 19 and 20 reuse WebGL-trained weights in their required comparisons; Assignment 17 includes its unchanged-weight final-position copy. The checks do not retrain these copies. See `backend-validation.json`, `backend-packaged-results.json`, and each assignment's private evidence.

## Reproduction

Use a separate clean app checkout with `npm ci`, a Vite development server for engine checks, and a production build/preview for UI checks. Set `NEURALCANVAS_ROOT`, `PLAYWRIGHT_ROOT`, and optionally `CHROMIUM_PATH` and `APP_URL`. The Chromium flags deliberately select SwiftShader for reproducible cloud WebGL coverage.

```sh
BACKEND_CASES=07-,08-,09-,10-,12-,13- BACKEND_SEEDS=137,211,307 node scripts/assignments/validate-backends.cjs
BACKEND_CASES=15- BACKEND_SEEDS=137,211 node scripts/assignments/validate-backends.cjs
BACKEND_CASES=14-,16-,17-,18- BACKEND_SEEDS=137 node scripts/assignments/validate-backends.cjs
EXPECTED_DEFAULT_ENGINE=tensor node scripts/assignments/browser-backend-settings.cjs
node scripts/assignments/browser-early-training-settings.cjs
node scripts/assignments/check-backend-fallback.cjs
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/grade-backend-validation.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/check-backend-packages.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/summarize-backend-validation.py
```

The engine runner defaults to the local development server on port 5183; the text UI runner defaults to the production preview on port 5184. For the two UI scripts, set `APP_URL=http://127.0.0.1:5185/NeuralCanvas/` and `NEURALCANVAS_ROOT` to the Tensor-default app checkout used for that preview. All use the app's `/NeuralCanvas/` base. `BACKEND_CASES` filters comma-separated folder/part substrings; `BACKEND_SEEDS` defaults to 137,211,307. The commands above reproduce the recorded coverage; omitting the filters requests all three seeds for every architecture. Completed pilot files are retained unless `BACKEND_REPLACE=1` is set. Initial weights are restored by ID or display name with a completeness assertion; older pilots lacking initial weights are rebuilt with the existing seeded course recipe builder. Original trained weights are never a fresh-training starting point.

Saved models and per-epoch records live in each assignment's private `reference/backend-validation/`. The independent graders recompute metrics without trusting browser reports. The package check derives the intervention, temperature and final-position copies without changing learned weights, then executes each existing upload ZIP. UI checks start from editable untrained reference graphs; they do not simulate drawing every wire. The deployed website and hosted Gradescope are not exercised.

A separate fallback check removes WebGL registration inside its isolated browser and confirms Auto completes an epoch on CPU with a reported fallback reason. This is a failure-path check, not a full assignment training pilot; see `backend-fallback.json`.
