# Person-location warm-up for football questions

This private experiment follows the [short-story pilot](FOOTBALL-SHORT-PILOT.md).
It tests whether learning where both people are before learning the football
questions improves reliability. Current assignment releases are unchanged.
For the subsequent decision across all candidates, see the
[consolidated Assignment 20 recommendation](ASSIGNMENT-20-RECOMMENDATION.md).

## Result: do not adopt the warm-up

The proposed curriculum did **not** give a reliable improvement. It rescued the
previously failed seed 211, but weakened three other paired runs. Across the five
fixed initializations:

| Criterion | Person-location curriculum | Restart control |
|---|---:|---:|
| At least 80% in each of the four question categories | 3 / 5 | 3 / 5 |
| Also at least 80% on drop-before-moving questions | **1 / 5** | **3 / 5** |

These are counts of the tested initializations, not estimates of a guaranteed
student success rate. Neither recipe meets the reliability goal for an assignment.
The extra UI run uses seed 211 again and is not counted as a sixth independent
initialization.

Each main category contains 480 held-out questions, and each overall score covers
1,920. The drop-before-moving subset has 120 questions. "Pickup person" and
"other person" are analysis categories; students' prompts ask about Nora or Liam
by name. The pickup person may have dropped the ball by the end.

### Person-location curriculum

| Seed | Football | Pickup person | Other person | Holder | Overall | Drop before moving |
|---|---:|---:|---:|---:|---:|---:|
| 137 | 88.3% | 100% | 100% | 100% | 97.1% | 53.3% |
| 211 | 100% | 100% | 100% | 100% | 100% | 100% |
| 307 | 88.5% | 97.3% | 27.5% | 100% | 78.3% | 54.2% |
| 419 | 87.7% | 79.8% | 24.8% | 100% | 73.1% | 50.8% |
| 503 | 94.0% | 99.8% | 100% | 100% | 98.4% | 76.7% |

### Matched restart control

| Seed | Football | Pickup person | Other person | Holder | Overall | Drop before moving |
|---|---:|---:|---:|---:|---:|---:|
| 137 | 100% | 100% | 100% | 100% | 100% | 100% |
| 211 | 86.0% | 98.8% | 41.7% | 100% | 81.6% | 48.3% |
| 307 | 100% | 100% | 100% | 100% | 100% | 100% |
| 419 | 87.1% | 62.5% | 32.5% | 100% | 70.5% | 48.3% |
| 503 | 100% | 100% | 100% | 100% | 100% | 100% |

Seed 137 illustrates why the extra diagnostic matters: the curriculum model gets
97.1% overall and clears every broad category, but answers only **64 / 120**
drop-before-moving questions correctly. Seed 503 also clears every broad
category while answering only **92 / 120** of those cases correctly.

The person-location errors show confusion about which person's room to return.
For curriculum seed 307, all 348 wrong other-person answers name the pickup
person's initial room (167) or final room (181). For curriculum seed 419, the
same breakdown is 60 initial and 301 final. For control seed 211, all 280 wrong
other-person answers name the pickup person's initial room. These are patterns
in predicted answers, not proof of a particular internal attention mechanism.

The isolated seed-211 curriculum timing is **109.8 + 407.1 = 516.9 seconds
(8.6 minutes)** of fine-tuning, after generative pretraining. The successful
actual-UI run is reported separately below. A fast successful initialization
does not establish a reliable recipe.

A useful next experiment would keep all question types mixed from the first
update, while modestly oversampling the two difficult cases: the other person's
location and a ball dropped before moving. That would test extra practice without
an initial person-only phase. It is a proposal, not a validated improvement;
keep the architecture, compute budget, grouped split, and per-category checks
controlled. Do not revise assignment datasets or grading targets on the basis
of this unsuccessful warm-up.

## Controlled comparison

Both arms start from the same ten-epoch generative checkpoint for each native
initialization. They use the same six-room stories, vocabulary, split, two-block
two-head transformer, 5,689 parameters, Adam 0.003, batch 32, clipping norm 1,
zero weight decay, and disabled early stopping.

| Arm | First phase | Second phase | Examples processed | Updates |
|---|---|---|---:|---:|
| Person-location curriculum | 6 epochs × 1,920 person questions | 12 epochs × 3,840 mixed questions | 57,600 | 1,800 |
| Restart control | 3 epochs × 3,840 mixed questions | 12 epochs × 3,840 mixed questions | 57,600 | 1,800 |

Both restart Adam after 360 updates. The second phase has identical data and
shuffle order. The control distinguishes a curriculum benefit from the benefit
of an optimizer restart. Both have the same example/update budget as the earlier
uninterrupted fifteen-epoch fine-tuning run. Their extra reporting boundary adds
some evaluation overhead.

Matching total updates does not match each category's exposure. The curriculum
sees 17,280 examples of each person-location category and 11,520 of each football
category; the control sees 14,400 of each of the four categories. Thus this tests
the curriculum recipe as a whole, including extra practice on person locations.
It cannot attribute a difference solely to ordering versus category frequency.

The person subset filters the existing canonical documents without regenerating
stories, changing targets, or moving examples between splits. It includes
questions about both people, including the person who never carries the ball.
There are 1,920 training and 960 validation person questions; the mixed task has
3,840 training and 1,920 validation questions. The split keeps complete room-layout
groups together. These held-out layouts have been used for pilot selection, so
they are not an untouched final test.

The five fixed native seeds are 211, 137, 307, 419, and 503. The first three reuse
the existing native generative checkpoints. The remaining two receive ten epochs
of native generative pretraining. Every arm uses its prescribed final checkpoint,
including failures; there is no best-epoch or successful-seed selection.

The acceptance diagnostic is at least 80% in every question category plus at
least 80% on the 120 football questions where dropping precedes moving. The
diagnostic is not a released grading threshold.

## Execution and workflow limits

The pilot runs Neural Canvas's native Tensor CPU training and serialization
modules in Chromium, using clean app commit
`2b6b96d8d07ee52e0beca3c3432f9f4e5819af79`. It does not substitute PyTorch training.
The width-16 position table remains 32 × 16 when adapting the 23-token question
prompts. Saved parameters must match exactly across each phase boundary.
The shared transformer builder sets `fixedLength: false` for variable-length
inference; the language-training examples are still full context windows.
The UI replacement imports the canonical `fixedLength: true` setting. Every
question window contains exactly 23 input tokens in either case, so this does
not change padding, targets, or the training examples.

Seed 211's curriculum arm is an isolated timing run. Replications run concurrently
on the cloud host and are accuracy tests, not laptop benchmarks. Up to three
paired-pilot workers run at once; an additional UI training check overlaps part
of the experiment. The 900-second
per-phase cap prevents shared-host contention from silently changing the prescribed
epoch count. The runner rejects incomplete phases. Training timings include loss
reporting, but exclude browser startup, graph preparation, and post-run inspection.

Each pilot phase starts its epoch-based shuffle sequence at zero. This is matched
between arms. Source inspection found that the student UI instead retains its
epoch count when a dataset is replaced: `applyDatasetSelection` clears results
but does not reset `epoch`, and the multi-epoch training callback passes
`epochOffset: startEpoch` to the trainer. The additional UI run below exercises
that continuing count for one initialization.
Opening an ordinary saved project also restores its saved epoch count. Do not
instruct students to randomize their pretrained parameters to reset the counter.

An actual UI check confirmed that replacing the person dataset with the mixed
dataset preserves both every parameter and the epoch counter. It also imported
the completed seed-211 model, obtained 100% over 1,920 held-out answers in Test,
and answered five typed prompts correctly, including a matched pair with the
drop/move order reversed. Saving after inference preserved parameters. This
check did not train through UI callbacks; its evidence is `browser-ui.json`.

The additional seed-211 run imports the answer-only graph with the exact
pretrained weights at epoch 10, trains six person-only epochs using **Run epochs**,
replaces the dataset through the importer, then trains twelve mixed epochs.
The saved epoch counts are **10 → 16 → 28**, so the actual UI shuffle offsets
are 10 and 16. The dataset replacement preserves every parameter.

The final UI model answers **1,918 / 1,920** held-out questions correctly:
480/480 football locations, 480/480 carrier locations, 478/480 other-person
locations, and 480/480 holders. It answers all 120 drop-before-moving cases and
both halves of all 120 matched drop-order pairs correctly. Independent NumPy
evaluation gives loss 0.00181177, consistent with the rounded UI display of
0.002. No page errors occurred.

Those UI phases took 206.7 and 684.8 seconds while other training jobs were
running, or **14.9 minutes combined**. That is a concurrent full-UI measurement,
not the isolated engine benchmark or a validated ten-minute UI/laptop guarantee.
Evidence and the saved phase boundaries are under `browser-ui-training/`.
This single UI check verifies the workflow; it does not replace the five-seed
paired comparison.

The Vite server became unreachable when the second batch of paired runs was
launching. Four launches failed before any training began. Their diagnostics are
preserved in `infrastructure-restart.json`; they were retried against the same
clean app checkout on a new local port. Completed checkpoints were reused.
No model was retrained or selected because of its accuracy.

## Reproduction and independent checks

The preparation script writes a person-only subset, checks its counts, records
source/subset hashes, and creates fixed configurations:

```sh
.venv/bin/python scripts/pilots/story-language/prepare-football-curriculum.py

NEURALCANVAS_ROOT=/path/to/clean/app \
PLAYWRIGHT_ROOT=/path/to/playwright \
APP_URL=http://127.0.0.1:5186/NeuralCanvas/ \
.venv/bin/python -u scripts/pilots/story-language/run-football-curriculum.py \
  --seeds 211 --arms curriculum --workers 1 \
  --timing 'Isolated curriculum benchmark: no concurrent training. Two phases use a fresh Adam optimizer at the boundary.'

NEURALCANVAS_ROOT=/path/to/clean/app \
PLAYWRIGHT_ROOT=/path/to/playwright \
APP_URL=http://127.0.0.1:5186/NeuralCanvas/ \
.venv/bin/python -u scripts/pilots/story-language/run-football-curriculum.py \
  --workers 3 \
  --timing 'Concurrent native accuracy replication; not an isolated runtime benchmark.'

NEURALCANVAS_ROOT=/path/to/clean/app \
PLAYWRIGHT_ROOT=/path/to/playwright \
APP_URL=http://127.0.0.1:5186/NeuralCanvas/ \
node scripts/pilots/story-language/browser-football-curriculum.cjs

NEURALCANVAS_ROOT=/path/to/clean/app \
PLAYWRIGHT_ROOT=/path/to/playwright \
APP_URL=http://127.0.0.1:5186/NeuralCanvas/ \
node scripts/pilots/story-language/browser-football-curriculum-train.cjs

OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/pilots/story-language/audit-football-curriculum.py
```

The audit checks the subset, answers with an independent state machine, disjoint
history groups, example/update budgets, paired starting weights, exact parameter
continuity, settings, completed epochs, and parameter shapes/count. It recomputes
held-out loss and every predicted answer with the independent NumPy evaluator;
it also reports how often both answers in a drop/move-order pair are correct.
Cached app metrics are not treated as sufficient evidence. Configurations, saved
models, reports, and the audit are under `pilots/football/curriculum/`.

The completed audit passed for **all 20 paired-training phases**, both new
generative pretraining checkpoints, and the additional UI training run. It
verified every saved held-out prediction for the paired phases, independently
recomputed loss and accuracy, and confirmed the UI result within its display
rounding. Both UI checks completed without page errors.
