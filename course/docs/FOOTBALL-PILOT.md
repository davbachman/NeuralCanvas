# Football narratives and mixed-question fine-tuning

2026-10-20. Private feasibility work; Assignments 11–20 and their upload packages
remain unchanged.

The subsequent [two-person no-pass replication](FOOTBALL-REPLICATION.md) tests the
instructor's newly accepted 80%-per-question-category pilot target. The earlier
85% release rubric is not changed by either pilot.

## Finding

The requested task is representable by the existing app operations. A prototype
can answer where either person is, where the football is, and who holds it,
including transfers and a ball left behind after a drop. One screened checkpoint
answers all 512 held-out questions correctly in both independent evaluation and
native Neural Canvas inference.

**This is not yet a dependable replacement assignment.** In the same constrained
task, three independently pretrained/fine-tuned seeds reached 75.0%, 100%, and
74.6%. The weaker runs learned ball location and possession but confused the
people. Less constrained event orders and a larger cast were substantially harder.
Increasing width, adding a third block, changing question position, and a higher
fine-tuning rate did not establish a reliable solution.

The successful checkpoint was trained in **PyTorch screening**, then exported as
ordinary editable Neural Canvas blocks. Native import, all held-out predictions,
12 direct prompts, and an independent NumPy evaluator agree. This does not claim
that the full successful training trajectory was reproduced in the app. A separate
native QA training epoch verifies the operation path; the full native training
benchmark was on the earlier movement/drop corpus.

Keep the current releases while improving this prototype. These screens did not
change the current 85% QA grading requirement.
Do not select only the successful seed and promise that students will reproduce it.

## Histories and questions

The transfer corpus uses eight events:

1. Two people enter the same room.
2. One picks up the football.
3. That person either passes it to the other person or keeps it.
4. The current carrier moves twice; the other person also moves.
5. The carrier drops the ball before or after the final move, or keeps holding it.

Every pass is checked for co-location and ownership. Each state is tracked by a
generator and checked independently using the event history. The initial pickup
introduces the football at the picker's location; no prior contradictory ball
location is supplied.

There are four question examples per history: ball location, each of the two
people's locations, and current holder. “Nobody” is correct for a dropped ball.
Half of the transfer variants retain a holder; half end on the ground. Both pass
and no-pass variants are included. A model cannot get full possession credit by
always answering “nobody.”

Two drop-order variants have the same event words but different football
locations. A word-count predictor cannot exceed 50% on that paired subset.
The same-room initial facts, two subsequent carrier moves, and the other person's
move keep the task beyond simply choosing the last room token. The current
carrier's final room is correct for 75% of ball questions, so the critical
drop-before-final-move subset is reported separately.

The constrained transfer corpus uses Mary and John and four rooms. There are
32 training groups and 16 validation groups, each defined by the ordered people
and all four room assignments. All eight transfer/drop variants and all four
questions stay within that group. This yields 256 training / 128 held-out
histories and 1,024 training / 512 held-out questions. The larger corpus uses four
people, 128 training / 64 held-out groups, and 4,096 / 2,048 questions.

The constrained grammar fixes the order of the initial facts and the distractor
move. Drop order, possession, names, and room assignments still vary. Its perfect
checkpoint is evidence for this narrow grammar, not for arbitrary event order,
longer stories, repeated passes, unfamiliar names, or general language reasoning.
The broader corpus varies the initial order and distractor placement.

The generative data contains only narratives and concluding statements, never
questions. Its vocabulary comes from training narratives plus declared reserved
`where` and `who`. Narratives sometimes conclude with a person's location, the
football's location, or possession. This introduces `has` and `nobody` naturally.
Generative inputs are 64-token stream windows, stride 17. QA inputs contain the
48 event tokens plus a four-token question and one separate answer target.

This pilot standardizes question length to the app's current language-window
format: `where is mary now`, `where is the football`, and `who has the football`,
without a question mark. A separate screen tested `now where is mary`. Shorter
or arbitrary wording has not been validated. The demo should be used with its
documented question format.

## Screening results

These results are PyTorch feasibility screens, not browser timings or new grading
thresholds. The standard model has width 16, two blocks, two width-8 heads, and
6,300 parameters on the 28-token constrained corpus. Narrative pretraining uses
Adam 0.003, batch 32, clip norm 1, ten epochs with all-position loss. Fine-tuning
uses answer-only loss and the same optimizer/rate/batch/clip settings, with a fresh
optimizer. All parameters remain trainable.

### Constrained transfer grammar: three independent seeds

After 60 fine-tuning epochs:

| Seed | Football location | Carrier location | Other person location | Who holds it | All questions |
|---|---:|---:|---:|---:|---:|
| 137 | 100% | 37.5% | 62.5% | 100% | 75.0% |
| 211 | 100% | 100% | 100% | 100% | 100% |
| 307 | 98.4% | 68.8% | 31.2% | 100% | 74.6% |

“Carrier location” identifies the person who carries or carried the football;
the actual question contains that person's name. It is not an extra role label
given to the model. Every run is evaluated on the same held-out histories.
Seed 211 reached 99.2% overall by epoch 40; the other seeds did not show the same
transition. Full curves and checkpoints, including scratch controls, are retained.
Scratch controls also learned some versions well, so the results do not establish
an advantage for pretraining.

### Harder variants and controls

| Screen | Football location | Person locations | Holder | Overall |
|---|---:|---:|---:|---:|
| Pickup/drop, two people, no passes; 80 epochs | 90.2% | 85.2–88.3% | 100% | 90.9% |
| Variable-order transfers, two people; 100 epochs | 50.0% | 25.0–47.7% | 100% | 55.7% |
| Variable-order transfers, four people; 40 epochs | 61.1% | 44.9–64.6% | 99.2% | 67.5% |
| Four people, width 32; 80 epochs | 56.4% | 35.7–61.5% | 100% | 63.4% |
| Four people, three blocks, rate 0.006; 80 epochs | 64.1% | 49.0–49.2% | 99.6% | 65.5% |
| Four people, question ends in person's name; 80 epochs | 61.9% | 51.4–68.8% | 100% | 70.5% |

These are single-seed explorations, not controlled estimates of an architectural
effect. The narrower no-pass variant is promising but is not independently
replicated or verified with a full native fine-tuning run. On the constrained
transfer task, raising the fine-tuning rate to 0.01 gave only 53.5–71.7% overall
across three seeds at 40 epochs.

Eight generative model families were screened on both the initial movement/drop
corpus and the constrained transfer corpus: bigram, full-context MLP, attention,
residual/MLP, normalized single-block, all-position single-block, two blocks, and
two heads per block. The generated summary in `pilots/football/summary.json`
reports overall token accuracy separately from predicting the football's location
after a fully supplied story. High token accuracy alone does not establish
successful state tracking. No new handout or grader threshold follows from these
screens.

For the constrained transfer corpus, seed 137 gave:

| Generative model | Objective | Epochs | Parameters | Overall held-out tokens | Football continuation |
|---|---|---:|---:|---:|---:|
| Bigram | Final position | 5 | 924 | 66.4% | 0% |
| Context MLP | Final position | 5 | 67,868 | 81.6% | 20.3% |
| Attention only | Final position | 5 | 2,716 | 27.2% | 0% |
| Residual and MLP | Final position | 5 | 4,044 | 64.7% | 0% |
| Normalized block | Final position | 5 | 4,140 | 74.5% | 0% |
| Normalized block | All positions | 10 | 4,140 | 76.7% | 31.2% |
| Two blocks | All positions | 10 | 6,300 | 78.3% | 15.6% |
| Two heads per block | All positions | 10 | 6,300 | 78.4% | 23.4% |

Final-position and all-position token accuracies score different targets. The
football column is a separate fully supplied story followed by “now the football
is in the”; no correct location is supplied to the model. Shorter prompts are
left-padded for the fixed-context MLP and use their actual length for attention
models. Thus it is a practical continuation probe, not an isolated comparison of
architecture or training efficiency. The weak state results are another reason
not to transplant this corpus into all ten assignments yet.

## Native checks and runtime

Native checks use clean app
`2b6b96d8d07ee52e0beca3c3432f9f4e5819af79`, Chromium 151, Tensor CPU, on a cloud
machine with a four-core quota. The app source was not edited.

- A full ten-epoch native generative run on the initial six-move, pickup/drop
  corpus took **818.5 seconds (13.6 minutes)** with reporting every epoch. It
  reached 77.5% overall token accuracy but only **23.3%** on fully observed
  narrative consequence targets. Some fast screening jobs ran concurrently;
  this is a contended measurement, not a student-laptop guarantee.
- A separate native QA epoch on the constrained transfer corpus took **45.6
  seconds**, including initial/final evaluation, for 1,024 training and 512
  validation questions. It began from a PyTorch-pretrained checkpoint and
  reached 38.3% held-out accuracy. This establishes training compatibility,
  not full-budget convergence or a linear runtime prediction.
- The successful screened checkpoint gives **512/512** correct native held-out
  predictions and **12/12** correct direct prompts. Independent NumPy predictions
  match all 512, with close agreement on sampled logits.

No physical GPU benchmark or full-UI training run was performed. These checks do
not yet meet the course's requirement for reproducible full-budget student
training within roughly five minutes per ordinary model or ten minutes late in
the sequence. Fewer reports may reduce overhead; that alone has not been tested
as a solution.

## Demonstration and next decision

Import `pilots/football/demo/model.json` into Neural Canvas and run held-out
inference. Its small bundled dataset uses the constrained transfer grammar.
`pilots/football/demo/README.md` contains verified prompts and the checkpoint's
provenance. Keep it private: it is a selected demonstration, not a starter model
or student-facing reference.

The useful next pilot is to find a shorter, sufficiently varied history grammar
and a training schedule that reliably separates the requested person's state
from the football's state. Retain counterfactual drop-order checks and all
question types. Success should require strong results per question type across
several independent native runs, plus an isolated runtime check. The current
evidence does not justify rewriting Assignments 11–20 yet.

## Files and reproduction

All work is under `pilots/football/` and `scripts/pilots/story-language/`;
`assignments/` is unchanged. Per-screen `config.json` records the exact data,
seed(s), dimensions, update budgets, source checkpoint and settings. Failed
comparisons are retained alongside successes. Validation data was used for
selection; it is not an untouched final test set.

```sh
.venv/bin/python scripts/pilots/story-language/prepare-football.py
.venv/bin/python scripts/pilots/story-language/prepare-transfer-football.py \
  --fixed-order --out pilots/football/transfers-ordered
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/pilots/story-language/screen-football.py \
  --data pilots/football/transfers-ordered \
  --out pilots/football/screen-transfers-ordered \
  --epochs 10 --qa-epochs 60 --seeds 137,211,307 --scratch
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/pilots/story-language/validate-football.py
```

The generators accept the recorded cast size, group count, compact-wording,
order and question-position flags; replay the corresponding configs/commands for
other exploratory datasets. Browser scripts use `NEURALCANVAS_ROOT`,
`PLAYWRIGHT_ROOT`, and `APP_URL` as in the earlier story pilots.
`export-football.mjs` converts a selected screen checkpoint to native blocks;
`check-football-browser.cjs` checks saved-model parsing, held-out predictions and
direct prompting. These operations do not upload anything to Canvas or Gradescope.
