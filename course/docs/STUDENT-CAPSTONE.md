# Student-owned capstone: Assignment 18 pretraining → Assignment 19 QA

Numbering updated 2026-10-23 for [early sampling](SAMPLING-SEQUENCE.md); original
training evidence and pilot filenames are retained.

Updated 2026-10-21 after instructor approval. Assignments 11–18 now use the same
25-token football narratives. The separate pretraining Assignment 19 was deleted;
former Assignment 20 is now `19-qa-finetuning`. There is no Assignment 20.
Assignments 1–10 retain their tasks and rubrics.

## Current workflow

Assignment 18 trains the two-block/two-head generator and saves multihead.json.
Both blocks have independent width-8 heads, model width 16 and MLP width 32.
The position table is 32 × 16; the model has 5,689 parameters. It trains for ten
epochs on 1,534 narrative windows, with 765 validation windows and 32 next-token
targets per window. Its loss targets are 0.60 training and 0.65 validation,
identical to the QA pretraining prerequisite. Sampling is introduced in Assignment
12 using the bigram model, and its generation controls remain available throughout
the sequence. Earlier architecture
changes randomize fresh parameters; their epochs do not accumulate into the
Assignment 18 checkpoint.

In Assignment 19, save the chosen multihead.json as pretrained.json. Import
story-questions.json in a separate copy, retain every learned value, and select
Slice axis 0, start 22, end 23 before cross entropy and output Softmax. The
shorter input uses the first 23 position-table rows; no tensor is resized.
Save before.json, then train five additional epochs and save after.json.
Submit Assignment 10's football.json as earlier.json alongside those files.
There is no second narrative-pretraining run and no supplied model.

Use Tensor engine, Adam 0.003, batch 32, shuffle each epoch, clip norm 1 and no
penalty, decay or early stopping. The current student recipes report every epoch, including the five-epoch QA run. WebGL can use a supported GPU, with CPU as fallback.
The default epoch counter continues 0 → 10 → 15. Runtime and epoch counts are
not graded. Cloud software-rendering timings are not laptop GPU benchmarks.

## Optional training experiments

Students may extend narrative training in Assignment 18 by five-epoch runs,
keeping checkpoints and the same settings. Complete this before creating the
QA baseline. Assignment 19 checks its submitted pretrained.json/before.json
pair, not a previous Gradescope upload. Their learned parameters must match
exactly and the narrative model must meet both loss targets for preservation
credit. A different pretraining checkpoint requires a new baseline and QA run.

After the recommended five QA epochs, additional five-epoch runs are optional.
Keep pretrained.json and before.json fixed, compare validation accuracy and
loss, and submit a checkpoint meeting the targets. More training can improve,
plateau or worsen QA accuracy. Each Run starts fresh Adam state; several short
runs need not equal one continuous run. Continuing pretraining also changes the
starting epoch counter and therefore QA shuffle order. The fixed-size graders
evaluate the saved weights and canonical examples; extra student training does
not increase grader work. Validation used for checkpoint selection remains
development data.

The handout also suggests a controlled ten-epoch comparison. Reopen unchanged
before.json for each arm, preserving its epoch counter and learned values. Use
AdamW with weight decay 0.1, learning rate 0.003 and **ten continuous QA epochs in
one run**, reporting every epoch. Keep batch 32, shuffle, clipping 1, patience 0
and no Loss-block penalty. This is ten QA epochs total from before.json. Compare
with ten continuous Adam epochs from the same file, then optionally try AdamW
with decay 0.01. A smaller learning rate such as 0.001 is another optional
investigation; that exact ten-epoch/pretrained-start combination has not been
piloted here. These suggestions do not change the five-epoch default or rubric.

The matched native Tensor CPU pilot used three ten-epoch Assignment 18 starts
and independently evaluated every checkpoint. At ten QA epochs:

| Optimizer | Decay | Mean held-out accuracy | Mean held-out loss | Starts meeting all QA targets |
| --- | --- | --- | --- | --- |
| Adam | 0 | 73.33% | 0.6756 | 2/3 |
| AdamW | 0.01 | 82.08% | 0.5204 | 2/3 |
| AdamW | 0.1 | 82.22% | 0.4586 | 3/3 |

Only one of the three decay-0.1 starts reached 100%; the other two remained near
73%. Seven of nine runs had better accuracy at epoch five than ten. Do not
promise that AdamW or longer training always improves accuracy. This app applies
AdamW decay to all learned parameters, including biases and normalization values.
The pilot used clean app 9d535a02b53bff5859e963d07cdf2d65df6cfacf, counters 10→20
and shuffle seeds 52–61, with a fresh optimizer per run. See
[pilot evidence and loss curves](https://github.com/davbachman/NeuralCanvas-Coursework/tree/main/pilots/football/qa-adamw-10). These are
validation results on the same 240 questions, not independent final tests or
physical GPU timings. Optional settings and epochs are not graded.

## Retained QA evidence and direct handoff

The three pretraining runs already used the exact Assignment 18 construction
on this corpus. The revised Assignment 18 reference is taken directly from that
retained training, and the QA validator verifies its exact parameter match to
the baseline. Reusing this evidence does not claim new QA training.

| Seed | Narrative validation loss | QA accuracy | QA validation loss | Drop before move | Move before drop |
| --- | --- | --- | --- | --- | --- |
| 137 | 0.4857 | 73.33% | 0.6745 | 46.67% | 100.00% |
| 211 | 0.5089 | 77.08% | 0.6002 | 54.17% | 100.00% |
| 307 | 0.4834 | 78.33% | 0.4530 | 56.67% | 100.00% |

The retained real-UI run randomized, trained ten narrative epochs, imported QA
data, set the Slice, preserved parameters, trained five more epochs and saved
through File → Save. It reached **80.00%** QA accuracy, loss
**0.4967**. Its pretraining and QA times were
889.8 / 487.6
seconds in contended cloud Chromium/SwiftShader. These are historical runtime
observations, not physical GPU predictions. The harness prepared familiar graph
wiring; it did not draw every wire. Clean app commit `9d535a02b53bff5859e963d07cdf2d65df6cfacf`.
The current release separately checks imports, generation, one CPU UI epoch,
inference, saving and handout layout on the new corpus; see narrative-browser.json.

## Common task and grading

QA training has 3,840 mixed questions; evaluation has the exact same 240
football-location questions as Assignment 10, 120 per drop/move order. Whole
room-layout groups and actor/event-order variants stay in one split. Narrative
and QA histories align exactly and answers are excluded from the 23-token input.
Assignment 10's five-epoch, 6,441-parameter whole-story cross-attention models
score 58.33%, 60.42% and 68.75%. The corresponding transformer QA runs score
73.33%, 77.08% and 78.33%, a matched-seed gain of 9.58–16.67 percentage points.
The minimum gap between any retained transformer and any earlier run is
**4.58 percentage points**. There is no required improvement over a student's
own earlier model.
This compares complete course workflows, including architecture and supervision
budgets. It does not establish that pretraining is necessary or better than
training a transformer from scratch. Using one corpus simplifies continuity;
it does not inherently raise QA accuracy.

The final rubric remains 15 points for earlier.json, 25 for pretrained.json,
15 for before.json, 35 for after.json and 10 for preserved parameters. QA targets
are training/validation loss ≤ 0.75 and overall accuracy ≥ 70%. Event-order
halves are diagnostic, with no separate 80% threshold and no penalty for a strong
earlier model. Graders use semantic graph operations and independently computed
metrics, not cached scores or claimed training history.

## Reproduction and release

```sh
.venv/bin/python scripts/assignments/advanced/configure-narratives.py
.venv/bin/python scripts/assignments/football/build-student-capstone.py
RELEASE_ASSIGNMENTS=11-next-word,13-context,14-self-attention,15-residuals,16-normalization,17-stacking,18-multiple-heads,12-sampling,19-qa-finetuning .venv/bin/python scripts/assignments/package-advanced.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/advanced/validate-narratives.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/football/validate-student-capstone.py
.venv/bin/python scripts/assignments/advanced/test-narratives.py
.venv/bin/python scripts/assignments/football/write-student-report.py
.venv/bin/python scripts/assignments/advanced/audit-narratives.py
```

See each Canvas handout and Instructor/INSTRUCTOR.md for uploads. The final upload ZIP is
gradescope-19-qa-finetuning.zip. Source and reference files stay private.
Hosted Gradescope and the deployed student app are not tested by local checks.
Older pilot paths/numbers describe the experiments at the time; Git retains the
removed assignments rather than keeping alternate handouts or packages.


Assignment 10’s 6,441-parameter whole-story cross-attention with the encoded
question concatenated to the attended row is required for earlier.json. The
superseded fact-encoder architectures are rejected. Equivalent score-mask
formulas remain accepted through bounded AST parsing. The whole-story revision
changes the earlier-model input and check; narrative checkpoints, QA training,
point allocations and transformer performance targets are preserved.
