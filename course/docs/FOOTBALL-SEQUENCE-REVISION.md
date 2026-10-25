Current whole-story update: see [WHOLE-STORY-REVISION.md](WHOLE-STORY-REVISION.md).
Assignments 9/10 now use one story sequence and a separate question, five epochs,
and 37,417/6,441 parameters. The fact-row recipes below are historical.

# Football sequence: Historical Assignments 7–10 and 20

Updated 2026-10-21: the current Assignment 10 concatenation head has 8,153
parameters (about 51% fewer than Assignment 9). The 7,129-parameter/57.1% figures
below describe the earlier addition head. The current grader requires the
concatenation head. See the current Assignment 10 handout and instructor notes.

Current release (2026-10-21): Assignments 11–18 now use football narratives;
Assignment 17 supplies the pretrained checkpoint, and former Assignment 20 is
now Assignment 19 QA. The separate pretraining assignment was removed. See
[NARRATIVE-REVISION.md](NARRATIVE-REVISION.md) and
[STUDENT-CAPSTONE.md](STUDENT-CAPSTONE.md). The report below is historical;
its numbering, datasets and reproduction commands describe that earlier release.


Capstone revision: [STUDENT-CAPSTONE.md](STUDENT-CAPSTONE.md) replaces the
Assignment 19 synthesis and Assignment 20 supplied-checkpoint workflow described
below. Assignments 7–10 and 11–18 retain their current datasets and rubrics.
Students now pretrain their own football generator in 19 and fine-tune it in 20.
Earlier supplied-model evidence below remains historical.

Implemented in place after approval on 2026-10-20. Assignments 1–6 are preserved.
Assignments 11–19 retain their validated narrative datasets, models, rubrics and
generation progression. Their instructor notes and Assignment 19's closing bridge
now explain that Assignment 20 supplies a different pretrained football model.
A navigation correction in Days 11–18 locates generation controls at Dataset →
Details → Continue the text; it changes no dataset, model or rubric.
There is one current assignment per day; older exercises remain in Git history.

## Lessons and required artifacts

| Day | Student work | Automatically graded files |
| --- | --- | --- |
| 7 | Guided IMDb sentiment classifier, then an independent linear word-count classifier on football questions | reviews.json, stories.json |
| 8 | Ordered one-hot MLP, then independently integrate a learned four-dimensional embedding and wider hidden layer | onehot.json, embedding.json |
| 9 | Shared ordered fact encoder, separate question encoder and flattened-story MLP | baseline.json |
| 10 | Question-directed attention with a smaller classifier, then a fixed-weight evaluation on football questions | attention.json, football.json |
| 20 | Adapt a supplied pretrained generator to answer-only supervision; compare with the actual Day 10 model | earlier.json, before.json, after.json |

Guidance is detailed for new blocks and inputs, lighter for familiar wiring and
shape decisions. Every required comparison has a submitted artifact. Review
reversal, probability inspection and direct text generation are optional.
No quiz, prose grading, Python assignment or hosted notebook is required.

## Data and comparison

The stories use Nora, Liam, six rooms and five events. Taking transfers a football
to a person; moving carries it along; dropping leaves it behind. Passes and
arbitrary-length histories are outside the selected exercise. This is generated
course data, not an official bAbI benchmark.

Days 7–8 share exactly the same CSV: 480 training and 240 held-out football-location
questions, each 23 tokens long. Paired drop/move orders have identical word counts
but different correct answers. Whole room-layout groups and both actor variants
stay together in the split. The six room labels are bathroom, bedroom, garden,
hall, kitchen and office. The fitted counts vocabulary has 16 entries; token-ID
input adds padding and has 17.

Days 9–10 use 3,840 mixed training questions and 1,920 mixed held-out questions.
The question types cover both people's locations, football location and
possession. The structured representation has 25 vocabulary entries, nine answer
classes, ten fact slots and eight word slots. Five facts are real; masks remove
padding after word lookup and after fact bias/position addition.

Day 10's football-test.json keeps the same training documents and preprocessing,
and evaluates 240 football-location questions: 120 for each event order.
Day 20 uses exactly these training histories/questions and these 240 held-out
questions, expressed as next-token prediction. The answer is never part of the
23 input tokens. The audit also verifies that pretraining and QA use identical
training/held-out history splits; held-out QA histories were not in narrative
training. Full mixed-question accuracy and focused football accuracy
describe different distributions and must not be equated.

The held-out sets were used in curriculum selection and are validation sets,
not untouched final tests. Canonical truth, group membership and provenance are
retained in the private pilot and reference directories.

## Measured teaching advantages

Three word-count initializations scored 30.83% on football questions, with
training loss about 1.207 and held-out loss about 1.353. They all meet the rubric.
The unchanged IMDb exercise remains a successful application of word counts.

The selected Day 8 comparison gives both models ten Adam epochs:

| Model | Trainable parameters | Held-out accuracy, three initializations |
| --- | ---: | --- |
| Ordered one-hot, 8 hidden units | 3,190 | 65.83%, 61.25%, 25.83% |
| Width-4 embeddings, 32 hidden units | 3,242 | 72.92%, 74.58%, 76.25% |

The parameter counts differ by less than 2%; update budgets match. This does not
establish equal computational cost or a universal cross-entropy improvement.
An eight-dimensional embedding candidate failed to give a consistent advantage
and is retained as rejected pilot evidence. The published rubric requires
absolute targets for both models, never deliberately poor baseline performance.

The original 30-epoch, batch-32 Day 9 pilots scored 83.33–89.58% on focused
football tracking. That MLP has **16,601 parameters** and can outperform the
**7,129-parameter** single-attention model, whose three matched mixed-question
training runs scored 55–58.33% on the focused evaluation. Its mixed-question
accuracy was 94.11–94.74%; all person-location and possession questions were
correct. Moving before dropping scored 100%, but dropping before moving scored
10–16.67%. Report that limitation instead of promising an accuracy improvement
from Day 9 to Day 10. Attention's demonstrated advantage is a 57.1% parameter
reduction and compact question-dependent fact selection.

Day 20's **5,689-parameter** transformer successfully adapts to the difficult
football task. The current released-start UI replication scored 240/240,
including 120/120 on both halves, in **454.851 seconds on Tensor CPU**. An earlier
replication scored identically in 447.009 seconds, and the separate mixed
diagnostic scored 1,920/1,920. The latter and the earlier runtime are retained
selection evidence, not a second current student exercise.

This is a practical course comparison. The transformer has extra narrative
pretraining; Day 10 does not. The larger Day 9 MLP also solves much of this task.
The evidence does not establish that only transformers can do tracking, that
pretraining is essential, or that fine-tuning beats a matched scratch model.

## Controlled pretrained start

The supplied Canvas/pretrained-football.json preserves all learned tensors from
native-short-diverse/137-pretrain/model.json, trained for ten narrative epochs.
Its new-task epoch counter is zero, and its reset snapshot contains the pretrained
values. Labels and selected-block/run-control metadata are adjusted for students.
The 32 × 16 position table remains intact; QA uses the first 23 rows. Students
add a final-row Slice, preserving the weights and output Softmax branch.

The app's epoch counter affects shuffle order. Counter-10 controls reached only
77–79%, and arbitrary independently pretrained starts were not consistently
successful. The current recommendation is therefore for the supplied checkpoint
and fixed recipe: Adam .003, batch 32, ten epochs in one run, shuffle, report
every five, clip norm 1, no early stopping, no decay. Reimport before.json for a
fresh attempt. Do not replace it with randomization or an arbitrary prior model.

The supplied model has 25 vocabulary entries, whereas Days 11–19 have 21. Students
load it as a model rather than resize their own narrative tables. This avoids
disrupting the validated generative sequence or spending a class on migration.
The instructor explicitly authorized a supplied pretrained start for this lesson.

A larger-batch CPU screen (batch 128, rate .01) did not complete the requested
30 epochs within its five-minute Node budget. These aborted trials were not used
to set thresholds or replace the validated batch-32 recipe. Their snapshot
metrics are not claimed as final-epoch scores. Day 9 CPU runs can take about ten
minutes on the cloud host. Full 30-epoch WebGL checks of Days 9 and 10 then
completed in 290.736 and 306.121 seconds, with training losses .004190 and .116185.
The Day 9 football accuracy was 85.83%; Day 10 mixed-question accuracy was 94.11%.
These concurrent software-rendered browser runs support allowing Auto or WebGL
for both assignments, but do not guarantee five minutes on student hardware.

The ten-minute budget may use GPU training. CPU is a demonstrated viable option.
A full WebGL UI replication also scored 240/240 in 442.192 seconds, with
independent held-out loss .001504016. It used SwiftShader software rendering
with concurrent CPU screening; this verifies convergence, not physical-GPU
speed. Assignment 20 allows Auto or explicit WebGL, with Tensor CPU as the
fallback. WebGPU remains unvalidated. Training time is not graded.

## Grading and release validation

The bounded NumPy evaluator executes submitted operations against canonical
inputs and saved parameters. It does not execute student code or trust cached
scores, epoch counters or labels. Equivalent supported constructions remain
acceptable. Day 10 verifies unchanged semantic parameter fingerprints across its
two files. Day 20 checks before.json against the actual supplied checkpoint,
not just against another student file. Its accuracy credit requires at least
80% in **each** event-order half; high accuracy on the easy half cannot hide the
tracking failure.

The rubric never penalizes an unusually good earlier model. Required comparison
files are graded for their structure/data and appropriate independently computed
targets. No score relies solely on a student's claimed improvement.

Evidence:

- `docs/football-release-validation.json`: independent full-budget/reference
  scores, native parity, subgroup results and current capstone UI saves.
- Day 7/8 `reference/football-pilot/`: selected three-seed trials, canonical data
  and provenance; `reference/browser/`: CSV import, UI training, inference and Save.
- Day 9/10 `reference/football-pilot/`: structured-model trials;
  `reference/browser-football/`: prepared-data replacement, inference and Save.
- Day 20 `reference/browser/workflow.json`: current public-checkpoint replication,
  actual Slice fields and prepared importer, ten training epochs, inference and
  saved before/after models. Familiar Slice wiring was prepared by the harness;
  it does not claim every wire was drawn manually.
- Day 20 `reference/browser/generation.json`: actual Dataset Details controls
  correctly generate football-location, person-location and possession answers
  for the handout story.
- Each changed assignment's `reference/test-log.txt` and
  `reference/packaged-test-results.json`: grader tests and extracted upload ZIP.

All new app checks use clean commit
`2b6b96d8d07ee52e0beca3c3432f9f4e5819af79`. These are local browser checks.
They do not establish hosted Gradescope behavior or the public app deployment.
The capstone CPU timing had no concurrent training or numerical grading; routine
documentation work could overlap. Early-model timings and old native runs were
partly concurrent and are not promises for student laptops.

## Reproduction

Use committed fixtures to package and test without retraining:

```sh
.venv/bin/python scripts/assignments/football/install-early.py
.venv/bin/python scripts/assignments/football/build-later.py --days 9,10,20
.venv/bin/python scripts/assignments/football/install-backends.py
.venv/bin/python scripts/assignments/football/audit-release.py
.venv/bin/python scripts/assignments/package-word-counts.py
.venv/bin/python scripts/assignments/package-embeddings.py
RELEASE_ASSIGNMENTS=09-story-models,10-attention,19-synthesis,20-qa-finetuning \
  .venv/bin/python scripts/assignments/package-advanced.py
```

Run each affected assignment's unittest suite and then
`scripts/assignments/advanced/audit-narratives.py` for package/layout consistency.
The football browser scripts require PLAYWRIGHT_ROOT, NEURALCANVAS_ROOT and the
local app URL; browser-capstone.cjs accepts PILOT_BACKEND=webgl for a separate
software-WebGL run without overwriting CPU evidence. Pilots overwrite their
evidence files, so rerun independent scoring after training.

The older bAbI/text and two-fact capstone writers are retired or redirected to
the current builder. Historical pilot records remain private and identified
by their original data/app metadata. Only each assignment's Canvas directory
contains student materials. GitHub push does not upload to Canvas or Gradescope.
