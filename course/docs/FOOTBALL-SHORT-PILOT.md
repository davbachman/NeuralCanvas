# Shorter football stories

2026-10-20. Private follow-up to `FOOTBALL-REPLICATION.md`.

## Decision

**Shortening the histories makes the task feasible within the runtime budget,
but the tested recipe is still not reliable enough to release.**
The strongest candidate uses five events, two people, six rooms, and compact
three-token questions. All five PyTorch initializations passed the 80% target
for every question category and for drop-before-moving examples at 15 QA epochs.
Of three native initializations, two passed; the third scored only **39.2%**
on the other person's location, despite **81.7% overall**.

The isolated native checkpoint reached fifteen QA epochs in 8.1 minutes, after
3.4 minutes of narrative pretraining. These are separate training stages and
cloud Tensor CPU measurements, not student-laptop promises. The two successful
native models show that the architecture can solve this task; the failed run
prevents recommending the recipe as a dependable student workflow.

Assignment datasets, handouts, graders, and upload packages remain unchanged.
Before replacing them, the native initialization sensitivity needs to be
resolved and the earlier generative progression needs its own validation.
These experiments do not validate passes, longer variable histories, or
arbitrary question wording.

## Task and data

The five-event task places two people in rooms, has one pick up the football,
and then has that person drop or hold the ball before or after moving. Dropping
before moving leaves the ball in the first room. Dropping afterward, or continuing
to hold it, leaves it in the person's final room. Both people can be the carrier.

For example:

> nora entered kitchen . liam entered bedroom . nora took football .
> nora dropped football . nora entered office .

The answers are kitchen for the football, office for Nora, bedroom for Liam, and
nobody for possession. Swapping the final two events changes the football answer
to office without changing the bag of words. Held variants ensure that always
answering “nobody” fails half of the possession questions.

The model receives no role labels or answers in its prompt. Reports separate
carrier and other-person location, but the actual questions contain their names.
The three-token prompts are `where is nora`, `where is liam`,
`where is football`, and `who has football`. This deliberately restricted grammar
omits the article before football and the question mark. Arbitrary paraphrases,
passes, and longer stories are not covered by these results.

There are four rooms and two people. The split is by **room layout**, keeping both
actor assignments and all four drop/hold order variants together. Of 48 possible
ordered room layouts, 32 are training and 16 validation. This produces 256
training / 128 validation histories and 1,024 / 512 question examples. The
validation layouts are used for pilot selection, not as an untouched final test.
The 512 questions are not 512 independent story layouts.

Narrative pretraining contains stories and declarative consequences, never
questions. Its 23-token vocabulary reserves `where` and `who`. Context length is
32; the compact question prompts are 23 tokens. The unchanged two-block,
two-head, width-16 architecture has **5,623 parameters**. Dataset adaptation
retains the original 32-row position table.

The six-event comparison adds a move by the other person, interleaved with the
carrier's actions. Its question inputs are 27 tokens with compact questions.
The original four-token question controls use `where is nora now`,
`where is the football`, and `who has the football`, requiring one extra token.

## Screening comparisons

The default recipe uses Adam 0.003, batch 32, clip norm 1, no weight decay or
early stopping, ten narrative-pretraining epochs, and a fresh optimizer for
question fine-tuning. Seeds are 137, 211, 307, 419, and 503 throughout.

All screened configurations and failures are retained under
`pilots/football/screen-short-*`. The audit summarizes them in
`pilots/football/short-summary.json`. Screening uses PyTorch, not browser training.

| Configuration | QA epochs | Seeds clearing 80% in every category |
|---|---:|---:|
| Five events, original four-token questions | 40 | 1 / 5 |
| Same, longer training | 80 | 2 / 5 |
| Five events, compact three-token questions | 40 | 4 / 5 |
| Same, longer training | 80 | 4 / 5 |
| Six events, original questions | 40 | 0 / 5 |
| Six events, compact questions | 80 | 2 / 5 |
| Five events, compact questions, rate 0.006 | 80 | 2 / 5 |
| Five events, compact questions, width 32 | 40 | 3 / 5 |
| Five events, compact questions, both initial orders | 40 | 2 / 5 |
| Five events, compact questions, 30 pretraining epochs | 40 | 1 / 5 |
| Five events, compact questions, three blocks | 40 | 1 / 5 |
| Five events, compact questions, six rooms / more training layouts | 10 | 4 / 5 |
| Same six-room task, shortened budget replication | 15 | 5 / 5 |
| Same six-room task | 20 | 5 / 5 |
| Six rooms, batch 16 instead of 32 | 10 | 4 / 5 |

The question-wording comparison uses identical narratives, underlying event
histories, split membership, and actual pretrained weights. It supports the
wording change on this task. It does not isolate whether the benefit comes from
the named entity being last, the removed filler token, or the shorter prompt.
Other comparisons change data size, model size, training rate, or update count
and are explicitly exploratory.

The balanced-order variant doubles the stories by including both opening orders
for each layout and actor assignment. It is retained even though it did not
improve reliability. The width-32 variant has 19,415 parameters and likewise did
not eliminate failed initializations.
Longer pretraining and a third block (7,783 parameters) also failed to make the
four-room task reliable.

At 40 epochs, the five-event compact-question screen gives:

| Seed | Football | Carrier location | Other person | Holder | Overall |
|---|---:|---:|---:|---:|---:|
| 137 | 96.9% | 100% | 100% | 100% | 99.2% |
| 211 | 100% | 100% | 100% | 100% | 100% |
| 307 | 100% | 100% | 100% | 100% | 100% |
| 419 | 84.4% | 100% | 29.7% | 100% | 78.5% |
| 503 | 100% | 100% | 100% | 100% | 100% |

Seed 419's failure is not discarded: its other-person accuracy remains 37.5%
even after 80 epochs. At 80 epochs its overall score is 80.9%, illustrating why
an overall-only 80% threshold is insufficient. The fixed 40-epoch recipe, rather
than a different best epoch per seed, is used for the native test.

### More training diversity

The final screen keeps five events, two people, and the compact questions, but
uses six rooms. Of 180 possible room layouts, 120 are training and 60 validation.
That yields 960 / 480 histories and 3,840 / 1,920 QA examples. The vocabulary
grows from 23 to 25, and the model from 5,623 to **5,689 parameters**.
Both actor assignments and all four variants still stay in the same split.

At 20 QA epochs, all five pretrained runs score **100% in every question
category**, including the subset where the football is dropped before moving.
At ten epochs, four of five clear the category threshold, but only two clear
80% on the dropped-before-moving subset. Training from scratch for 20 epochs
passes every category in four of five seeds, and that difficult subset in
three of five. This is evidence for this recipe, not a general proof that
pretraining always improves question answering.

The comparison changes both diversity and training volume. Ten QA epochs here
process 38,400 examples, close to 40,960 for 40 epochs of the four-room task.
Twenty epochs double that budget. Narrative pretraining also has more examples.
Consequently, the 20-epoch result cannot be attributed solely to diversity.

After the native runtime check, the same five pretrained checkpoints were
fine-tuned afresh for **15 epochs**, preserving the optimizer, shuffle seed, and
other settings. The repeated runs have identical epoch-ten metrics to the
20-epoch runs. At epoch 15:

| Seed | Football | Carrier location | Other person | Holder | Drop before moving |
|---|---:|---:|---:|---:|---:|
| 137 | 94.4% | 99.8% | 100% | 100% | 80% |
| 211 | 100% | 100% | 100% | 100% | 100% |
| 307 | 100% | 100% | 100% | 100% | 100% |
| 419 | 96.3% | 100% | 100% | 100% | 85% |
| 503 | 100% | 100% | 100% | 100% | 100% |

All five clear 80% in every category and on the difficult subset. This budget
processes 57,600 QA examples. The seed-137 subset score is exactly 80%, so the
results do not justify setting a student threshold higher than that.

A final targeted screen reduced QA batches from 32 to 16 while using the same
pretrained checkpoints and ten epochs. This increases parameter updates from
1,200 to 2,400 at the same number of processed QA examples. It still failed:
seed 307 scored 33.8% on the other person's location and 71.5% on the carrier.
Only two of five seeds cleared 80% on the drop-before-moving subset. Since this
did not resolve the screening failures, it was not advanced to native runs.

## Native checks

Native runs use the app's initializer, optimizer, batching, and editable graph
operations. The app checkout is clean commit
`2b6b96d8d07ee52e0beca3c3432f9f4e5819af79`; the browser is Chromium 151,
backend Tensor CPU, on an AMD EPYC 9V74 cloud host with a four-core quota.
These are isolated browser documents running native engine modules, not complete
UI callback tests or physical-GPU/laptop benchmarks.

Raw configurations, saved models, timing, and per-question predictions are under
`pilots/football/native-short-five/`. The independent NumPy evaluator checks all
held-out predictions and losses. Dataset replacement must preserve every starting
parameter, including the position table. The native run targets ten pretraining
and 40 fine-tuning epochs, with reporting every five epochs and a 600-second cap
per phase.

The four-room native results at the fixed final epoch are:

| Seed | Football | Carrier location | Other person | Holder | Overall |
|---|---:|---:|---:|---:|---:|
| 137 | 79.7% | 100% | 59.4% | 100% | 84.8% |
| 211 | 100% | 100% | 100% | 100% | 100% |
| 307 | 89.1% | 100% | 84.4% | 100% | 93.4% |

On the harder drop-before-moving subset, these runs score 21.9%, 100%, and
56.3%, respectively. Thus only one of the three clears 80% on that subset.
It is reported separately because an 80% aggregate football score can still
hide reliance on the carrier's final room.

Seed 137 was run alone: 54.7 seconds pretraining and 348.2 seconds fine-tuning,
6.7 minutes total. Seeds 211 and 307 were concurrent with each other and some
screening work: their pretraining/fine-tuning times were 68.3/424.3 and
69.3/417.3 seconds. Those latter timings are not isolated benchmarks.
All completed the prescribed epochs without reaching the time limit.
Two of three native initializations clear 80% in every category; the failed
initialization remains in the evidence.

### Six-room native candidate

The candidate uses the same width-16, two-block, two-head construction,
Adam 0.003, batch 32, clip norm 1, ten narrative-pretraining epochs, and
**fifteen question fine-tuning epochs**. It has 5,689 parameters. The vocabulary
and every parameter are preserved when adapting the dataset; the position table
stays 32 × 16 while question prompts use 23 positions.

Seed 137's pretraining took 201.3 seconds. Its epoch-15 fine-tuning checkpoint
was reached in **487.3 seconds**, with 100% accuracy on all 1,920 held-out answers,
including all 120 drop-before-moving questions. Independent evaluation of the
saved model matches the native accuracy and loss.

The initial timing run requested 20 epochs. It reached its 600-second cap after
18 complete epochs and saved the last fully evaluated weights, at epoch 15.
The epoch-15 timing was measured before other training began; only the
subsequent unused updates overlapped other jobs. The report retains both the
600.2-second full run and the 487.3-second saved-checkpoint time. It does not
claim that 20 epochs finished within ten minutes.

Seeds 211 and 307 test the fixed 15-epoch recipe from fresh native initializations.
They run concurrently for accuracy replication. Their fine-tuning caps are
900 seconds to avoid making shared-host contention decide which epoch is saved;
these runs are not isolated ten-minute benchmarks.

| Seed | Football | Carrier location | Other person | Holder | Overall | Drop before moving |
|---|---:|---:|---:|---:|---:|---:|
| 137 | 100% | 100% | 100% | 100% | 100% | 100% |
| 211 | 87.5% | 100% | 39.2% | 100% | 81.7% | 50% |
| 307 | 99.6% | 100% | 100% | 100% | 99.9% | 98.3% |

Seeds 211 and 307 completed all 15 epochs in 572.2 and 585.2 seconds under
concurrent load. Their pretraining took 246.1 and 246.6 seconds. Seed 211 remains
a substantial failure despite exceeding 80% overall. Therefore, the five-of-five
PyTorch result does **not** establish a reliable native assignment recipe.
Differences in initialization and shuffle implementation prevent treating this
as an isolated numerical-backend comparison.

The next [curriculum-learning experiment](FOOTBALL-CURRICULUM-PILOT.md) reuses
these checkpoints and tests a person-location warm-up against a control with the
same number of updates and an identical optimizer-restart boundary.

## Reproduction

```sh
.venv/bin/python scripts/pilots/story-language/prepare-short-football.py \
  --out pilots/football/short-five-diverse --direct-questions \
  --rooms 6 --train-groups 120 --test-groups 60

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/pilots/story-language/screen-football.py \
  --data pilots/football/short-five-diverse \
  --out pilots/football/screen-short-five-diverse \
  --epochs 10 --qa-epochs 20 --seeds 137,211,307,419,503 --scratch

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/pilots/story-language/screen-football.py \
  --data pilots/football/short-five-diverse \
  --out pilots/football/screen-short-five-diverse15 \
  --epochs 0 --qa-epochs 15 --seeds 137,211,307,419,503 \
  --source 'pilots/football/screen-short-five-diverse/heads-{seed}-pretrain.json'

NEURALCANVAS_ROOT=/path/to/clean/app PLAYWRIGHT_ROOT=/path/to/playwright \
APP_URL=http://127.0.0.1:5186/NeuralCanvas/ \
node scripts/pilots/story-language/browser-pilot.mjs \
  pilots/football/native-short-diverse/211-pretrain.json \
  pilots/football/native-short-diverse/211-pretrain

NEURALCANVAS_ROOT=/path/to/clean/app PLAYWRIGHT_ROOT=/path/to/playwright \
APP_URL=http://127.0.0.1:5186/NeuralCanvas/ \
node scripts/pilots/story-language/browser-pilot.mjs \
  pilots/football/native-short-diverse/211-finetune.json \
  pilots/football/native-short-diverse/211-finetune

OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/pilots/story-language/audit-short-football.py
```

Each fixture's provenance records generator options and hashes; each screening
folder has its full configuration. Reproduce the other controls using those
options, rather than overwriting an assignment dataset.

All six fixture variants were regenerated in temporary directories with the
recorded options. Every dataset and truth-file hash matched the committed
candidate files exactly. The audit also checks rendered event text against the
independent state-machine inputs, layout separation, matched order-sensitive
pairs, target alignment, and native prediction agreement.
