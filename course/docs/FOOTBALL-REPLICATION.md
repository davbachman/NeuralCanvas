# Two-person pickup/carry/drop replication

2026-10-20. Private pilot; assignment datasets, handouts, and graders are unchanged.
This follows the instructor's approval of a target of 80% held-out accuracy in
each question category. Passes are deferred for this experiment.

Follow-up: [shorter histories and more varied room layouts](FOOTBALL-SHORT-PILOT.md).
The results below describe the original longer-history recipe.

## Decision

The simpler task is promising, but **this fixed recipe is not ready to release**.
Four of five PyTorch runs cleared 80% in every question category. One failed
substantially. The separate native CPU run reached the ten-minute fine-tuning
limit; its last evaluated checkpoint scored 63.0%, with location questions near
50%. Keep the current assignment releases while pursuing a shorter, more
reliable recipe. The evidence does not justify an 80% student grading requirement
for this variant yet.

## Fixed-recipe screening

The committed `pilots/football/mixed-small` fixture is unchanged: Nora and Liam,
four rooms, five movements plus pickup and drop/hold events. There are 512
training / 256 validation histories, with four question examples per history
(football location, each person's location, and holder). All four variants of a
base history remain together in one split. There are 2,048 training / 1,024
validation questions. Narrative pretraining contains no questions.

The same model and recipe were used for all five seeds: width 16, two blocks,
two heads per block, 5,879 parameters, ten narrative-pretraining epochs followed
by 80 answer-only fine-tuning epochs. Both phases use Adam, learning rate 0.003,
batch 32, gradient norm limit 1, and no early stopping or weight decay. The
optimizer is reset at the phase transition. The fixed validation histories are
used for pilot selection; they are not a new untouched test set.

| Seed | Football | Carrier's location | Other person's location | Holder | Overall | Every category ≥80% |
|---|---:|---:|---:|---:|---:|---|
| 137 | 90.2% | 88.3% | 85.2% | 100% | 90.9% | Yes |
| 211 | 100% | 100% | 100% | 100% | 100% | Yes |
| 307 | 100% | 100% | 91.4% | 100% | 97.9% | Yes |
| 419 | 62.9% | 66.8% | 55.1% | 100% | 71.2% | No |
| 503 | 98.8% | 98.4% | 93.8% | 100% | 97.8% | Yes |

“Carrier” is a reporting category, not an input hint: the actual question names
the person. Both people are equally eligible to carry the ball. The three
conceptual question types are further separated here so one person's role
cannot hide failures on the other.

The original seed 137 was reproduced exactly, including all learned weights.
Four of five seeds pass the proposed per-category target; the failed run is
retained. Its training accuracy reaches 98.0%, but validation remains near 71%,
so more epochs alone are not an established remedy. This is a promising task,
not yet a recipe that every student can depend on.

Matched scratch controls received the same 80 question-training epochs and
achieved 73.9%, 99.3%, 74.0%, 78.7%, and 81.5% overall. Only seed 211 cleared
80% in every category. Pretraining helped four of five paired runs in this
screen, but adds ten epochs and narrative exposure; this is not an equal-total-
computation comparison or a universal claim about pretraining.

All these results are PyTorch screens. Browser timing and native training are
assessed separately below.

## Native validation

The native experiment uses clean app
`2b6b96d8d07ee52e0beca3c3432f9f4e5819af79`, Chromium, Tensor CPU, on an
AMD EPYC 9V74 cloud host with a four-core CPU quota. Other training jobs are
stopped during timing. It loads the app's own training modules in an isolated
browser document. It does not exercise the complete user interface or measure
a physical GPU or student laptop.

The initial parameters come from Neural Canvas's initializer. Narrative training
and question fine-tuning use the app's optimizer and batching. Dataset adaptation
preserves every parameter, including the original 48-row position table, and
adds the answer-position Slice for the 39-token question inputs. No PyTorch-
trained weights are used in this native run.

Each phase has a 600-second training budget checked between updates. Reporting is
every five epochs. A stopped run retains the **last fully evaluated checkpoint**,
identifies its actual epoch, and records any later completed epochs separately;
it does not silently label that checkpoint as an 80-epoch model. Training time
includes initial and scheduled train/validation evaluation, but excludes script
startup, post-training independent inference, and file serialization.

Native results are recorded in `pilots/football/native-mixed-small/` and audited
in `pilots/football/replication-summary.json`.

The native seed-137 experiment completed all ten narrative-pretraining epochs
in **313.0 seconds (5.2 minutes)**. Held-out token accuracy was 76.1%, but only
21.1% of 142 fully observed narrative consequence targets were correct. These are
different metrics; overall token accuracy is not evidence of reliable state
tracking.

Fine-tuning stopped at **600.3 seconds**, after 17 complete epochs and part of the
next. The latest fully evaluated snapshot is epoch 15, recorded at 523.3 seconds.
Epochs 16–17 and the partial epoch are not represented in the saved model.

| Evaluated epoch | Training accuracy | Held-out accuracy | Elapsed fine-tuning time |
|---|---:|---:|---:|
| 0 | 0% | 0% | 13.9 s |
| 5 | 69.5% | 65.4% | 183.4 s |
| 10 | 76.2% | 64.3% | 354.1 s |
| 15 | 82.3% | 63.0% | 523.3 s |

Independent evaluation of the saved epoch-15 model gives:

| Question category | Correct / total | Accuracy |
|---|---:|---:|
| Football location | 124 / 256 | 48.4% |
| Carrier's location | 128 / 256 | 50.0% |
| Other person's location | 139 / 256 | 54.3% |
| Holder | 254 / 256 | 99.2% |
| Overall | 645 / 1,024 | 63.0% |

The dropped-before-final-move football subset scores 43.8%. All 1,024 saved-model
predictions match the independent NumPy evaluator; native loss also agrees.
The audit confirms that dataset adaptation preserved every starting parameter.

This was one native initialization. The PyTorch and native runs use different
initial weights, shuffle sequences, and completed epoch budgets; their accuracy
gap does not isolate a backend effect. A full 80-epoch native run was not
completed, and no student-laptop or physical-GPU speed claim follows from this
cloud CPU measurement. Further native seeds were not launched after the recipe
failed this bounded check. No assignment package was rebuilt.

## Reproduction and evidence

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python \
  scripts/pilots/story-language/screen-football.py \
  --data pilots/football/mixed-small \
  --out pilots/football/replicate-mixed-small \
  --epochs 10 --qa-epochs 80 --seeds 137,211,307,419,503 --scratch

NEURALCANVAS_ROOT=/path/to/clean/app \
PLAYWRIGHT_ROOT=/path/to/playwright \
APP_URL=http://127.0.0.1:5186/NeuralCanvas/ \
node scripts/pilots/story-language/browser-pilot.mjs \
  pilots/football/native-mixed-small/137-pretrain.json \
  pilots/football/native-mixed-small/137-pretrain

NEURALCANVAS_ROOT=/path/to/clean/app \
PLAYWRIGHT_ROOT=/path/to/playwright \
APP_URL=http://127.0.0.1:5186/NeuralCanvas/ \
node scripts/pilots/story-language/browser-pilot.mjs \
  pilots/football/native-mixed-small/137-finetune.json \
  pilots/football/native-mixed-small/137-finetune

OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  scripts/pilots/story-language/summarize-football-replication.py
```

Raw screening files retain every seed, learning curve, final weights, and scratch
control. The audit checks history split separation, independent state-machine
answers, exact token/target alignment, paired drop-order cases, unchanged data,
the original seed's reproducibility, and native metrics against the independent
NumPy evaluator.
