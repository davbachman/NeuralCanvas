# Choosing the Assignment 20 task

The student-owned capstone in [STUDENT-CAPSTONE.md](STUDENT-CAPSTONE.md)
supersedes the supplied-checkpoint recommendation below. The instructor has
removed the absolute 80% requirement and permits relaxing the ten-minute budget.
The goal is clear improvement over Assignment 10 on the same questions.

Current football update (2026-10-20): [FOOTBALL-SEQUENCE-REVISION.md](FOOTBALL-SEQUENCE-REVISION.md) describes the implemented Assignments 7–10 and 20. Their earlier dataset/backend statements below are historical. The generation evidence for Assignments 11–19 is retained.

The instructor's priorities, updated 2026-10-20, are:

1. A question-answering task on which the Assignment 10 model struggles.
2. At least 80% held-out answer accuracy.
3. Under ten minutes of training for the student.

Fine-tuning is desirable as an illustration of instruction tuning, but need not
beat training from scratch. The football, question types, and story length can
all change. A larger task is not inherently a better capstone.

This report consolidates the private pilots. It does not revise any assignment
handout, dataset, or grading threshold.

The instructor also permits a two-part capstone replacing Assignment 19's
synthesis exercise. If football tracking is selected, Assignments 7–10 must be
rebuilt around compatible football stories, with an exact common QA evaluation
for Assignments 10 and 20. Those earlier lessons need their own validation.

## Recommendation

Use **five-event football stories, two people, six rooms, and no passes**.
Fine-tune a supplied, pretrained two-block/two-head transformer using a fixed
ten-epoch recipe. Keep person-location, football-location, and possession
questions in training, but make the scored comparison the **balanced
football-location test**, where dropping before moving and moving before dropping
are equally frequent.

The verified full-UI run answers **240/240** scored questions correctly in
**447.009 seconds (7 minutes 27 seconds)**. It also answers all **1,920/1,920**
questions in the retained full mixed validation set correctly in independent
evaluation. Before fine-tuning it answers none of the scored questions correctly.
The Assignment 10 architecture achieves **55–58.33%** on the identical scored
test across three native initializations, after **three times as many QA updates**
as the chosen ten-epoch transformer recipe.

This meets the three priorities in the tested setup. It is a recommendation for
the **specific supplied checkpoint and training protocol**, not for arbitrary
student-pretrained models or training schedules. The failed UI controls below
are important evidence for that restriction.

The short stories expose a concrete need for context: a fact saying that Nora
dropped the football does not name its location. The model must connect it to
Nora's preceding location and distinguish a later move from an earlier one.
Assignment 10's independently encoded facts and single attention read struggle
with this here; the contextual transformer succeeds. This is an observed
advantage of the tested workflows, not a proof of impossibility for every
single-attention model or an isolated estimate of the benefit of pretraining.

With a supplied checkpoint, the fine-tuning itself fits Assignment 20. Assignment
19 remains available for substantial preparation if the rewritten student
workflow needs it; the timing result does not require a split.

## What the earlier experiments establish

| Candidate | Evidence | Assessment against the new priorities |
|---|---|---|
| Two static facts, person-location questions; current Assignment 20 | Three native runs: 98.4–100%; isolated fine-tuning 6.14 minutes | Good fine-tuning demonstration, but no demonstrated advantage over Assignment 10 |
| Long movement/drop stories, no passes | Four of five PyTorch seeds pass every category; native run reaches the ten-minute limit with 63% overall | Too slow/unreliable for the requested capstone |
| Eight-event stories with passes and four question categories | Constrained PyTorch runs: 75%, 100%, 74.6%; more varied stories substantially worse | Too ambitious for this budget |
| Five-event stories, four rooms, compact questions | Native overall 84.8–100%, but drop-before-moving scores 21.9%, 100%, 56.3% | Aggregate accuracy conceals the important failure |
| Five-event stories, six rooms, all question categories | Native overall 81.7–100%; one run confuses the other person and fails half the drop-before-moving cases | Strong candidate with a selected shared checkpoint; not reliable across arbitrary pretraining initializations |
| Person-location warm-up before mixed questions | Only one of five native runs clears every category and the hard drop-before-moving subset; matched control clears three of five | Do not add the warm-up as the default recipe |

Sources: [original football pilot](FOOTBALL-PILOT.md),
[longer-story replication](FOOTBALL-REPLICATION.md),
[shorter stories](FOOTBALL-SHORT-PILOT.md),
[curriculum comparison](FOOTBALL-CURRICULUM-PILOT.md), and
[current QA capstone](QA-CAPSTONE-PILOT.md).

These are validation results used in selecting a task, not an untouched final
test. PyTorch screens are not native-app training or laptop timing measurements.

## Targeted comparisons

The new focused fixture retains only football-location questions and the two
drop-order variants from the six-room task. It has 480 training and 240
held-out questions. Each room layout stays in one split, including both actor
assignments and both drop orders. There are 120 matched held-out pairs.
The two orders have different correct answers but the same bag of words.
Always returning the carrier's final room scores exactly 50%.

The Assignment 10 comparator is its actual ordered fact encoder, position
embeddings, one question-to-facts attention read, and hidden classifier. It
retains ten fact slots, eight word slots, and width 16. Only the vocabulary,
class labels, and dataset are adapted. **It is not a bag-of-words encoder**;
the comparison tests learning across facts, not a fabricated inability to see
word order within a sentence.

Both architectures use native Neural Canvas Tensor CPU, Adam 0.003, batch 32,
gradient norm limit 1, and no weight decay or early stopping. Three fixed seeds
are retained, including failures. The focused transformer gets 40 QA epochs
from each corresponding native generative checkpoint. Assignment 10 gets 120
epochs from scratch, with the learning curve reported every ten epochs.

An additional comparator trains Assignment 10 on the complete mixed-question
dataset: 30 epochs, reporting at 15 and 30. Fifteen epochs matches the
earlier fifteen-epoch transformer fine-tuning budget; thirty gives the earlier
architecture three times the QA updates of the final ten-epoch recommendation.
Both see exactly the same underlying stories,
questions, targets, and split, represented through their appropriate dataset
interfaces. The transformer also has narrative pretraining, so this is a
comparison of practical course workflows, not an isolated architecture ablation.

Native models, full learning curves, and configurations are under
`pilots/football/focused-location/`. The independent audit recomputes logits,
losses, and every held-out answer from saved operations and weights.

### Completed native controls

| Model and training questions | Seed | Full mixed test | Balanced football test | Drop before moving |
|---|---:|---:|---:|---:|
| Assignment 10, football only, 120 epochs | 137 | — | 76.25% | 52.5% |
| Same | 211 | — | 74.58% | 49.17% |
| Same | 307 | — | 76.25% | 52.5% |
| Assignment 10, all question types, 30 epochs | 137 | 94.11% | 55.0% | 10.0% |
| Same | 211 | 94.74% | 58.33% | 16.67% |
| Same | 307 | 94.38% | 55.0% | 10.0% |
| Transformer, football only, 40 epochs | 137 | — | 69.17% | 38.33% |
| Same | 211 | — | 72.5% | 45.0% |
| Same | 307 | — | 69.58% | 39.17% |

All these runs answer the move-before-drop half correctly. The mixed-task
Assignment 10 models also answer all person-location and possession questions
correctly. Their high overall scores therefore do not demonstrate successful
football tracking. The two Assignment 10 controls have 6,934 and 7,129 parameters,
respectively; the output class counts differ. The transformer has 5,689.

The football-only transformer pilot uses fewer total QA updates than the
mixed-question recipe. It establishes failure of that forty-epoch recipe,
not that auxiliary questions are causally necessary or that further tuning
could never make football-only training work.

### Proposed scored task

The `challenge.json` fixture keeps **all 3,840 mixed training questions** and
evaluates **240 balanced football questions**: 120 drop-before-move and 120
move-before-drop. This deliberately changes the evaluation question distribution
to target the intended lesson. It is not presented as accuracy over all questions.
All 1,920 mixed held-out questions remain available for a separate diagnostic.
No histories move between training and validation.

Both Assignment 10 and the transformer must use this same scored test, with the
same mixed training examples. The earlier architecture's measured benchmark on
that task is 55–58.3%, despite receiving three times the final transformer
recipe's QA updates.
The football-only control also shows that simply focusing Assignment 10's
training on the test's question type does not cross 80% in the tested runs.

A shared generative checkpoint is a practical alternative to requiring every
student to arrive with an equally good independently pretrained model. Its
selection uses the pilot validation results and must be disclosed. A fixed
starting checkpoint is a narrower, reproducible assignment protocol; it does
not erase the failed initialization results or establish reliability for
arbitrary pretrained models.

## Full-UI validation and the supplied checkpoint

All three UI runs use the same seed-137 generative weights, Adam 0.003, Tensor
CPU, batch 32, reshuffling each epoch, reporting every five epochs, clipping
norm 1, no decay, and no early stopping. The model has 5,689 parameters.

| UI recipe | Starting epoch counter | Football test | Drop before moving | Training time |
|---|---:|---:|---:|---:|
| 15 QA epochs | 10 | 79.17% | 58.33% | 639.7 s |
| 10 QA epochs | 10 | 77.08% | 54.17% | 462.1 s |
| **10 QA epochs, supplied checkpoint protocol** | **0** | **100%** | **100%** | **447.0 s** |

The app derives its shuffle order from the epoch counter. The original native
pilot starts the new QA phase at counter zero, while importing an ordinary
pretraining save and replacing its dataset retains counter ten. This is not an
app arithmetic error: the optimizer sees a different order of examples.
The otherwise matched ten-epoch UI controls show substantial sensitivity to
that order. Longer training with the retained counter did not fix it.

The private `supplied-generator.json` is the existing ten-epoch generative
checkpoint with **unchanged learned tensors, graph operations, vocabulary, and
narrative dataset**. Only the new task's epoch/loss bookkeeping and the
reset-to-start parameter snapshot are prepared afresh. The audit checks these
claims directly against the original source checkpoint. It does not pretend
that the supplied model is untrained: its provenance explicitly records ten
pretraining epochs. Students would receive that prepared checkpoint, rather
than edit JSON or reset/randomize its learned weights themselves.

The successful UI run starts with the prepared answer-only graph and exact
pretrained parameters, uses the real Dataset importer to replace its data,
checks that every parameter is preserved, and runs ten epochs in one invocation.
It then runs Test inference and saves through File → Save. The epoch counter
goes from zero to ten. Independent NumPy evaluation agrees with the UI:
football loss **0.000886823**, both drop-order halves **120/120**, and the complete
mixed diagnostic **1,920/1,920**. No page errors occurred. The required construction
of the final-row Slice is prepared by the harness; this is not a recording of
every student wire being drawn.

The fifteen-epoch timing is isolated from other training. The ten-epoch control
and final run overlap for part of their duration; their 7.7/7.5-minute times are
actual, contended UI measurements on the four-core AMD EPYC cloud host, using
Chromium 151 and clean app commit
`2b6b96d8d07ee52e0beca3c3432f9f4e5819af79`. They are not student-laptop or physical-GPU
benchmarks. The first UI launch failed before import/training because of stale
Vite optimized dependencies; the unchanged run was retried after restarting the
local development server. That infrastructure failure is recorded separately.

For the final assignment, preserve this common starting checkpoint and the
one-run ten-epoch recipe. An 80% target on each drop-order half would leave a
substantial margin below the observed result. This is a proposed threshold,
not a change to the current grader. A more open-ended task using students' own
pretrained models still needs additional reliability work.

## Implications for the course sequence

If the football capstone is selected:

- Keep Assignment 7's successful movie-review model, and use football stories
  for its independent word-count challenge. Matched drop-order pairs have the
  same word counts and different answers.
- Re-pilot Assignment 8's ordered one-hot versus embedding comparison on
  compatible football stories. A harder corpus does not automatically preserve
  its equal-cost embedding advantage.
- Re-pilot the Assignment 9 structured MLP and Assignment 10 attention model
  on the same underlying histories. Earlier lessons may use clearly identified
  task subsets, but keep the split boundaries consistent.
- Make the final Assignment 10/20 comparison use identical training questions,
  targets, and held-out football cases. Give each architecture its normal input
  representation, containing the same story/question information. Report person
  questions separately from the order-sensitive football task.
- Do not require students to produce a poor Assignment 10 score. Grade valid,
  trained artifacts and report the limitation; a student who finds a stronger
  earlier model should not lose credit.

The optional two-assignment ending should have distinct purposes. Assignment 19
can establish the harder tracking task, evaluate the earlier attention approach,
and prepare/inspect the narrative-generation starting model. Assignment 20 then
changes the objective to answer-only prediction and fine-tunes it. Any required
comparison must produce a saved model artifact for automatic grading.

A supplied **pretrained** transformer checkpoint is the most controlled starting
point for the final lesson. Students already know how to build its architecture.
This lets the capstone teach adapting a model rather than repeating all the
construction steps, and avoids making success depend on each student's earlier
random initialization. It is also faithful to the usual instructional use of
a shared pretrained model. It does not require claiming that pretraining beats
learning this small task from scratch.

A successful checkpoint alone does not validate new materials for Assignments
7–10 or a newly trained student model from Assignment 19. Those remain separate
implementation/pilot steps. Current assignment releases stay unchanged while
this recommendation is reviewed.

## Reproduction and audit

The clean app checkout, local Vite server, Chromium, and Playwright are required
for native runs. Set `NEURALCANVAS_ROOT`, `PLAYWRIGHT_ROOT`, and `APP_URL`.
Run UI benchmarks separately from training replications when an isolated timing
is wanted.

```sh
.venv/bin/python scripts/pilots/story-language/prepare-focused-qa.py
.venv/bin/python scripts/pilots/story-language/run-focused-qa.py --workers 3
node scripts/pilots/story-language/browser-focused-qa.cjs 137 challenge 15 10
node scripts/pilots/story-language/browser-focused-qa.cjs 137 challenge 10 10
node scripts/pilots/story-language/browser-focused-qa.cjs 137 challenge 10 0
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/pilots/story-language/audit-focused-qa.py
```

The original first two UI output folders predate adding the epoch offset to the
folder name; their report fields identify the exact settings. Native runners
reuse complete reports. Use a fresh output location for genuinely fresh native
replications rather than treating reused reports as new training.

The final audit passes for all nine native comparisons and all three completed
UI runs, including failures. It checks data/split identity, deterministic fixture
hashes, exact checkpoint preservation, example/target alignment, full completed
epoch budgets, and independently recomputed predictions/losses. Its separate
`recommendationCriteria` fields confirm all four checks: 80% overall, 80% on
each drop-order half, measured UI time below 600 seconds, and Assignment 10
below 80% on the same task. Private evidence is in
`pilots/football/focused-location/audit.json`.
