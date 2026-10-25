# Assignment design: independent investigation and automatic grading

Recorded 2026-10-18 at the instructor's request. The principles below are standing preferences. Updated after instructor approval of the Day 7/8 rewrite. The current assignments implement the football sequence below. Assignments 7–19 are unpublished and are revised in place, with earlier drafts retained only in Git history.

## Current dataset workflow (2026-10-24)

All Assignments 1–19 now select their exact dataset in **Dataset → Details →
Browse datasets…**. Name the category, family, configuration/preset and any
changed control using the app's exact visible labels. Dataset uploads and Canvas
attachments are unnecessary. Preserve model-file imports and saved-model
submissions. Vocabulary, token lengths/shapes, answer space and split sizes
remain in the data-properties tables. The file-import instructions in older
dated decisions below are superseded by this browser workflow.

Use scripts/assignments/dataset_browser.py to keep choices consistent. Narrative
final-position and all-position presets must be distinguished, as must mixed QA
training/evaluation, mixed training with football-only evaluation, and combined
answer-token QA. Keep the capstone pretraining checkpoint before switching a
copy to QA. Data, splits, rubrics and numerical targets do not change. The only
grader edits are three messages pointing to the browser. Archived issued
releases and original datasets remain intact. See
[DATASET-BROWSER-ASSIGNMENTS.md](DATASET-BROWSER-ASSIGNMENTS.md).

## Current lesson order (2026-10-23)

Sampling now immediately follows the bigram model: 11 bigram, 12 sampling,
13 context MLP, 14 causal self-attention, 15 residuals/FFN, 16 normalization and
all-position training, 17 stacking, 18 multiple heads, 19 QA fine-tuning.
Assignment 12 uses the unchanged Assignment 11 bigram weights and final-position
dataset; it does not retrain. Assignment 13 starts with that saved base.json.
Generation controls apply throughout the sequence; later handouts include optional
reminders. Assignment 18 now hands multihead.json directly to Assignment 19.
See [SAMPLING-SEQUENCE.md](SAMPLING-SEQUENCE.md) for the numbering map and checks.
Older dated decisions below retain the numbering used when recorded.

## Standing principles

- Each assignment should motivate a new tool through a limitation or tradeoff in the students' current tools. A measurable improvement in every lesson is not required; aim for a clear advantage the instructor can point to, such as better predictions, faster training, fewer parameters, lower memory use, more parallel computation, greater training stability, or a useful new capability. Lecture prepares students for the corresponding assignment. Reduce procedural guidance for familiar steps.
- Distinguish measured advantages from architectural properties. Derive parameter counts and explain which operations can run in parallel; validate claims about runtime, memory, convergence, and stability under stated conditions. Parallelism does not guarantee faster execution in the browser, and equal parameter counts do not imply equal computation. A lesson can remain valuable without an accuracy gain; do not merge it solely because loss stays similar. Required student work still needs an automatically graded artifact, and grades must not depend on student hardware speed or unverifiable training history.
- Every substantial required independent task must produce an automatically graded artifact. Eliminate required but unsubmitted investigations. Inspection can support a submitted model. Clearly labeled optional activities are allowed when the required work independently delivers the lesson; the instructor explicitly allows the ungradable review-reversal experiment on this basis.
- All assessed work must be automatically graded. Do not require explanations that an instructor must read, and do not substitute AI scoring of prose for objective grading.
- Keep model construction and investigations in Neural Canvas. Do not add a Gradescope quiz or another question-answering platform. The existing upload of saved models for automatic grading can remain; this preference concerns where students do the intellectual work.
- Pair a guided model with an independent, explicitly constrained variant on the same dataset when pedagogically useful. Potential changes include vocabulary or embedding size, hidden width, input order, activation, or learning rate. Students should determine the consequent changes themselves rather than follow another complete recipe.
- Prefer variants that reliably improve a specified metric over an intentionally limited but meaningful baseline. Give students an attainable target, validated across initializations. Do not make the baseline poor through an arbitrary mistake or promise that every change improves generalization. A representation limitation may instead make improvement impossible; that can be the intended discovery.
- Change a small number of factors at a time. Reward relevant design decisions rather than extensive random tuning, repeated initialization, or long training runs. Retain the original data and split; vocabulary and other preprocessing remain fitted on training data only.
- Debugging is appropriate when the failure illustrates the current lesson, including a model that runs but discards necessary information. Subtle errors can take hours to find: bound the investigation, make the symptom observable, and offer targeted hints when needed. Avoid arbitrary shape errors, hidden typos, and lengthy fault-finding exercises unrelated to that lesson.
- PyTorch export, adaptation for Colab, and automatically graded code are acceptable exceptions when they serve a specific lesson about code or execution. Do not assume exports run unchanged, or introduce Python environment repair merely to add difficulty. No such addition is currently proposed for Days 7/8.
- Remove explanations that prematurely disclose an intended discovery, as far as practical. Keep enough instruction to operate the app, enough conceptual scaffolding to make the challenge fair, and explicit submission requirements. Instructor notes should retain the explanations and expected outcomes.

## What automatic grading can establish

Grade saved model operations, representation settings, fitted preprocessing, parameters, and independently recomputed outputs on canonical data. Accept equivalent constructions within the learning constraints. Do not trust cached losses, probabilities, declared epoch counts, or arbitrary block labels.

A saved model does not prove that a student performed an inspection, predicted an outcome, or tried a particular learning rate. Reversal with fixed weights is a useful in-app self-check, but do not award points for allegedly completing it unless a meaningful artifact records it and can be verified. No written explanation or self-reported metric is required.

For improvement challenges, prefer fixed, independently piloted baseline and final requirements over a score based only on the difference between a student's two submissions. A relative-only rubric can reward deliberately damaging the baseline. If improvement is graded explicitly, require a valid, adequately trained baseline as well. The held-out set is validation during tuning, not an untouched final test.

Representation changes require the grader to reconstruct the corresponding canonical features from the original dataset. Allowing a different vocabulary size must not permit altered reviews, labels, split membership, or validation-fitted vocabulary. Test malformed and adversarial submissions as well as full-credit models.

## Approved whole-story sequence and presentation

Assignments 9 and 10 now use one whole story token sequence plus a separate
question, with the same original texts, answers and group-disjoint splits.
Assignment 9 adapts the Assignment 8 embedding MLP with learned token positions
and a question encoder (37,417 parameters). Assignment 10 replaces the flattened
story path with cross-attention over story tokens (6,441 parameters). Both use
five continuous Adam epochs at 0.003, batch 32, reporting every five, clip norm 1,
no decay/penalty and no early stopping. Longer training is optional; grade saved
quality rather than training-history claims. The shared football benchmark and
Assignment 9's required 5-point unchanged-weight evaluation remain.

Use the three-seed epoch-five evidence for numerical targets: Assignment 9 train
loss at most 0.75; Assignment 10 train/held-out loss at most 0.80/0.85 and mixed
accuracy at least 60%. Football accuracy remains diagnostic in both. Assignment
19 accepts the new Assignment 10 model as earlier.json; its transformer, QA
recipe and performance targets are unchanged. Earlier fact-row architectures and
30-epoch cutoffs below describe superseded releases.

The instructor requests continuity and understandability without additional
solution disclosure. State the previous saved model to copy, the architectural
change, and which outputs to keep. Give dataset file imports in prose. Present dataset properties, training and
initialization settings in tables. Preserve independent shape and wiring work.
The deliberate fresh start in Assignment 11 introduces next-token generation;
subsequent assignments progressively modify that generator. See
[WHOLE-STORY-REVISION.md](WHOLE-STORY-REVISION.md).

The direct story-question representation now supplies [32] story IDs, [8]
question IDs and [32,1]/[8,1] word masks; the six legacy input adapters are gone.
Prepared JSON selects these lengths. Preserve the useful embedding/classifier
reshapes that belong to the model. See [DIRECT-STORY-INPUTS.md](DIRECT-STORY-INPUTS.md).

## Prepared datasets and app settings tables (2026-10-22)

Distinct student datasets must have distinct filenames across current assignments,
including differences in representation, targets or preprocessing. Identical
datasets may reuse the same filename. Canvas uploads share a course namespace;
do not rely on assignment folders to disambiguate different files. Validate this
with `scripts/assignments/audit-dataset-filenames.py`. Narrative final-position
targets use `football-narratives-last.json`; all-position targets use
`football-narratives-all.json` throughout the sequence.

Use prepared dataset JSON throughout Assignments 7–19. Each file embeds its task,
tokenizer, representation, fitted vocabulary, lengths, documents and split.
Students open it through the Dataset text importer, choosing the file and
clicking Import text dataset. The current app ignores the other import-dialog
controls for JSON. Custom text import can remain a separate app option; no app
changes are part of this coursework update. Assignments 1–6 retain their issued
materials.

Tables that prescribe app settings must use the app's exact visible labels,
option values and UI order. Do not combine dataset descriptions or inferred
values with controls in a settings table. Use short prose to identify prepared files and the import action; do not make a
table containing only a filename. Use a separate Prepared dataset properties
table for vocabulary size, input lengths/shapes, answer space and split sizes
needed to determine Param shapes. Identify
the panel for each table. Train controls, Loss → Details → Penalty, and each
Param's Initialize choices are separate. Conditional controls belong only to
recipes where they are visible, such as Weight decay with AdamW. State any
prerequisite selection that reveals controls before the ordered table.

Export early prepared files from validated canonical data without refitting or
resplitting. Preserve private raw bytes and provenance, numerical features,
independent student shape/wiring work, model targets and grading rules. See
[PREPARED-DATASETS.md](PREPARED-DATASETS.md) for validation and reproduction.

## Current football sequence (2026-10-21)

The instructor approved football-narratives throughout Assignments 11–17,
with Assignment 18 sampling the same trained model. Assignment 17's ten-epoch
training supplies the pretraining checkpoint. Delete the separate pretraining
Assignment 19 and move former QA Assignment 20 to `19-qa-finetuning`.
There is one current assignment numbered 1–19 and no Assignment 20.

Use the exact validated 25-token football corpus and history split. Keep the
existing architectural lessons, 5/10-epoch starting budgets, independent model
construction and automatically graded saved artifacts. Recalibrate early loss
bounds using the three-seed corpus pilots; Assignment 11's old 0.90 training
bound is below its empirical best possible bigram loss. Assignment 17 and the
final QA prerequisite must use identical narrative-quality targets.

The default QA recipe remains five additional epochs from the student's own
Assignment 17 values. Optional longer pretraining and fine-tuning use five-epoch
runs and saved checkpoints. Assignment 19 also offers optional controlled
experiments from unchanged before.json: ten continuous QA epochs with AdamW,
weight decay 0.1 and learning rate 0.003; a matching Adam comparison; and smaller
decay/rate trials. These are suggestions, not additional required work or graded
training-history requirements. Preserve the chosen narrative values exactly in
pretrained.json and before.json. Grade saved model quality, not runtime or epoch
counts. GPU support may help; the ten-minute preference is relaxed. Extra
training offers no guaranteed accuracy gain. Keep 70% overall QA accuracy and
both drop/move orders as diagnostics, with no separate 80% gate. Compare the
identical held-out questions against Assignment 10. See
[STUDENT-CAPSTONE.md](STUDENT-CAPSTONE.md) and
[NARRATIVE-REVISION.md](NARRATIVE-REVISION.md) for current evidence and recipes.
The sections below preserve earlier decisions and do not override this sequence.

### Earlier supplied-checkpoint release

The instructor approved implementing the selected football capstone and rebuilding
Assignments 7–10 so the final attention and transformer models face the same
questions and held-out histories. Assignments 1–6 remain protected. The original
start/end Day 8 exercise and two-fact capstone are superseded; Git retains them.

The ten-minute late-assignment budget may use GPU training; CPU-only execution
is not a requirement. Validate convergence on the proposed backend and record
the actual hardware. Software-rendered cloud WebGL can establish compatibility,
but its timing is not evidence of physical laptop-GPU performance. Do not grade
students on runtime or require them to own a particular GPU.

Assignment 7 preserves guided IMDb sentiment classification, then requires a
linear word-count model on five-event football stories. Both files are graded;
review reversal remains optional because the saved counts cannot prove reversal.
Assignment 8 uses exactly the same football CSV, preserves token order, and
compares an eight-unit one-hot MLP with four-dimensional embeddings and a 32-unit
hidden layer. Parameter counts are 3,190 and 3,242; both train ten epochs. The
six selected native pilots gave 25.8–65.8% versus 72.9–76.2% accuracy. This is a
parameter/update-budget comparison, not equal measured computation or a promise
of uniformly lower cross entropy. An eight-dimensional candidate showed no
consistent advantage and was rejected.

Assignment 9 introduces structured facts/questions and a shared ordered encoder.
Assignment 10 replaces the flattened classifier with question-directed attention,
reducing parameters from 16,601 to 8,153 with the corrected concatenation head
(the older addition head used 7,129). The larger MLP can outperform attention
on football tracking: do not fabricate an accuracy improvement. Assignment 10
requires a trained mixed-question model and a fixed-weight football-only
evaluation copy. The latter is reused in Assignment 20 for an exact common-task
comparison. Every required comparison produces an automatically graded artifact.

Assignment 20 uses a supplied, genuinely pretrained two-block/two-head football
generator (5,689 parameters) and a controlled answer-only fine-tuning recipe.
Its new-task epoch counter starts at zero without changing learned values.
Different training orders and arbitrary pretrained initializations were not
uniformly successful; keep that limitation explicit. Require preservation of
the canonical supplied parameters in before.json, not just agreement between
two arbitrary submissions. Grade at least 80% on each of the two drop-order
halves so easy cases cannot hide the intended failure. A stronger earlier model
must never lose points. Pretraining benefit over scratch training is not claimed.

Assignments 11–19 retain their independently validated narrative generation
sequence, vocabulary, datasets and models. The supplied capstone model uses the
same conceptual architecture with a separate football vocabulary. Assignment 19
remains synthesis because the capstone does not need a separate preparation day.
See [FOOTBALL-SEQUENCE-REVISION.md](FOOTBALL-SEQUENCE-REVISION.md) for implementation,
validation and current recipes. The sections below retain the earlier decisions
and experimental history; they do not supersede this current sequence.

## Revised Days 9–20 trajectory (2026-10-19)

### Synthetic-story pilot authorization (2026-10-19)

The instructor has authorized exploring a replacement for Assignments 11–20 using
word-level generation of synthetic narratives. The entire 11–20 sequence may
change. Assignments 7–10 may also change if that better prepares students; none of
Assignments 7–20 has been assigned. Preserve issued Assignments 1–6.

Pilot before revising the handouts or graders. Look for observable benefits from
new tools, and distinguish meaningful story-state predictions from easy sentence
templates. The instructor clarified that every lesson need not produce an
empirical improvement: prefer a defensible advantage, including parameter or
memory efficiency, parallelizability, training speed or reliability, or a new
capability. Record the comparison, supporting evidence or architectural argument,
and relevant tradeoffs. Do not require a separate accuracy win for each component.
Instruction tuning to answer questions about the stories is a desirable optional
endpoint, contingent on evidence; it is not a required final assignment. Compare
with a model trained on question-answer examples without narrative pretraining.
The instructor subsequently clarified that Assignments 11–19 should focus on
generative models and need not relate to question answering. Judge those models
by narrative continuation and generation, including consistency with preceding
events; keep any question-answer adaptation separate as an optional final
extension. Do not let that extension dictate the generative-model sequence.
Target at most about five minutes per required training run, with up to ten minutes
acceptable for the last few assignments. Report the actual validation hardware
and backend rather than promising these runtimes for every student laptop.

The instructor subsequently authorized continuing the pilots and rewriting the assignments once a workable progression emerged. The current narrative implementation is documented in [NARRATIVE-REVISION.md](NARRATIVE-REVISION.md); it replaces the unpublished Alice materials in place. Assignments 1–9 are unchanged; Assignment 10 only corrects an obsolete handout reference. There is one current assignment per day; Git retains earlier drafts.

The earlier [stacking revision](STACKING-REVISION.md) and [v2 review](ADVANCED-V2-REVIEW.md) remain historical records. The current sequence below supersedes their Alice release details.

### Historical narrative progression

Days 11–14 introduce next-word generation, ordered context, self-attention and residual/feed-forward processing on the fixed synthetic corpus. Assignment 15 now combines normalization with all-position supervision: students submit the independently trained final-position baseline, the all-position model and an unchanged-weight final-position evaluation copy. Parts A and B each train for ten epochs; Part C never retrains. Compare final-position accuracy/loss across the two trained models, not the all-position average with the final-position average. Both training runs use the same architecture, input windows and optimizer settings: ten epochs and 480 updates each. The comparison examines final-position prediction when every position contributes supervision. Compute cost and random initialization can still affect results; no improvement is guaranteed.

The instructor requested this merge because changing the dataset and removing a Slice was too little independent work for a separate assignment, and approved renumbering the remaining assignments 17–20 as 16–19. Assignment 16 adds a second block, 17 adds a second head per block at fixed total width, 18 constructs temperature branches without retraining, and 19 compares existing generators. Day 20 now introduces question-answer fine-tuning, as authorized below. Keep one current folder and upload ZIP per assignment; the retired standalone 16 and former later numbers remain only in Git history.

All required work produces automatically graded saved models. Graders additionally generate held-out continuations and report grammar and story consistency; those diagnostics are informational, not hardware-sensitive or fragile performance requirements. Parameter-count hints remain in the handouts. The benefit of each lesson is explicit in the instructor guide. Empirical comparisons and architectural benefits are kept distinct. The final fine-tuning task is documented separately from the generative progression in `QA-CAPSTONE-PILOT.md`.

### Question-answer capstone authorization (2026-10-19)

The instructor requested experiments toward a final assignment that fine-tunes
the generative model on two-fact stories ending in a question and correct answer,
connecting back to Assignment 10. The question mark may be omitted. The instructor
also authorized preparing the earlier generative vocabulary with `where` to avoid
resizing learned tensors in Assignment 20. The selected approach reserves that
word from Assignment 11 onward without changing the story text, splits or other
token IDs. Its input embedding is unused during narrative pretraining and learns
during question-answer fine-tuning. This reservation is explicit; it is not
vocabulary learned from held-out data.

Pilot the revised vocabulary and exact fine-tuning recipe before releasing the
capstone. Keep Days 11–19 focused on generation. The final lesson is adapting a
pretrained next-token predictor to a question-answer task; do not promise an
advantage over training from scratch or over the dedicated Assignment 10 model.
Required comparisons must end in saved, independently graded model files. A
matched before file can verify preservation of the starting parameters, but a
saved JSON cannot prove the optimizer trajectory. Keep interactive prompting
optional unless its outcome can be independently assessed.

### Harder football pilot (2026-10-20)

The instructor requested piloting generative stories for Assignments 11–19 with
multiple moves and a football, followed by fine-tuning in Assignment 20. The final
model should answer person-location questions, football-location questions, and
“Who has the football?” Include passes between people, as well as carrying and
dropping. A dropped ball remains behind when its former carrier moves.

Keep questions out of the earlier narrative training task, reserve necessary
question words in its vocabulary, and keep all variants of a history in the same
split. Evaluate each question type separately, including counterfactual drop-order
pairs, rather than relying on overall token accuracy. Retain the five-minute
ordinary / ten-minute late-model training goals. Pilot before revising the current
releases. The private results and limitations are in `FOOTBALL-PILOT.md`.

The instructor subsequently accepted an 80% success target and authorized the
next pilot: two people, four rooms, pickup/carry/drop stories, and all three
question types, leaving passes for a possible later extension. Repeat the
existing promising configuration across initializations, aiming for at least
80% held-out accuracy in each question category, then validate native training
and runtime. A favorable single seed or an overall average that hides poor
person-location performance is insufficient. This changes the pilot's acceptance
target; it does not change any released grading threshold.

The instructor then authorized a shorter-history pilot retaining the
drop-before-moving distinction. Keep both person-location and football questions,
test multiple initializations, and check the actual app runtime before replacing
materials. The shorter-story experiments and all unsuccessful controls are
recorded in `FOOTBALL-SHORT-PILOT.md`.

The instructor next authorized a curriculum-learning comparison: six epochs of
person-location questions, then twelve epochs of all question types. Compare
against three mixed-question epochs followed by a fresh twelve-epoch mixed run.
Both arms see 57,600 examples in 1,800 updates and restart Adam after 360 updates.
Use the same pretrained parameters within each pair and retain all five selected
native initializations, including the previously failed seed 211. Check each
question category and the drop-before-moving subset independently. This is a
private pilot, not authorization to publish unvalidated numerical targets.
Protocol, results, runtime qualifications, and remaining workflow checks belong
in `FOOTBALL-CURRICULUM-PILOT.md`.
The completed pilot did not meet the reliability goal: one of five curriculum
runs cleared every category and the drop-before-moving subset, versus three of
five matched controls. Keep this unsuccessful recipe out of assignment releases.

### Consolidate the capstone decision (2026-10-20)

The instructor's current priorities are a QA task that Assignment 10 struggles
with, at least 80% held-out accuracy, and under ten minutes of student training.
Question types, story length, and whether to include a football or passes are
flexible. Fine-tuning is desirable to illustrate instruction tuning, but need
not outperform training from scratch. Targeted further pilots are authorized
when needed to support a concrete recommendation.

A substantial two-part capstone may replace Assignment 19's synthesis exercise:
the first part becomes Assignment 19 and the second Assignment 20. Do not split
merely to disguise a training-time or reliability problem.

If football tracking is selected, rebuild Assignments 7–10 to include football
stories. In particular, compare the final Assignment 10 and Assignment 20 models
on the same questions and held-out histories, using appropriate representations
of the same input information. Re-pilot the earlier lessons rather than assuming
a vocabulary change preserves their learning objectives or grading thresholds.
These are conditional design directions; preserve current releases until the
replacement task and progression are validated. The consolidated evidence and
recommendation belong in `ASSIGNMENT-20-RECOMMENDATION.md`.


### Required Assignment 9 common-task baseline (2026-10-21)

The instructor requires students to check the shared 240-question football
benchmark in the app and identify its accuracy as the baseline for Assignment 10
and the final QA capstone, Assignment 19. This is a 5-point fixed-weight
comparison: submit baseline.json with stories.json and football.json with the
identical model parameters and football-test.json. Move 5 points from training
loss so the total stays 100. There is no accuracy minimum for this comparison.
The grader verifies both canonical datasets and model forms, recomputes results
and checks matching learned parameters; it does not trust saved inference reports
or attempt to prove a student read the result. No prose or screenshot is required.
Assignments 7 and 8 are unchanged.


### Assignment 10 architecture and grading correction (2026-10-21)

Use the handout’s concatenation head: attended fact values and the encoded
masked question’s ReLU output, before Q, produce 32 classifier inputs and 8,153
parameters. The old 7,129-parameter addition head is no longer accepted, as
requested by the instructor. Retain the required word masks, scaled
attention-score mask and output probabilities. Accept mathematically equivalent
spellings of the documented score mask and equivalent handling of padded fact
rows. Apply the same architecture and masking requirements to Assignment 19’s
earlier model. Preserve datasets, point allocations, loss
thresholds and accuracy targets; report actual comparisons without imposing an
artificial advantage for a later model.
