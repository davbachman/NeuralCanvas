# Football narrative progression: Assignments 11–19

Numbering updated 2026-10-23 for [early sampling](SAMPLING-SEQUENCE.md); original
training evidence and pilot filenames are retained.

Implemented 2026-10-21 after approval of the corpus-swap pilots. Assignments
11–18 now share football-narratives. Assignment 18 supplies the trained checkpoint
for QA. The separate pretraining assignment was deleted and former Assignment 20
is now Assignment 19. Assignments 1–10 retain their tasks and rubrics.

## Models, evidence and grading targets

Three native initializations per trained architecture were independently graded
against the new releases. There were 21 fresh Tensor CPU runs plus three retained
exact Assignment 18 architecture football runs on software WebGL. These are
local app-engine results; timings are contended and do not benchmark laptop GPUs.

| Assignment | Model | Target positions | Parameters | Held-out loss, three seeds | New train / held-out limits |
| --- | --- | --- | --- | --- | --- |
| 11 | bigram | Final | 825 | 1.073–1.080 | 1.15 / 1.2 |
| 12 | sampling (unchanged bigram) | Final | 825 | 1.073–1.080 | 1.15 / 1.2 |
| 13 | context | Final | 34,857 | 0.514–0.533 | 0.6 / 0.7 |
| 14 | self-attention | Final | 2,105 | 1.138–1.247 | 1.35 / 1.4 |
| 15 | residual | Final | 3,433 | 0.616–0.892 | 0.95 / 1.0 |
| 16 | transformer | Final | 3,529 | 0.495–0.525 | 0.65 / 0.7 |
| 16 | all_positions | All | 3,529 | 0.470–0.483 | 0.65 / 0.7 |
| 17 | stacked | All | 5,689 | 0.470–0.530 | 0.65 / 0.7 |
| 18 | multihead | All | 5,689 | 0.483–0.509 | 0.6 / 0.65 |

All three seeds meet the new targets. The bigram's empirical minimum possible
training loss is 0.921477, so its retired 0.90 bound was impossible on this corpus.
The attention/residual cutoffs were also recalibrated. These bounds support the
piloted recipes and allow margin; three seeds do not guarantee every initialization.

The context MLP reduces loss sharply relative to the bigram. Shared attention
uses far fewer parameters but has worse loss than that large MLP. Historical five-epoch normalization
runs improved all three seeds with exactly matching shared initial parameters.
Assignment 16 now trains both objectives for ten epochs on the same input windows,
with 480 optimizer updates each. In three matched-initialization trials,
all-position training improves final-token accuracy by 2.0–4.6 percentage points
and lowers final-token loss. Supervision and compute cost differ; improvement
is not guaranteed for every initialization.
See pilots/football/assignment15-matched-epochs/README.md for the matched pilots.
Depth and multiple heads teach contextual composition and distinct attention
patterns, not guaranteed accuracy gains. Two heads retain the same 5,689 Params
as one head in each of two blocks, while score storage and execution cost can rise.
The bigram's 825 Params exceed a direct 25 × 25 transition table's 625; its lesson
is autoregressive generation, not parameter compression.

## Data and student workflow

Two people, six rooms and five pickup/carry/drop events precede a consequence
statement. The vocabulary has 25 tokens, including punctuation, reserved IDs,
and explicitly reserved question words where/who. Narrative training contains
no questions. Vocabulary fitting uses training text; reserved question words are
explicit additions. Room-layout groups and their actor/event-order variants stay
in one split, aligned exactly with the final QA task.

Windows have 32 input tokens, stride 17: 1,534 training and 765 validation windows.
Final/all-position files have identical documents, vocabulary and inputs. There
are 48 Adam updates per epoch at batch 32, compared with 80 on the earlier corpus.
Equal epoch counts therefore do not imply equal optimization budgets. Overlapping
windows and repeated histories are not independent observations. Held-out data
is validation during development, including task and checkpoint selection.

Assignments 11–15 use football-narratives-last.json with a final-position target.
Assignment 16 uses football-narratives-last.json in Parts A/C and
football-narratives-all.json in Part B. Submit transformer.json (40 points),
all_positions.json (40), last_position.json (10), plus 10 for equal B/C parameters.
All three graphs have 3,529 Params. Assignments 17–18 use all-position
football-narratives-all.json. Architecture changes train fresh values; fixed-weight
evaluation and sampling copies never randomize or retrain.

Starting budgets are five epochs for Assignments 11, 13, 14 and 15;
ten for 16A, 16B, 17 and 18; and no training in 12 or 16C. Use Tensor/Adam 0.003, batch 32,
shuffle, clip norm 1, no decay, penalty or early stopping. Tensor CPU was tested;
a supported WebGL GPU may help. Grades ignore runtime and epoch counts.
Assignment 18 uses the same narrative targets
as the final QA prerequisite, 0.60/0.65. Optional extra pretraining is completed
before creating the QA baseline. No separate pretraining run or vocabulary resize
remains in Assignment 19. See [STUDENT-CAPSTONE.md](STUDENT-CAPSTONE.md).

## Diagnostics and validation

Graders independently evaluate canonical data and semantic graph operations,
including exact trainable parameter counts and causal masking. Generation uses
16 fixed held-out openings ending in now, generates up to 12 tokens without
feeding future words, and reports consequence grammar and event consistency.
These diagnostics are informational: fluent syntax and low average loss do not
establish reliable state tracking. Students submit editable saved models; there
are no required written answers, screenshots or unsubmitted investigations.

Evidence: narrative-seed-grading.json (24 model evaluations),
narrative-data-validation.json (vocabulary/windows/split), narrative-browser.json
(current UI imports, independent generation parity, CPU training, inference,
saving and responsive handouts), narrative-ui-grading.json, current-assignment-checks.json,
and each reference/test-log.txt plus packaged-test-results.json. The QA validator
regrades retained native/UI checkpoints and verifies direct Assignment 18 handoff.
Original narrative/browser app commit 9d535a02b53bff5859e963d07cdf2d65df6cfacf;
the matched ten-epoch Part A runs use clean app
42708105d80c2c47fd2b90a26f5add201a4c2ce5. Hosted Gradescope and the
deployed student app were not checked. Older backend and pilot reports describe
their original corpora and numbering, not new football convergence results.

Rebuild using configure-narratives.py, build-student-capstone.py and
package-advanced.py; then run validate-narratives.py, validate-student-capstone.py,
test-narratives.py and audit-narratives.py. Current sources and references are
self-contained. Canvas descriptions and datasets are the student materials;
source/reference files stay private. One current folder and stable ZIP exist per
assignment. Removed releases remain in Git history.
