# Introduce sampling after the bigram model

Implemented 2026-10-23 at the instructor's request. Sampling now comes immediately
after the first next-token generator, so students can use probabilistic generation
throughout the remaining architecture sequence.

| Current assignment | Lesson | Previous number |
| --- | --- | --- |
| 11 | Bigram next-token generation | 11 |
| 12 | Temperature and probabilistic generation | 18 |
| 13 | Ordered-context MLP | 12 |
| 14 | Causal self-attention | 13 |
| 15 | Residual paths and feed-forward network | 14 |
| 16 | Normalization and all-position training | 15 |
| 17 | Two transformer blocks | 16 |
| 18 | Multiple attention heads | 17 |
| 19 | QA fine-tuning capstone | 19 |

## Student workflow and grading

Assignment 12 starts with Assignment 11's trained bigram.json. Students save an
unchanged base.json and independently add Softmax(logits / T) branches for T=0.5
and T=2 in sampling.json. The original logits, loss and output Softmax remain.
Both files have 825 parameters and use the identical prepared
football-narratives-last.json. Their loss limits match Assignment 11: training
1.15 and held-out 1.20. No training or randomization is required. The existing
100-point rubric is retained, including 10 points for matching learned values.

Optional generation compares Most likely and Sample with the same prompt,
temperatures 0.5/1/2 and Top-k 25. The handout distinguishes generation controls
from inspection branches and explains that sampling cannot add context to a
bigram model. Generation clicks and stochastic continuations are not graded.

Assignment 13 starts with base.json, preserving the previous-model workflow
without carrying unnecessary temperature branches into the architecture work.
Assignments 13–18 remind students that the same generation controls apply. The
capstone includes an optional revisit and explains that sampled answers do not
determine grader accuracy. Assignment 18's trained multihead.json now goes
directly to Assignment 19. Architecture recipes, learned reference values,
training budgets, targets and rubrics for these moved lessons are preserved.
No fixed-seed requirement has been added.

Folder names, current upload ZIPs, Canvas descriptions, instructor guides,
package manifests, active build/validation scripts and the course schedule use
the new numbering. Historical pilot filenames, reports and their recorded app
commits retain their original numbers. Use the table above when interpreting
older evidence; no new full-budget architecture training is claimed.

## Validation and reproduction

The early sampling validator checks all three retained five-epoch bigram starts:
losses, unchanged predictions/parameters, exact Assignment 11 handoff, matching
datasets in 11/12/13, and both temperature branches. Full grader suites exercise
model form, incorrect branches, data/weight tampering and the extracted upload
ZIPs. Existing native evidence is independently regraded under the new numbers,
and the capstone validator checks the exact Assignment 18 → 19 handoff.

Browser validation uses clean app
871a4a8a7dc733e6296e4c1d6c017abb18eef297. It checks greedy next-token parity with
NumPy, actual sampling controls at all three temperatures, no generated padding,
unchanged parameters/epochs, saving and reimport. Prepared dataset imports and
settings tables are checked against the rendered app; all current handouts are
checked at phone and desktop widths. No hosted Gradescope or Canvas publishing
is part of this update.

```sh
.venv/bin/python scripts/assignments/package-advanced.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/advanced/validate-early-sampling.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/advanced/validate-narratives.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/football/validate-student-capstone.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/advanced/test-narratives.py
PLAYWRIGHT_ROOT=/tmp/course-browser NEURALCANVAS_ROOT=/workspace/NeuralCanvas APP_URL=http://127.0.0.1:5192/NeuralCanvas/ node scripts/assignments/advanced/browser-early-sampling.cjs
PLAYWRIGHT_ROOT=/tmp/course-browser node scripts/assignments/football/check-handouts.cjs
.venv/bin/python scripts/assignments/advanced/audit-narratives.py
```

Evidence is in Assignment 12's reference/early-sampling-validation.json and
reference/browser/sampling-ui.json, docs/narrative-seed-grading.json,
docs/narrative-ui-grading.json, docs/student-capstone-validation.json,
docs/current-assignment-checks.json, docs/settings-table-ui-validation.json and
docs/football-handout-layout.json. Reference material stays private.
