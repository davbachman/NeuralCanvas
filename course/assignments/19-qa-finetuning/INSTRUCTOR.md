# Assignment 19 — Fine-tune your generator to answer questions

Public teaching release: see [the course guide](../../README.md) for standalone setup, packaging, and validation. Wiring, templates, grader sources, and these notes are public. Trained reference models, browser model exports, solution-based tests, and pilot/reproduction tools mentioned below remain in the private NeuralCanvas-Coursework checkout; their paths are historical development references, not required teaching files. Test your own saved models when configuring Gradescope.

Use the student's trained multihead.json from Assignment 18 as pretrained.json.
There is no separate narrative-pretraining assignment. See
[STUDENT-CAPSTONE.md](../../docs/STUDENT-CAPSTONE.md) for the recipe, optional
training experiments, retained evidence, exact handoff and limitations.

## Publish

Paste Canvas/description.html into Canvas. Dataset attachments are unnecessary; students use the dataset browser.
Upload Gradescope/gradescope-19-qa-finetuning.zip. Require earlier.json,
pretrained.json, before.json and after.json. Keep trained reference models private.
Canvas/description.html is authoritative; the builder does not overwrite it.

## Grading

| File | Data / form / training / held-out / accuracy points | Limits (train / held-out / accuracy) |
| --- | --- | --- |
| earlier.json | 5 / 10 / 0 / 0 / 0 | None / None / None |
| pretrained.json | 5 / 10 / 5 / 5 / 0 | 0.6 / 0.65 / None |
| before.json | 5 / 10 / 0 / 0 / 0 | None / None / None |
| after.json | 5 / 10 / 5 / 5 / 10 | 0.75 / 0.75 / 0.7 |

Another 10 points require identical learned parameters in adequately trained
pretrained.json and before.json. The narrative limits match Assignment 18
exactly. All three retained initializations and the separate retained UI run pass.
Both drop/move orders are diagnostic; no per-group 80% requirement remains.
Epoch counts, optimizer choices and runtime are ignored. Optional experiments
include AdamW with decay 0.1, rate 0.003 and ten continuous QA epochs from the
unchanged before.json, compared with a matching ten-epoch Adam run. Smaller
decay/rate trials and additional five-epoch runs are optional. Keep checkpoints;
no setting guarantees higher accuracy. The five-epoch default remains unchanged.
See the matched three-start evidence in STUDENT-CAPSTONE.md and
`pilots/football/qa-adamw-10/`.

## Validation and reproduction

Clean app `9d535a02b53bff5859e963d07cdf2d65df6cfacf`. Independent native/UI checkpoint evaluation,
exact Assignment 18 handoff, canonical question/history split identity, causal
prediction, answer exclusion, parameter preservation, malformed/partial files
and extracted ZIP execution are checked locally. SwiftShader observations are
not physical GPU benchmarks; hosted Gradescope and deployed-site checks remain
outside this local validation.

Current control visibility and simplified handouts were checked separately on
clean app `d2a649d94ffb0d4dfab3f37ca248a08aa4b22bae`; see
[training-control update](../../docs/TRAINING-CONTROLS-UPDATE.md). This is a
settings/save check, not new convergence evidence.

```sh
.venv/bin/python scripts/assignments/football/build-student-capstone.py
RELEASE_ASSIGNMENTS=19-qa-finetuning .venv/bin/python scripts/assignments/package-advanced.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/football/validate-student-capstone.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/19-qa-finetuning/source -p test_grader.py -v
```


## Whole-story Assignment 10 import

The earlier.json check now requires Assignment 10's five-epoch whole-story
cross-attention architecture: 6,441 parameters, token padding/score masks and the
encoded question concatenated before the classifier. Superseded sentence-row
architectures are rejected. Equivalent bounded affine score-mask formulas remain
accepted. Its three retained football accuracies are 58.33%, 60.42% and 68.75%.
Narrative pretraining, QA data/objectives, loss targets and point allocations
remain unchanged. The default transformer checkpoints still pass.

The handout presents the starting files and their distinct import paths in
tables, separates the default five-epoch recipe from optional ten-epoch optimizer
experiments, and leaves the answer-position Slice bounds independent.
Actual browser-saved earlier/pretrained/before/after files are independently
checked with the rebuilt local upload ZIP. See
reference/browser/attention-zip-grading.json and the whole-story revision report.
No hosted publishing is implied.

## Direct story/question inputs and handout update (2026-10-22)

Assignments 9/10 and this capstone's earlier-model reference now use the app's
story-question representation. Six nontrainable input adapters are removed;
vocabulary, class IDs, original story/question text, document order/splits,
learned parameters, epoch counts, recipes and numerical targets are preserved.
The capstone's transformer QA input remains one 23-token sequence and an answer
next-token target. See ../../docs/DIRECT-STORY-INPUTS.md for conversion and checks.

The current handout uses file instructions in prose and a Prepared dataset
properties table for vocabulary, lengths, shapes, answer space and split sizes.
Independent Param-shape and model-wiring work is retained. Browser and native
checks use clean app 871a4a8a7dc733e6296e4c1d6c017abb18eef297; historical full-budget
training evidence retains its original app commit.

Current numbering and handoffs reflect [the early sampling update](../../docs/SAMPLING-SEQUENCE.md); retained pilot filenames keep their historical numbering.

## Dataset browser workflow

Students open **Dataset → Details → Browse datasets…** and click **Use this dataset** after selecting the following choices. Dataset files retained beside the handout are canonical provenance/compatibility copies; they do not need to be attached in Canvas. Model JSON submissions and grading targets are unchanged.

- **Text and sequences → Football stories**; preset: **Answer-token fine-tuning**.


Save the selected Assignment 18 generator as pretrained.json before selecting the QA preset in a separate copy. Its narrative data and learned values stay in pretrained.json.


Browser/grader compatibility is checked with app 417be2e and the original canonical data. See [the dataset browser update](../../docs/DATASET-BROWSER-ASSIGNMENTS.md).
