# Assignment 13 — Give the generator a longer memory

Public teaching release: see [the course guide](../../../README.md) for standalone setup, packaging, and validation. Wiring, templates, grader sources, and these notes are public. Trained reference models, browser model exports, solution-based tests, and pilot/reproduction tools mentioned below remain in the private NeuralCanvas-Coursework checkout; their paths are historical development references, not required teaching files. Test your own saved models when configuring Gradescope.

## Teaching purpose

An ordered context distinguishes situations that have the same last word.

Assignments 11–18 build and sample football-narrative generators. Assignment 18's trained model supplies pretraining for Assignment 19 QA; no separate pretraining assignment or vocabulary resize remains. See [STUDENT-CAPSTONE.md](../../../docs/STUDENT-CAPSTONE.md) and [NARRATIVE-REVISION.md](../../../docs/NARRATIVE-REVISION.md).

## Publish

Paste `../description.html` into Canvas. Dataset attachments are unnecessary; students use the dataset browser. Upload `Gradescope/gradescope-13-context.zip`. Require `context.json`. Keep trained reference models private. A Git push does not publish to course sites.

## Dataset and training

The exact football corpus used in the pretraining pilots has 25 vocabulary tokens, including reserved `where` and `who`, two people, six rooms, five-event pickup/carry/drop histories and consequence statements. Context length is 32, stride 17: 1,534 training windows and 765 validation windows. Room-layout groups, actor variants and event-order variants stay in one split, aligned with Assignment 19 QA. Final-position and all-position fixtures differ only in targetMode. Repeated or overlapping windows are not independent observations. Held-out data is validation during development.

Tensor CPU pilot: Adam 0.003, batch 32, shuffle, 5 epochs, report every 1, clip norm 1, patience 0, minimum improvement 0, no penalty or decay. Architecture changes start from fresh parameters; evaluation copies retain learned values.

Initialize embedding tables Uniform, matrices Xavier, biases Zeros, norm scales Ones and offsets Zeros. Use a supported WebGL GPU if helpful; Tensor CPU is the tested baseline and fallback. Cloud/browser timing is not a physical laptop GPU benchmark. Epoch counts and runtime are not graded.

## Grading

| File | Data / form / train / held-out / accuracy points | Train loss ≤ | Held-out loss ≤ | Parameters |
|---|---|---|---|---|
| context.json | 10 / 50 / 20 / 20 / 0 | 0.6 | 0.7 | 34,857 |

The bounded NumPy evaluator verifies canonical data, semantic graph operations, parameter shapes/counts and causal masks. It recomputes predictions; cached scores and epoch counts are ignored. Informational generation diagnostics use 16 fixed held-out football openings ending in `now`, generate up to 12 tokens without feeding future words, and distinguish consequence syntax from factual consistency. There are no written submissions or extra generation thresholds.

## Validation and reproduction

Clean app `9d535a02b53bff5859e963d07cdf2d65df6cfacf`. Three seeds per architecture: 21 new Tensor CPU runs plus three retained exact two-block/two-head football pretraining runs on software WebGL. Objective and common final-position metrics were independently recomputed. See `pilots/football/progression-candidate/audit.json` and `docs/narrative-seed-grading.json`. Early bigram/attention/residual cutoffs were recalibrated; final multihead targets match the QA pretraining requirements; sampling uses the bigram targets. This does not promise an accuracy gain at every architecture change. Local package/browser checks do not establish hosted Gradescope or deployed-site behavior.

```sh
RELEASE_ASSIGNMENTS=13-context .venv/bin/python scripts/assignments/advanced/configure-narratives.py
RELEASE_ASSIGNMENTS=13-context .venv/bin/python scripts/assignments/package-advanced.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/13-context/source -p test_grader.py -v
```

The generator rebuilds this unpublished release in place. Update it before regenerating the handout. Retained pilot evidence keeps its original paths and numbering.

## Instruction organization update

The current Canvas handout uses short dataset import instructions and tables of dataset properties, training controls and initialization choices and makes the prior-model handoff explicit. Independent shape and wiring work, datasets, model forms, training budgets and grading rules are preserved. See [the whole-story and continuity revision](../../../docs/WHOLE-STORY-REVISION.md).

Current numbering and handoffs reflect [the early sampling update](../../../docs/SAMPLING-SEQUENCE.md); retained pilot filenames keep their historical numbering.

## Dataset browser workflow

Students open **Dataset → Details → Browse datasets…** and click **Use this dataset** after selecting the following choices. Datasets live in the app’s `public/datasets/` directory; no assignment-local copies or Canvas attachments are needed. Read [description.md](../description.md) on GitHub or use `../description.html` in Canvas. Model JSON submissions and grading targets are unchanged.

- **Text and sequences → Football stories**; preset: **Narrative generation**; Training targets: **Final position only**.


Browser/grader compatibility is checked with app 417be2e and the original canonical data. See [the dataset browser update](../../../docs/DATASET-BROWSER-ASSIGNMENTS.md).
