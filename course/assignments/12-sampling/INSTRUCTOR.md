# Assignment 12 — Control how the model chooses words

Public teaching release: see [the course guide](../../README.md) for standalone setup, packaging, and validation. Wiring, templates, grader sources, and these notes are public. Trained reference models, browser model exports, solution-based tests, and pilot/reproduction tools mentioned below remain in the private NeuralCanvas-Coursework checkout; their paths are historical development references, not required teaching files. Test your own saved models when configuring Gradescope.

## Teaching purpose

Temperature controls the concentration of a probability distribution without retraining.

Assignments 11–18 build and sample football-narrative generators. Assignment 18's trained model supplies pretraining for Assignment 19 QA; no separate pretraining assignment or vocabulary resize remains. See [STUDENT-CAPSTONE.md](../../docs/STUDENT-CAPSTONE.md) and [NARRATIVE-REVISION.md](../../docs/NARRATIVE-REVISION.md).

## Publish

Paste `Canvas/description.html` into Canvas. Dataset attachments are unnecessary; students use the dataset browser. Upload `Gradescope/gradescope-12-sampling.zip`. Require `base.json`, `sampling.json`. Keep trained reference models private. A Git push does not publish to course sites.

## Dataset and training

The exact football corpus used in the pretraining pilots has 25 vocabulary tokens, including reserved `where` and `who`, two people, six rooms, five-event pickup/carry/drop histories and consequence statements. Context length is 32, stride 17: 1,534 training windows and 765 validation windows. Room-layout groups, actor variants and event-order variants stay in one split, aligned with Assignment 19 QA. Final-position and all-position fixtures differ only in targetMode. Repeated or overlapping windows are not independent observations. Held-out data is validation during development.

No new training: preserve the trained models and fixed-weight copies.

Initialize embedding tables Uniform, matrices Xavier, biases Zeros. Use a supported WebGL GPU if helpful; Tensor CPU is the tested baseline and fallback. Cloud/browser timing is not a physical laptop GPU benchmark. Epoch counts and runtime are not graded.

## Grading

| File | Data / form / train / held-out / accuracy points | Train loss ≤ | Held-out loss ≤ | Parameters |
|---|---|---|---|---|
| base.json | 5 / 15 / 10 / 10 / 0 | 1.15 | 1.2 | 825 |
| sampling.json | 5 / 25 / 10 / 10 / 0 | 1.15 | 1.2 | 825 |

An additional 10 points verify unchanged parameters in base, sampling.

The bounded NumPy evaluator verifies canonical data, semantic graph operations, parameter shapes/counts and causal masks. It recomputes predictions; cached scores and epoch counts are ignored. Informational generation diagnostics use 16 fixed held-out football openings ending in `now`, generate up to 12 tokens without feeding future words, and distinguish consequence syntax from factual consistency. There are no written submissions or extra generation thresholds.

## Validation and reproduction

Clean app `9d535a02b53bff5859e963d07cdf2d65df6cfacf`. Three seeds per architecture: 21 new Tensor CPU runs plus three retained exact two-block/two-head football pretraining runs on software WebGL. Objective and common final-position metrics were independently recomputed. See `pilots/football/progression-candidate/audit.json` and `docs/narrative-seed-grading.json`. Early bigram/attention/residual cutoffs were recalibrated; final multihead targets match the QA pretraining requirements; sampling uses the bigram targets. This does not promise an accuracy gain at every architecture change. Local package/browser checks do not establish hosted Gradescope or deployed-site behavior.

```sh
RELEASE_ASSIGNMENTS=12-sampling .venv/bin/python scripts/assignments/advanced/configure-narratives.py
RELEASE_ASSIGNMENTS=12-sampling .venv/bin/python scripts/assignments/package-advanced.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/12-sampling/source -p test_grader.py -v
```

The generator rebuilds this unpublished release in place. Update it before regenerating the handout. Retained pilot evidence keeps its original paths and numbering.

## Early sampling revision

Use Assignment 11's trained bigram model, not a transformer. Both files have 825
parameters, one final-position target, and the same 1.15/1.20 loss limits as the
bigram. The two extra probability branches do not affect the original loss.
No new training or randomization is required. Students use base.json for Assignment
13; later generation reminders reuse the same controls. See
[the sequence update](../../docs/SAMPLING-SEQUENCE.md) for current validation.

## Dataset browser workflow

Students open **Dataset → Details → Browse datasets…** and click **Use this dataset** after selecting the following choices. Dataset files retained beside the handout are canonical provenance/compatibility copies; they do not need to be attached in Canvas. Model JSON submissions and grading targets are unchanged.

- **Text and sequences → Football stories**; preset: **Narrative generation**; Training targets: **Final position only**.


Browser/grader compatibility is checked with app 417be2e and the original canonical data. See [the dataset browser update](../../docs/DATASET-BROWSER-ASSIGNMENTS.md).
