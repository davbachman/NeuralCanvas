# Dataset browser workflow for Assignments 1–19

Updated 2026-10-24 at the instructor's request. Every current handout selects data
through **Dataset → Details → Browse datasets…**, then **Use this dataset**.
Students still use File → Import for saved models and submit saved model JSON to
Gradescope. Dataset file attachments are no longer needed in Canvas.

| Assignment | Category → family | Configuration / preset and changes |
| --- | --- | --- |
| 1 | Regression → Auto MPG | Configuration: Full sample |
| 2 | Regression → Auto MPG | Configuration: Small training sample |
| 3 | Classification → Breast cancer measurements | Use this dataset |
| 4 A | Regression → Auto MPG | Configuration: Full sample |
| 4 B | Classification → Iris flowers | Use this dataset |
| 5 | Regression → Concrete strength | Use this dataset |
| 6 | Classification → Concentric rings | Use this dataset; multiclass task is supplied |
| 6 optional | Images → Handwritten digits | Use this dataset; optional ungraded extension |
| 7 A | Text and sequences → Movie reviews | Use this dataset |
| 7 B | Text and sequences → Football stories | Football location · word counts |
| 8 | Text and sequences → Football stories | Football location · word counts; Model inputs and prediction: One ordered token sequence → answer class |
| 9–10 main | Text and sequences → Football stories | Mixed-question classifier |
| 9–10 evaluation copy | Text and sequences → Football stories | Mixed training · football benchmark |
| 11–15 | Text and sequences → Football stories | Narrative generation; Training targets: Final position only |
| 16 A/C | Text and sequences → Football stories | Narrative generation; Training targets: Final position only |
| 16 B | Text and sequences → Football stories | Narrative generation; Training targets: Every input position |
| 17–18 | Text and sequences → Football stories | Narrative generation; Training targets: Every input position |
| 19 QA copy | Text and sequences → Football stories | Answer-token fine-tuning; save pretrained.json before selecting QA in a separate copy |

Instructions retain vocabulary size, token lengths, masks, answer classes, split
counts and independent model-building work. Saved-model filenames, architectural
requirements, training budgets, loss/accuracy limits and point allocations stay
the same. The first six handouts also use the current Dataset **Rows** label for
selecting training rows; Data-tab **Examples** navigation remains unchanged.

The exact 24 original classroom file copies (14 distinct datasets) remain beside
the handouts as canonical provenance/compatibility fixtures, matching the app
assets byte-for-byte. Existing uploaded-dataset projects remain valid. Historical
issued archives are unchanged. Three grader error messages now name the browser
choices (MPG, concrete and rings); no evaluation code or criteria changed.
Their current upload ZIPs were refreshed. Other upload ZIPs remain byte-identical;
all 19 package manifests now cover the revised handout HTML.

## Validation and reproduction

The shared choices are in scripts/assignments/dataset_browser.py. The narrative
builder and the capstone guide generator use the current workflow. Canvas HTML
is authoritative; avoid rebuilding validated model/evidence directories merely
to update wording.

- audit-dataset-browser.py checks every current handout, 24 unchanged data copies,
  all canonical data.json files, source equality apart from the three declared
  feedback strings, self-contained ZIP contents, manifests and unchanged archives.
- browser-dataset-library.cjs follows all 25 required/optional choices in the
  rendered app, including Auto MPG's full/small configurations and Iris. It saves
  and reimports real model projects, checks canonical data and unchanged learned
  parameters/epochs, and checks all 19 handouts at phone and desktop widths.
- grade-dataset-browser.py evaluates those browser saves with each current upload
  ZIP, filling remaining comparison files from unchanged instructor references.
- The affected MPG, concrete and rings grader suites passed all 43 tests covering
  complete, partial, malformed and packaged submissions. The shared narrative audit still protects
  the course progression and the exact Assignment 18 → 19 handoff.
- The settings-table browser check follows the new library route while checking
  the unchanged Train, initializer, regularization and generation tables.

App version for local checks: 417be2e78cbb87a82e6605925ad3cdae5fd67c25. Source inspection and browser checks use the app repo without changing it.
The production build is served locally. No retraining, Canvas publishing or hosted
Gradescope action is performed by this update.

```sh
NEURALCANVAS_ROOT=../NeuralCanvas .venv/bin/python scripts/assignments/audit-dataset-browser.py
PLAYWRIGHT_ROOT=/path/to/playwright-project NEURALCANVAS_ROOT=../NeuralCanvas APP_URL=http://127.0.0.1:5196/NeuralCanvas/ node scripts/assignments/browser-dataset-library.cjs
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/assignments/grade-dataset-browser.py
.venv/bin/python scripts/assignments/advanced/audit-narratives.py
```

See dataset-browser-assignment-audit.json, dataset-browser-assignment-ui.json,
dataset-browser-packaged-grading.json and settings-table-ui-validation.json for
bounded evidence. Browser submission paths are temporary; rerun the browser check
before grading if they have been removed. Private reference models remain outside
the app repository.
