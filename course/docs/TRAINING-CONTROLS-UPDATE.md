# Training instructions and optional QA experiments

Numbering note: this dated report retains its original assignment numbers. See
[SAMPLING-SEQUENCE.md](SAMPLING-SEQUENCE.md) for the current sequence after moving sampling to Assignment 12.

Updated 2026-10-21 for clean app `d2a649d94ffb0d4dfab3f37ca248a08aa4b22bae`.
The app now presents controls and help according to the engine, optimizer and
early-stopping setting. The current handouts name the settings students can use:

- Assignments 1–5 retain Trace SGD and omit instructions about its hidden Tensor
  controls. Assignment 2 retains its required Loss-block L2 experiment.
- Assignments 6–17 and 19 retain early stopping patience 0, omitting minimum
  validation improvement, which is hidden and inactive when patience is 0.
- Adam and SGD recipes omit weight-decay instructions; that control applies
  only to AdamW. Gradient clipping remains explicit because it is still active
  for every Tensor optimizer. Loss-block penalties remain separate and explicit.
- Assignment 18 requires no training, so its handout needs no settings change.

Assignment 19 keeps the five-epoch Adam default and offers optional experiments:
AdamW with decay 0.1 and ten continuous QA epochs, an otherwise matching ten-epoch
Adam run, and smaller-decay or learning-rate trials. Reopen the same before.json
for each comparison, retain its epoch counter and parameters, and save each
result separately. Several shorter runs reset the optimizer's running averages;
they do not reproduce a continuous run. See [the capstone report](STUDENT-CAPSTONE.md)
for the earlier three-start training evidence and its limits.

These are instruction changes. Grader sources, datasets, reference models,
targets, rubrics and issued archives are preserved. Packaging refreshes handout
hashes; every current Gradescope ZIP must remain byte-for-byte identical to its
pre-edit version. Optional training does not add work to fixed-size grading.
The narrative handout generator and capstone report writer include the revisions.

## Local verification

[Browser evidence](training-controls-browser.json) checks Adam/SGD/AdamW control
visibility, the minimum-improvement field appearing only with positive patience,
Trace SGD's hidden Tensor controls, and File → Save with the optional ten-epoch
AdamW recipe. Old hidden values survive saves, so students should select the
prescribed optimizer and patience rather than try to clear invisible controls.
The check verifies unchanged learned values and epoch counter and renders all
19 handouts at widths 390 and 960. It does not train or validate convergence.
Original numerical pilots retain their original app commits. The current app
changes touch presentation, not the numerical training engine.

[Release verification](training-controls-validation.json) records all 19 unchanged
ZIP hashes, protected data/graders/models/archives, generator consistency and the
27 passing Assignment 19 grader tests. An actual ten-epoch AdamW/decay-0.1
checkpoint also receives all 35 available after.json points from the independent
grader. No grader update is needed for the optional experiments.

With the local app server running:

```sh
NEURALCANVAS_ROOT=/workspace/NeuralCanvas PLAYWRIGHT_ROOT=/tmp/course-browser node scripts/assignments/browser-training-controls.cjs
.venv/bin/python scripts/assignments/advanced/audit-narratives.py
```

These checks cover the local checkout. The deployed student app and hosted
Gradescope were not exercised. A Git push does not publish handouts to Canvas.
