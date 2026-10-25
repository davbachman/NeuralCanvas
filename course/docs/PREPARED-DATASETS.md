# Prepared datasets and app settings tables

Numbering note: this dated report retains its original assignment numbers. See
[SAMPLING-SEQUENCE.md](SAMPLING-SEQUENCE.md) for the current sequence after moving sampling to Assignment 12.

Assignments 7–19 now all use prepared dataset JSON. Students choose the supplied
file in the Dataset text importer and click **Import text dataset**. The JSON
restores the task, tokenizer, representation, fitted vocabulary, lengths,
documents and split. The current app ignores the dialog's other controls for
prepared JSON. **File → Import** opens saved models instead.

Assignment 7 now supplies `imdb.json` and `football-counts.json`; Assignment 8
supplies `football-tokens.json`. This makes the counts-to-ordered-token change
explicit while preserving the original football questions and split. Assignments
9–19 already supplied prepared JSON. Assignments 1–6 retain their issued files.
The custom text importer remains available. Assignments 9/10 now use the app’s direct story-question representation; see [DIRECT-STORY-INPUTS.md](DIRECT-STORY-INPUTS.md).

Distinct datasets have distinct upload filenames across current assignments.
`football-narratives-last.json` is shared by Assignments 11–14 and 15 Parts A/C;
`football-narratives-all.json` is shared by 15 Part B and Assignments 16–18.
Those variants use the same text and input windows with different targets. This
filename correction preserves the original JSON bytes and embedded metadata,
canonical grader data, models and upload ZIPs. The other reused dataset names
already identify identical files. The [filename audit](dataset-filenames.json)
covers current Assignments 1–19; historical pilot reports retain their original
filenames. Recheck with `.venv/bin/python scripts/assignments/audit-dataset-filenames.py`.

## Preserved behavior

The new early files are exported from the validated `reference/data.json`
preprocessing, changing only their display filename. No data was regenerated,
vocabulary refitted, or split reassigned. Numerical features, answer indices,
model architectures, parameter counts, training budgets and grading thresholds
remain the same. The raw CSV bytes and existing provenance hashes stay private
under each early assignment's `reference/raw/` directory. The graders continue
reconstructing features independently and accept the original filename metadata.

Student handouts retain the independent shape and wiring exercises. Their imports now use short file instructions. **Prepared dataset properties**
tables show vocabulary size, input lengths/shapes, answer space and split sizes
needed to determine Param shapes. These are supplied data properties, not controls.
All settings tables use the app's labels, option text and order. The Train tab,
Loss block's **Penalty**, and Param **Initialize** choices have separate tables.
Prerequisite choices explain conditional fields such as AdamW's **Weight decay**
and sampling's **Temperature** and **Top-k**.

## Local validation

The [browser report](settings-table-ui-validation.json) records checks against
clean app commit `871a4a8a7dc733e6296e4c1d6c017abb18eef297`. It verifies the
39 settings tables against rendered controls, imports all 17 prepared student files
with deliberately incompatible dialog settings, saves and compares the embedded
data, and independently grades the early-model exports. Learned parameters are
preserved. Training-setting changes preserve both weights and epoch count.

Assignment 7's 14-test suite passes with three legacy cases skipped; Assignment
8's 12-test suite passes. These cover canonical preprocessing and raw provenance,
prepared-data equivalence, original pilot/browser exports, malformed submissions
and extracted upload ZIPs. The thirteen handouts pass phone and desktop layout
checks; representative import and training tables were visually reviewed. The
single-version package audit passes for all thirteen unpublished assignments.

Full-budget training was not repeated for this equivalent input packaging
change. Existing training evidence remains valid and is retained. These are
local app, grader and archive checks; no hosted Gradescope test or Canvas upload
was performed.

## Reproduction

From the coursework root:

```sh
.venv/bin/python scripts/assignments/prepared_early.py
.venv/bin/python scripts/assignments/package-word-counts.py
.venv/bin/python scripts/assignments/package-embeddings.py
.venv/bin/python -m unittest discover -s assignments/07-word-counts/source -p test_grader.py -v
.venv/bin/python -m unittest discover -s assignments/08-embeddings/source -p test_grader.py -v
PLAYWRIGHT_ROOT=/path/to/browser-dependencies NEURALCANVAS_ROOT=/path/to/NeuralCanvas APP_URL=http://127.0.0.1:5190/NeuralCanvas/ node scripts/assignments/browser-settings-tables.cjs
PLAYWRIGHT_ROOT=/path/to/browser-dependencies node scripts/assignments/football/check-handouts.cjs
.venv/bin/python scripts/assignments/advanced/audit-narratives.py
```

Run the browser checks against the app server. `prepared_early.py` needs only
committed canonical fixtures; it performs no training or downloads. The football
installer calls the same export after installing its selected pilots. Shared
table formatting lives in `scripts/assignments/handout_settings.py`.
