# Dataset library validation

Implemented and checked locally on 2026-10-24, based on app commit
04952e85008fbaf33df318042464aa17205976ca. The library is opened from a Dataset
block's Details tab using Browse datasets. See [the user guide](DATASETS.md).

## Coverage and compatibility

- 17 categorized families, with 32 supported configurations: all 14 existing
  built-in sources, 14 distinct classroom files, and four additional datasets.
- The 24 current classroom dataset copies across Assignments 1–19 match the 14
  included app assets byte-for-byte. The classroom checkout used for this check
  was 3a4ee14c18b3eed884748b223ee09ab326be4ac0. No private graders, rubrics,
  instructor models or answer guides were copied into the app.
- Asset SHA-256 hashes are in src/learning/dataset-library.json. The additional
  Wine, Diabetes, Penguins and California housing files reproduce byte-for-byte
  with scripts/prepare-library-datasets.py, using vendored public source bytes.
- Domain tests check every configuration using the app's actual dataset APIs,
  fixed splits, metadata, shapes, target types and file hashes. Football checks
  cover shared training records, question coverage, common held-out questions,
  answer exclusion, separate masks and narrative target alignment.
- Original custom CSV and Text / sentiment source choices are preserved. The
  new Custom data page opens those same importers. Prepared JSON remains
  authoritative regardless of the text importer's default controls.
- Saved projects embed the selected data and retain the library source label.
  Installation uses existing target-wire remapping and undo behavior. Library
  CSVs include the existing companion numerical data file with Python/notebook
  export, so a separately downloaded original CSV is unnecessary.
- The library has no prepared-dataset download button.

## Checks

- Full app suite: 109 files passed, one skipped; 1,114 tests passed and 29 skipped.
- npm run build and npm run lint passed.
- Chromium 151 browser checks passed against both Vite development and the
  production build: family/category navigation, football controls and shapes,
  California housing loading, dataset installation, saving/reimport, and custom
  imports through both the existing and new routes. No page errors were recorded.
- Both current introductory text files, imdb.json and football-counts.json,
  imported successfully through the original source selector with every importer
  control left at its default. These are the active Assignment 7 instructions.
- Dialog/category and football/QA screenshots were visually reviewed.

Reproduction commands, from the app root:

```sh
npm ci
CI=1 npm test
npm run lint
npm run build
python scripts/prepare-library-datasets.py
npm run preview -- --host 127.0.0.1 --port 5196
# In another terminal, with Playwright available at PLAYWRIGHT_ROOT:
PLAYWRIGHT_ROOT=/path/to/playwright-project APP_URL=http://127.0.0.1:5196/NeuralCanvas/ node scripts/check-dataset-library.cjs
```

The browser script uses only the public linear example and app dataset assets.
It writes screenshots and a bounded browser report to ignored output/dataset-library
(or OUTPUT_DIR). It does not require the private coursework repository.
These are local checks; pushing invokes the repository's normal Pages CI/deploy
workflow. Hosted deployment is separate from the local validation above.
