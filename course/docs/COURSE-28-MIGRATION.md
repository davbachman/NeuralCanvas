# Migration to the 28-assignment course

The instructor authorized this revision on 2026-11-11. The sequence manifest
at [sequence.json](../sequence.json) is authoritative for current folders.
Earlier reports and recipe versions keep their historical identifiers; a recipe
version is distinct from the current assignment position.

| Former folder | Current assignment |
| --- | --- |
| `01-mpg` | [03-features](../assignments/03-features/description.md) |
| `02-regularization` | [05-regularization](../assignments/05-regularization/description.md) |
| `03-classification` | [06-classification](../assignments/06-classification/description.md) |
| `04-matrices` | [07-matrices](../assignments/07-matrices/description.md) |
| `05-mlp` | [08-mlp](../assignments/08-mlp/description.md) |
| `06-rings` | [10-rings](../assignments/10-rings/description.md) |
| `07-word-counts` | [12-word-counts](../assignments/12-word-counts/description.md) |
| `08-embeddings` | [13-embeddings](../assignments/13-embeddings/description.md) |
| `09-positional-embeddings` | [14-positional-embeddings](../assignments/14-positional-embeddings/description.md) |
| `10-attention` | [15-attention](../assignments/15-attention/description.md) |
| `11-next-word` | [16-next-word](../assignments/16-next-word/description.md) |
| `12-sampling` | [17-sampling](../assignments/17-sampling/description.md) |
| `13-context` | [18-context](../assignments/18-context/description.md) |
| `14-self-attention` | [19-self-attention](../assignments/19-self-attention/description.md) |
| `15-residuals` | [20-residuals](../assignments/20-residuals/description.md) |
| `16-normalization` | [21-normalization](../assignments/21-normalization/description.md) |
| `17-stacking` | [22-stacking](../assignments/22-stacking/description.md) |
| `18-multiple-heads` | [23-multiple-heads](../assignments/23-multiple-heads/description.md) |
| `19-qa-finetuning` | [24-instruction-tuning](../assignments/24-instruction-tuning/description.md) |
| `20-sparse-autoencoder` | [27-sparse-autoencoder](../assignments/27-sparse-autoencoder/description.md) |
| `21-feature-interventions` | [28-feature-interventions](../assignments/28-feature-interventions/description.md) |

New assignments occupy 01-predictions, 02-gradient-descent, 04-generalization,
09-backpropagation, 11-convolution, 25-rlvr and 26-rlhf. The earlier MPG and
regularization lessons are revised to distinguish multiple features,
generalization and regularization.

All current upload filenames are `gradescope-<current-folder>.zip`. This avoids
changing their names when a recipe version changes. Issued historical archives
remain in the private repository. Saved student artifact filenames are specified
by the current handout and are independent of directory numbering.

Old deployed student-page URLs redirect to the corresponding current lesson.
Private coursework compatibility links target the relocated public sources.
Public course files do not depend on the private checkout.

Rebuild from the app root:

```sh
python course/scripts/package-course.py
python course/scripts/sync-handouts.py
python course/scripts/validate-course.py
```

Package rebuilding regenerates Markdown from canonical HTML. Synchronization
also updates deployed pages, redirects, and manifests. Validation checks all 28
packages against source, dataset hashes, and missing/malformed submissions.
Full-credit references and semantic regression tests stay in the private
repository; their results are recorded separately from these public checks.
