# Assignment 27 — Interpret the final adapted transformer

This release follows instruction tuning, verifiable-reward adaptation and
preference-reward adaptation. Students use their own final `rlhf.json`. The
interpretability pair stays together at the end of the course. No trained
checkpoint or completed SAE construction is supplied for this assignment.

## Handoff and learning objective

Preserve the original QA policy as `rlhf.json`, then make a narrative-evaluation
copy named `baseline.json`. Use Train → Reward learning → Clear reward setup
to remove old reward references without changing the current weights. Select the unchanged all-position football narrative
dataset, reconnect the full vocabulary logits to Loss and Softmax, and remove
the unused final-position Slice. Do not train or change any parameter. The
vocabulary and 32-entry position table stay the same; QA used 23 input positions,
while narrative inspection uses 32. Both transformer blocks remain intact.

This changes the contexts used to inspect the final model, not the model's
learned weights. Narrative cross entropy is diagnostic. The old pretrained
narrative CE limits do not apply after QA and reward adaptation. The submitted
original policy establishes adequate QA and preference-reward behavior through
independent recomputation. Its parameter-and-operation fingerprint must match
the narrative copy and both SAE constructions. Merely labeling a checkpoint
“RLHF” or copying a cached training score does not establish the handoff.

Use the residual after the second block's feed-forward residual addition,
immediately before final output Layer norm. Freeze the base transformer and
construct `h = ReLU(r We + be)` and `r_hat = h Wd + bd` with 32 features and four
new trainable Params. Constrain decoder rows to unit norm. Keep the original
vocabulary path for inspection. Reuse the single Loss for MSE against the frozen
internal residual, with all valid token positions selected.

## Controlled comparison and discovery

From one student-constructed untrained SAE, train an unpenalized reconstruction
and a sparse comparison. Use Tensor CPU, Adam 0.003, batch 32, shuffled examples,
10 epochs, reporting each epoch, gradient norm limit 1, no early stopping, no
weight decay or parameter penalty. Use Xavier matrices and zero biases. The
sparse recipe uses activation L1 strength 0.03; values in [0.01, 0.10] are accepted
when they meet the same independently evaluated targets.

Inspect feature families such as words, names, rooms, punctuation roles and
story stages. These are hypotheses, not promised discoveries. Students choose
any feature and build its connected `h_j v_j` contribution using Slice and
Matrix product. This construction is the automatically graded investigation
artifact; semantic descriptions and claimed inspection histories earn no points.
Feature indices belong to the student's encoder and do not transfer to the
supplied checkpoint in the next lesson.

## Assessment and release

Submit `rlhf.json`, `baseline.json`, `reconstruction.json` and `sparse.json`.
The original policy is a zero-point prerequisite artifact; baseline quality and
comparison credit depend on it. The baseline contributes 20 points, dense
reconstruction 25, sparse reconstruction 45, and controlled comparison 10.
Quality gates and numerical tolerances are documented in the handout and
`reference/data.json`. The grader executes bounded allowlisted graph operations,
checks frozen internal targets and decoder ancestry, and recomputes metrics on
canonical data. Missing or malformed artifacts retain independent partial credit.

Current post-training validation is recorded in
[Post-training interpretability](../../../docs/POSTTRAINING-INTERPRETABILITY.md).
Earlier narrative-only experiments in INTERPRETABILITY-EXTENSION.md are
historical evidence and do not validate this final-model release.

Public packaging from `course/`:

```sh
python scripts/assignments/package-interpretability.py
```

Private validation from NeuralCanvas-Coursework:

```sh
node scripts/assignments/interpretability/posttraining-pilot.mjs
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/27-sparse-autoencoder/source -p test_grader.py -v
```

Use `description.html` in Canvas and the single ZIP in `Gradescope/` for a
100-point Programming Assignment. Use Python 3.12/Ubuntu 24.04, at least 4 GB
memory and a 20-minute external timeout. Local extracted-package checks do not
establish hosted Gradescope behavior. Uploading remains a separate instructor
action.
