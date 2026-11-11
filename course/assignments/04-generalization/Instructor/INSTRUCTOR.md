# Assignment 4 — Model flexibility and generalization

Students compare baseline.json and polynomial.json (50 each) on Auto MPG's
24-training/80-validation Small training sample. The linear and degree-five
models use separately standardized powers of standardized weight. Model year
is present in data but unused. All fitted transformations use training rows.

This lesson isolates generalization before introducing a penalty. Full-batch
Trace SGD uses rate .1; linear training runs 1,000 epochs (report50), polynomial
10,000 (report500). Both use no penalty. The correlated polynomial features
converge slowly; do not replace the finite-run errors with exact optimum errors.

Rubric per artifact: data5, model/scaling25, training fit10, validation10. MSE
must be within .20 of the independent least-squares optimum. Validation bounds
are 14 and 60 mpg², respectively; the polynomial is intentionally allowed to
overfit. Existing three-seed app evidence gives linear train/validation
22.01439/13.04985 and polynomial 17.23190/28.59490. The converged least-squares
polynomial has different validation error; the task grades the stated tolerance.

Preserve both saved models for the regularization lesson, especially
polynomial.json. Reproduce with
`node scripts/assignments/pilot-foundations-regularization.mjs`.

## Release and assessment

Publish description.html to Canvas and Gradescope/gradescope-04-generalization.zip to a
100-point Gradescope Programming Assignment. Use Ubuntu 22.04/24.04, 1 GB memory,
and at least a 60-second grading timeout. The ZIP includes its bounded runner,
canonical data, and complete source; no private checkout is needed to teach.
Run `python course/scripts/assignments/package-foundations.py` from the app root
to rebuild. The private fixture tests run from NeuralCanvas-Coursework:
`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/04-generalization/source -p test_grader.py -v`.

The runner grades each named file in a separate process with 15-second wall,
12-second CPU and 1 GB address-space limits. Wrong or absent files retain other
parts' credit. Graphs are data; submitted code is never executed. All assessed
investigations culminate in saved model JSON. Explanations and screenshots are
not required. Reported losses, block names, cached tensors and epoch counts are
not evidence of successful training. A valid final artifact cannot prove the
student's personal process.

The local tests cover full-credit fixtures, partial packaged submissions, absent
and malformed files, target leakage, changed datasets, duplicate IDs, incorrect
coefficients, penalty/update requirements, and forged cached results. Hosted
Gradescope is not validated by these checks. Browser evidence, when available,
is distinguished from engine calculations in the course validation report.

## Provenance and reproduction

The source data and original split are unchanged. Both visible partitions are
used for model comparison; call the second partition validation. Never fit
preprocessing on validation rows. All reference models, model-producing pilots,
and solution-dependent tests live in the private sibling checkout.

The revised-course pilots use app parent commit `16a9b4d` with ongoing course/app
changes. Exact run metadata and measured errors are recorded under private
`pilots/foundations/`; the engine uses the actual app graph evaluator, training
updates and saved-project serializer. These are local feasibility measurements,
not promises of student hardware speed. Public reports contain no fitted
solution coefficients. Earlier issued releases remain in Git history and the
private historical archives.
