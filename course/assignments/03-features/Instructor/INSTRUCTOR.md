# Assignment 3 — Multiple features and feature scaling

This replaces the former three-model MPG lesson with two linear models, each
worth 50 points: one_feature.json and two_features.json. Polynomial flexibility
belongs to the following lesson. Students construct fixed input preprocessing
and add an independent year coefficient. Scaling is assessed through the saved
model; optional raw-input experiments are explicitly unsubmitted.

Use all 398 Auto MPG observations: 318 training, 80 validation, original
measurements and deterministic supplied split. Standardize weight and year on
training rows, retaining mpg target units. Both recipes use full-batch Trace SGD,
rate .05, 318 examples/update and 1,000 epochs, reporting every 50.

Rubric per artifact: data 5, form/scaling 25, training MSE 10, validation MSE 10.
One-feature limits are 22/16 mpg²; two-feature limits 14/10. The known app pilots
produce 20.248399/13.015979 and 12.757442/7.555346. The restricted symbolic
polynomial evaluator verifies independent coefficients and absence of target
leakage. The legacy acceptance of equivalent fixed affine Arithmetic scaling
is retained; the recommended construction uses Standardize features fitted on
training rows. The rubric does not infer a Fit-button history from a graph.

Reproduce with `node scripts/assignments/pilot-foundations-features.mjs`.
The new release is features-03-v1; the former mpg-01-v3 package remains historical.

## Release and assessment

Publish description.html to Canvas and Gradescope/gradescope-03-features.zip to a
100-point Gradescope Programming Assignment. Use Ubuntu 22.04/24.04, 1 GB memory,
and at least a 60-second grading timeout. The ZIP includes its bounded runner,
canonical data, and complete source; no private checkout is needed to teach.
Run `python course/scripts/assignments/package-foundations.py` from the app root
to rebuild. The private fixture tests run from NeuralCanvas-Coursework:
`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/03-features/source -p test_grader.py -v`.

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
