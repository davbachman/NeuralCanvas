# Assignment 2 — Gradient descent

Students reuse the same scalar regression graph and directly derive the two MSE
parameter derivatives. Formal backpropagation is deferred until hidden layers.
Every one-update experiment restarts from zero. Grade exact resulting parameters
rather than stored learning rates, epoch counts or screenshots.

Artifacts: before.json (10), small.json (20), useful.json (30), excessive.json
(20), trained.json (20). Each gives 40% construction and 60% result credit.
The three full-batch updates have rates .01, .1 and 1 and use all 15 training
observations. The selected rates produce training MSE 14.706539, 1.771395 and
580.832804 respectively from the common 16.938853 starting loss. The deliberately
poor large-rate result earns credit when its exact update is correct. The grader
independently derives both new coefficients from canonical training rows.

Fifty full-batch updates at .1 produce training/validation MSE .016464/.033000,
within .03/.04. This is deterministic from zero; initialization seeds do not
apply. Avoid the single-example training shortcut: one Run epochs pass with
Examples per update=15 is the requested full-batch step. Reproduce with
`node scripts/assignments/pilot-foundations.mjs` in the private checkout.

## Release and assessment

Publish description.html to Canvas and Gradescope/gradescope-02-gradient-descent.zip to a
100-point Gradescope Programming Assignment. Use Ubuntu 22.04/24.04, 1 GB memory,
and at least a 60-second grading timeout. The ZIP includes its bounded runner,
canonical data, and complete source; no private checkout is needed to teach.
Run `python course/scripts/assignments/package-foundations.py` from the app root
to rebuild. The private fixture tests run from NeuralCanvas-Coursework:
`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/02-gradient-descent/source -p test_grader.py -v`.

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
