# Assignment 1 — Predictions, parameters, and loss

Students build b + w*x and manually select coefficients. A zero model is retained
as the reference point; a second artifact must fit both partitions within 0.20
MSE. Neither an exact optimum nor gradient descent is needed. The handout explains
residuals, squared errors and parameter effects before introducing optimization.
Use the built-in Line regression (1D), unchanged default split (15 training, 5
validation). Its deterministic synthetic values are copied into grading data.

Artifacts: zero.json (30), fitted.json (70). Each gives 40% construction credit
and 60% parameter-result credit. Two independent scalar coefficients and a
proper MSE target connection are required. Linear arithmetic equivalents are
accepted. Scalar constants cannot substitute for adjustable coefficients.

The exact zero artifact has training MSE 16.938853 and validation MSE 11.612900.
A manually selected pilot line gave 0.018053/0.021700; the deliberately looser
0.20/0.20 requirements permit an attainable visual fit without revealing fitted
coefficients in public materials. Reproduce with
`node scripts/assignments/pilot-foundations.mjs` in the private checkout.

## Release and assessment

Publish description.html to Canvas and Gradescope/gradescope-01-predictions.zip to a
100-point Gradescope Programming Assignment. Use Ubuntu 22.04/24.04, 1 GB memory,
and at least a 60-second grading timeout. The ZIP includes its bounded runner,
canonical data, and complete source; no private checkout is needed to teach.
Run `python course/scripts/assignments/package-foundations.py` from the app root
to rebuild. The private fixture tests run from NeuralCanvas-Coursework:
`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/01-predictions/source -p test_grader.py -v`.

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
