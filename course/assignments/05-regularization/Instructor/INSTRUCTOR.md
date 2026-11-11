# Assignment 5 — Regularization

Students import their polynomial.json and train the same six-coefficient model
with two fixed L2 strengths. The only submitted files are regularized.json and
strong.json (50 each). Their earlier baseline/polynomial remain comparison
artifacts from the generalization lesson. This is regularization-05-v2 and
changes the previous issued rubric; rebuild/upload its current ZIP.

Moderate lambda .1 uses rate .1 and 1,000 epochs (report50); strong lambda10 uses
rate .05 and 5,000 epochs (report250). Full batches contain 24 training examples.
Select exactly the five feature coefficients in the penalty and exclude b.
The objective is MSE + lambda/2 * sum(w²); reports show data MSE without penalty.

Rubric per file: data5, model/scaling15, correct penalty10, training fit10,
validation10. Objective must lie within .20 of the exact ridge optimum for its
specified lambda. Validation bounds are 16 and 45 mpg². The moderate recipe
produces data MSE 19.23691/13.99684; the strong recipe 43.89897/43.92075. A correct
strong-penalty model receives full credit despite its poorer predictions.

The polynomial need not beat the simpler linear baseline. The intervention
compares the same polynomial architecture at different penalty strengths.
Reproduce with `node scripts/assignments/pilot-foundations-regularization.mjs`.

## Release and assessment

Publish description.html to Canvas and Gradescope/gradescope-05-regularization.zip to a
100-point Gradescope Programming Assignment. Use Ubuntu 22.04/24.04, 1 GB memory,
and at least a 60-second grading timeout. The ZIP includes its bounded runner,
canonical data, and complete source; no private checkout is needed to teach.
Run `python course/scripts/assignments/package-foundations.py` from the app root
to rebuild. The private fixture tests run from NeuralCanvas-Coursework:
`OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/05-regularization/source -p test_grader.py -v`.

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
