# Assignment 9: Backpropagation through hidden and shared parameters

Version: backpropagation-09-v1. Two student-built graphs, `chain.json` and `shared.json`, each earn 50 points. This lesson follows construction of an MLP and precedes nonlinear classification. Students already understand gradient descent; this lesson explains how the gradient is calculated through hidden computation and accumulated at a shared parameter.

Read the public course guide and assignment-design principles before editing. Canonical HTML is `../description.html`; generated Markdown and deployed pages come from `course/scripts/sync-handouts.py`. No written responses, supplied trained models, external quizzes, or screenshots are required. Inspection supports the independently graded derivative constructions.

## Construction and purpose

Use the built-in Line regression (1D) source through Regression → Simple regression examples. Keep one example per forward/backward calculation. The four scalar Params are w=0.8, b=−0.2, v=1.1, c=0.1. The forward model is one tanh hidden unit and a linear output. Students build the derivative of single-example MSE with respect to w from existing values and local derivatives. Tanh supplies a smooth, nonconstant derivative and makes numerical gradient comparison unambiguous; this exercise does not claim to train a useful predictor of this dataset.

The second model shares the original w between two tanh units, with b₁=−0.2, b₂=0.4, v₁=1.1, v₂=−0.7, c=0.1. Students build both path contributions and their total. Equal-valued independent Params do not count as sharing. The model uses six Params. The contribution branches are detached from the loss's prediction path and do not alter the training objective. They must remain dynamically connected to Params, inputs, and target; numeric copies of displayed gradients are insufficient.

Use Run forward for the branch values. Trace engine Step can expose the backward phase; stop before updating, or restore prescribed values before submission. There is no accuracy or convergence target. The grader uses fixed model values to identify parameter roles, checks the forward computation across inputs and parameter perturbations, then verifies the derivative branches under those perturbations. It does not grade claims about inspection or training history. Parameter derivatives include all uses of the parameter. Target values may appear in the derivative branches but cannot appear in predictions.

## Grading and validation

Chain: 15 forward model +35 derivative. Shared: 15 forward model +10 first path +10 second path +15 accumulated gradient. Names, layout, grouped display, and algebraically equivalent scalar Arithmetic constructions are accepted. The test varies parameters and target values independently of student files, preventing hardcoded coefficients or cached numerical derivatives from passing. Grader execution uses a bounded AST evaluator, never Python eval or submitted code. Each saved JSON is limited to 8 MiB, 100 blocks, 200 wires, and small finite Params. Each part is a subprocess bounded to 1 GiB, 30 CPU seconds, and a 40-second wall timeout.

The app's actual forward/backward engine and project parser were used at parent app commit `16a9b4d7dae77ab92bbbda8d2f8592d1323fb9f5`, with concurrent revised-course/app edits present. Across all 20 canonical line examples, both constructed derivatives matched the app's w gradient with maximum absolute error 0. Private evidence is `pilots/neural-foundations/backprop.json`; scripts, complete reference constructions and tests remain private. Those exact fixed scalar examples do not require stochastic convergence pilots.

Fourteen private tests cover full credit, partial/missing/malformed packaged submissions, changed splits, labels, cached values, cycles, malicious expressions, fixed derivative values, copied coefficients, missing path accumulation, unshared weights, and target leakage. Browser validation imports these independent graphs, runs forward, and saves projects; it does not demonstrate full construction by mouse. Local package/browser checks do not certify hosted Gradescope or the deployed student site.

## Publishing and reproduction

Upload `Gradescope/gradescope-09-backpropagation.zip` as a 100-point Gradescope Programming Assignment. Use Ubuntu 22.04/24.04 with at least 1 GiB memory and a timeout of 120 seconds. Setup installs pinned NumPy 1.26.4 in `/opt/neural-foundations-grader`. Paste the canonical HTML into Canvas. Test saved files on hosted Gradescope before issuing the assignment.

From the app root: `python course/scripts/assignments/package-neural-foundations.py` and `python course/scripts/sync-handouts.py`. From the private coursework root:

```sh
BACKPROP_ONLY=1 node scripts/assignments/neural-foundations/pilot.mjs
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/09-backpropagation/source -p test_grader.py -v
node scripts/assignments/neural-foundations/browser.cjs
```

The pilot overwrites private reference constructions only. No data download is required.
