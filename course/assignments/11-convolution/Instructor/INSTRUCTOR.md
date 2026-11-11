# Assignment 11: Image MLP and convolutional classifier

Version: convolution-11-v1. Students build two models on the same 400 training /100 validation UCI 8×8 digit images. `mlp.json` and `cnn.json` each earn 50 points. The objective is locality and shared parameters, with the preceding backpropagation lesson explaining gradient accumulation across spatial uses of a filter. Neither architecture must outperform the other.

Canonical HTML is `../description.html`; update generated Markdown/student pages with `course/scripts/sync-handouts.py`. All construction occurs in Neural Canvas. No trained starter model, written explanation, screenshot, or external coding exercise is required. Trained references and pilots remain in the private repository.

## Exact architectures and recipe

Both Dataset outputs already contain normalized pixels in [0,1]. Do not divide by 16 again. The MLP reshapes [8,8,1] to [1,64], uses W₁ [64,16], b₁ [16], ReLU, W₂ [16,10], b₂ [10], then raw-logit cross-entropy against the [1] digit target. It has 1,210 parameters.

The CNN uses four valid stride-one filters W [4,3,3,1], bias [4], ReLU on [6,6,4], 2×2 stride-two average pooling to [3,3,4], reshape to [1,36], and a linear classifier W [36,10], bias [10]. It has 410 parameters. Optional Softmax is an inspection branch, never the cross-entropy input. Students determine shapes themselves; the handout supplies architecture constraints rather than every wiring step.

Use Trace engine · SGD, learning rate 0.03, 30 epochs, report every 5, one example per update, and reshuffle each epoch. Convolution and Average pooling are supported by the trace engine, not the tensor engine. First weight tensor: Xavier seed 137; classifier matrix: Xavier seed 139; biases: zeros. Additional pilots use seeds 211/213 and 307/309. All Params remain trainable and unconstrained; no dropout or regularization. Each construction uses four independent Params.

## Data and grading

The app includes a deterministic educational split of 500 UCI Optical Recognition of Handwritten Digits records: 50 per class, of which 40 train and 10 validate. Creators E. Alpaydin and C. Kaynak; DOI 10.24432/C50P49; CC BY 4.0. This subset comes from the distributed UCI test partition via its scikit-learn mirror and does not reproduce the original writer-independent split. See app `docs/CNN-DATA.md` and committed `src/learning/digits.json`. No download/regeneration is needed for grading or testing. `reference/data.json` embeds those exact pixels, labels, and split. The held-out set is validation because the assignment uses it for model comparison.

Each file: 15 architecture  + 15 training cross-entropy ≤0.40  + 20 validation cross-entropy ≤0.60 and accuracy ≥85%. Architecture credit is required for all performance credit. Bounds have margin relative to the three-seed pilots; no relative-improvement condition rewards damaging a baseline. Cached scores, epoch counts, names, and layouts are ignored. A bounded NumPy interpreter independently recomputes all predictions, verifies exact semantic model shape, and rejects target leakage, extra learned paths, wrong filters/activations, and altered splits. Renamed blocks, connected aliases, and commuted bias sums are accepted.

Each file is limited to 8 MiB, 100 blocks, 200 wires and 4,096 learned values. Each part runs in a subprocess bounded to 1 GiB, 30 CPU seconds and 40 seconds wall time. Arithmetic uses a restricted AST, never eval or submitted code. Fifteen private tests include full-credit references for all seeds, undertrained models, cached metric changes, labels, missing/malformed/partial packaged submissions, extra Params, wrong activation/filter shape, altered splits, cycles, nonfinite numbers, and target leakage.

## Pilot evidence

Actual app Trace SGD execution, dataset evaluation and project parser were used at parent app revision `16a9b4d7dae77ab92bbbda8d2f8592d1323fb9f5`, with concurrent revised-course and reward-training work present. Complete trained models and raw evidence are private in `assignments/11-convolution/reference/models/` and `pilots/neural-foundations/`.

| Model | Seed | Train cross-entropy | Validation cross-entropy | Validation accuracy |
|---|---|---|---|---|
| MLP |137|0.00827|0.05550|97%|
| MLP |211|0.01035|0.13619|97%|
| MLP |307|0.01002|0.12040|95%|
| CNN |137|0.08111|0.26028|92%|
| CNN |211|0.07555|0.24138|91%|
| CNN |307|0.05977|0.26602|92%|

All six runs meet the fixed thresholds. The MLP is more accurate here; the CNN uses fewer parameters and shared local features. Engine runs took about 26–61 seconds per model in this environment, including concurrent pilots; these timings do not predict student hardware speed. Browser validation imports the initialized graphs, runs the complete 30-epoch recipe through the UI, and saves/regrades both models. It does not demonstrate construction of every block by mouse. Hosted Gradescope, deployed-site behavior and other browser engines remain unverified.

## Publishing and reproduction

Upload `Gradescope/gradescope-11-convolution.zip` to a 100-point Programming Assignment using Ubuntu 22.04/24.04, at least 1 GiB RAM and 120 seconds grading time. Setup pins NumPy 1.26.4. Paste `description.html` into Canvas. Test your own saved models on hosted Gradescope before issuing.

From the app root, package with `python course/scripts/assignments/package-neural-foundations.py`, then synchronize handouts. From the private coursework root:

```sh
CNN_ONLY=1 node scripts/assignments/neural-foundations/pilot.mjs
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s assignments/11-convolution/source -p test_grader.py -v
node scripts/assignments/neural-foundations/browser.cjs
```

The pilot overwrites private models/evidence and supports PILOT_SEED, PILOT_RATE and PILOT_EPOCHS for experiments. Defaults reproduce the handout. BUILD_ONLY writes initial private graphs without training; it does not replace the trained evidence.
