# Validation of the 28-assignment course

The current sequence is defined in [sequence.json](../sequence.json). This
release was developed against app parent commit
`16a9b4d7dae77ab92bbbda8d2f8592d1323fb9f5`, with the course and runtime changes
present in the working tree. Historical reports retain their original revisions.

## Teaching and implementation

The opening lessons cover manual regression, gradient descent, multiple features,
generalization, and regularization. Backpropagation follows the first MLP.
Convolution uses the existing Trace SGD implementation and the attributed UCI
8×8 digit dataset. Text models progress through a student-built transformer.

Instruction tuning, verifiable reward optimization, and preference-based reward
optimization adapt that same transformer. The reward objective is an exact
expectation over a small set of answer tokens with a reference-policy KL penalty.
It does not sample rollouts or implement PPO/GRPO. The preference model learns
pairwise comparisons using the frozen reference policy's answer probabilities.
Students enter the human comparisons themselves. Automated pilot comparisons
were generated from a correctness rule and are explicitly identified as
simulated feedback in the [reward validation report](REWARD-LEARNING-VALIDATION.md).

The SAE lesson freezes each student's final preference-trained policy. The
steering lesson supplies a separately validated final-policy/SAE checkpoint;
feature numbers and directions are specific to that checkpoint. It is the sole
authorized trained student asset. Complete graded constructions remain private.
See [post-training interpretability evidence](POSTTRAINING-INTERPRETABILITY.md).

## Independent checks

- The 17 carried-over assignment suites completed 371 tests: 368 passed and
  three existing cases were skipped. Their canonical grading-data files are
  byte-identical to the earlier release.
- The five foundational lessons passed 67 semantic, malformed-input, and
  packaged-submission tests. Actual app-engine pilots cover seven line-model
  artifacts and 19 feature/regularization recipes. Chromium reproduced the
  three prescribed gradient steps and the 50-epoch final model.
- Backpropagation and convolution passed 29 tests. Constructed derivatives
  agree with app backward gradients; six MLP/CNN training runs cover three
  initializations. Full browser workflows produced submissions earning
  100/100 in both extracted graders.
- Both reward assignments' final canonical submissions earned 100/100 in their
  extracted packages. Three native initializations, a complete app-engine
  training sequence, independent objective/gradient parity, and 14 adversarial
  cases were checked. Chromium exercised the transformer reward workflow,
  preference entry, save/import, evaluation, and clearing reward state.
- All six dense/sparse SAE runs used the actual app engine for ten epochs on
  three post-training bases. Sparse validation MSE ranged from 0.1194 to 0.1509;
  activity ranged from 13.40 to 14.91 of 32 features per token. The recalibrated
  activity bound is 16, with the controlled dense-to-sparse comparison retained.
  The two interpretability graders passed 31 tests, including packaged full
  credit, partial isolation, final-policy lineage, and causal intervention checks.
  Chromium imported the supplied checkpoint, scanned feature contexts, built one
  discovery branch, inspected prepared interventions, and saved/reimported all
  four models. Those browser-saved submissions earned 100/100 in the current
  extracted steering package.
- All 28 release packages passed source, manifest, dataset, and invalid-submission
  checks; all 29 app dataset references validated. Chromium checked every student
  handout at mobile and desktop widths and followed all 21 legacy redirects.
- The production build and ESLint passed. Every failure from the broad app test
  run was resolved by isolated reruns after reducing CPU contention. The initial
  run and both retries are retained; this is not a claim of a single clean full
  suite run. Existing skipped tests remain skipped.
  A final focused run passed all 71 reward, persistence, tensor-training, SAE,
  and reward-panel tests.

Private raw evidence, trained models, browser screenshots, and regression tests
are retained in the sibling coursework repository under `pilots/foundations`,
`pilots/neural-foundations`, `pilots/rewards`,
`pilots/interpretability/posttraining`, and `pilots/course28`.

## Reproduce public release checks

From the app repository, with the course's Python dependencies installed:

```sh
python course/scripts/package-course.py
python course/scripts/sync-handouts.py
python course/scripts/validate-course.py
npm run build
npm run lint
```

The public validator checks all 28 package/source pairs, release hashes,
canonical dataset hashes, trained-artifact boundaries, checkpoint deployment
parity, and missing/malformed submissions. The private browser course check
renders every handout at 375 and 1280 pixels and follows all legacy redirects.

These are local app-engine, Chromium, and extracted-package checks. They do not
certify hosted Gradescope execution or constitute uploads to Canvas or
Gradescope. Browser timings on this shared runner are not laptop benchmarks.
