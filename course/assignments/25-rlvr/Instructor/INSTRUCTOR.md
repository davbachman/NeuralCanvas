# Assignment 25 — Verifiable reward optimization

Students continue the same 5,689-parameter two-block/two-head football transformer from instruction tuning. The browser explicitly captures a fixed reference and uses exact expectation over the nine observed answer tokens. This is a one-step finite-action reward lesson, not a sampled exploration or PPO lesson. Its objective differs from cross-entropy: maximize expected correctness with KL to the captured policy.

## Release and assessment

Use the canonical description.html; upload Gradescope/gradescope-25-rlvr.zip. Students submit before.json (40 points) and rlvr.json (60). Use Ubuntu 22.04, Python 3.12/NumPy 1.26.4, 1 GiB memory and at least 300 seconds aggregate Gradescope time. Run package-rewards.py from course/scripts/assignments to rebuild. Hosted Gradescope publication is separate.

The grader independently evaluates the canonical graph on 96 evenly spaced training rows and all 240 held-out football questions. Full-vocabulary accuracy must reach 65% in the captured reference and final policy. Before must equal its reference exactly; the updated file must share that reference, change parameters and improve E[r] − 0.1 KL by at least 0.00001 on the training benchmark. Graph metadata does not prove a history of training; grades establish saved behavior and consistency only. Regularization and loss penalties remain off. Predictions have no access to targets. Cached scores/epoch counts are ignored.

The 3840/240 canonical split, training vocabulary, QA task, shapes and template come from the preceding instruction-tuning release without resplitting. Expected correctness and candidate accuracy are conditional on the nine answer tokens. Ordinary inference and grading accuracy retain the full vocabulary. Do not describe them as identical metrics. The held-out partition is validation because students use it for selection.

## Recipe and validation

Starting recipe: Adam, rate 0.0003, two reward epochs, batch 32, report each epoch, clip norm1, reference KL 0.1. The app's reward button uses the final parameters and no early stopping. Numerical targets must be checked against the retained pilot evidence in the private sibling repository before issuing.

Private reproduction: `REWARD_SEEDS=137,211,307 node scripts/assignments/rewards/pilot.mjs`. The script records app revision, settings, independent initializations and actual app-engine reports. Private `scripts/assignments/rewards/validate.py` recomputes positive and negative grader cases. No hardware-time or universal improvement claim is made. Numerical results and exact validation status are in `course/docs/REWARD-LEARNING-VALIDATION.md` when the build is complete.

The next lesson deliberately captures the final rlvr.json policy as its new preference reference. Preserve before.json for independent comparisons. The final RLHF checkpoint will later be frozen for the SAE; do not use an unrelated Q-network.
