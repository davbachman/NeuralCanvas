# Assignment 26 — Human comparisons, learned reward, policy optimization

Students import rlvr.json, collect their own pairwise comparisons in Neural Canvas, fit a small reward model, and adapt the same transformer. The handoff is rlhf.json, which the next assignment freezes before fitting an SAE. The guided steering checkpoint must be trained from a matching final RLHF transformer, not the former narrative-only checkpoint.

## Preference provenance and limits

The student workflow elicits genuine human choices: read a training context, choose a supported answer and a distinct false answer, and record the comparison. No precollected human dataset is claimed or supplied. In this bounded task the preference criterion is explicit factual correctness, so the autograder can check judgments objectively. It cannot establish that a person entered them. Open-ended stylistic or ethical preferences are outside this exercise.

Private reference pilots generate verifier-derived comparisons to validate the mechanics reproducibly. Those are explicitly **simulated preference fixtures**, not evidence from collected human judgments. They do not establish reliability for every student's preference selection. The canonical examples include diverse targets and contexts; the handout requires 24 pairs on at least 12 contexts and three preferred answers.

The reward representation is the frozen reference model's nine candidate probabilities. A learned 9×9 weight matrix and 9 biases produce one raw score per action; pairwise logistic loss trains score differences with L2 coefficient 0.001 on weights, 200 steps at rate 0.5. Sigmoid bounds scores into policy rewards. Both the features and reward weights stay fixed while the transformer changes. The exact correctness checker is diagnostic during this policy stage; it supplies no gradient to the learned-reward objective.

## Release and grading

Upload Gradescope/gradescope-26-rlhf.zip and issue canonical description.html. Require rlvr.json (20 points), preferences.json (40 points), and rlhf.json (40). The captured preference reference must exactly equal the submitted RLVR predecessor. Use Python 3.12/NumPy 1.26.4, 1 GiB memory, aggregate timeout 300 seconds. Rebuild using package-rewards.py. Canvas/Gradescope publishing is a separate instructor action.

The grader validates canonical architecture/data, finite reward parameters, training-only distinct comparisons, stated preference criterion, and consistent reference/reward tensors across files. It recomputes pairwise logistic loss ≤ 0.68 and comparison agreement ≥ 60%. The preference checkpoint must preserve the captured policy. The final policy must change parameters, increase E[learned reward]−0.1KL by ≥ 0.00001 on 96 deterministic training rows and retain ≥ 65% full-vocabulary validation accuracy on 240 questions. An adequately trained captured reference is also required. Stored histories, claimed human provenance and cached scores earn no credit.

The basic recipe matches RLVR: Adam 0.0003, batch 32, two reward epochs, report every epoch, clip 1. Human selections may require lower learning rates or shorter runs. Predicted reward can rise while correctness falls; the lesson makes these quantities separately visible and does not promise simultaneous improvement.

Private reproduction and actual evidence: scripts/assignments/rewards/pilot.mjs, scripts/assignments/rewards/validate.py, and pilots/rewards/. Public summary: course/docs/REWARD-LEARNING-VALIDATION.md. Actual browser workflows and engine pilots must be distinguished from hosted Gradescope validation.
