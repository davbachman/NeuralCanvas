<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 26: Learn a reward from human preferences

An exact answer checker is available in the football world. In many applications, preferences must instead be elicited from people and approximated by a model. You will supply human comparisons, fit a small reward model, and use its predictions to further adapt your transformer.

**Submit:** `rlvr.json`, `preferences.json`, and `rlhf.json`. Total: 100 points. The final `rlhf.json` is the transformer checkpoint for the sparse-autoencoder assignment.

## Start from the reward-trained transformer

Import your saved `rlvr.json`. Preserve that original file unchanged for submission and keep the architecture and learned parameters. In **Train → Reward learning**, select **Reward source for new reference → Human preference comparisons** and click **Capture current model as reward reference**. This deliberately establishes the reward-trained transformer as the new fixed reference and clears the previous reward configuration. Keep **Reference KL strength** at **0.1**.

Keep **Dataset → Details → Browse datasets… → Text and sequences → Football stories → Answer-token fine-tuning**, its 23-token inputs, 25-token vocabulary, and original split. Use only training contexts for preference fitting.

## Supply and learn human comparisons

Choose a **Training context** in the reward panel. Read the events and question. Select a **Preferred answer** supported by the story and a distinct, false **Rejected answer**, then click **Record human preference**. The judgments are yours; the app does not generate a preference dataset for you.

Record at least **24 distinct comparisons across 12 or more training contexts**, including at least **three different preferred answer tokens**. On a context you may compare the correct answer with two different rejected answers. Include differing event orders and question types. For a more reliable first fit, spread your contexts across the training list and include all nine possible preferred answers. A useful starting budget is two comparisons on each of 26 varied contexts (52 comparisons). Duplicate comparisons do not count. The specified preference criterion is factual correctness; the grader can check this criterion, although a saved file cannot establish who entered a comparison.

Click **Fit preference reward model**. The app performs 200 deterministic gradient steps at rate 0.5 on pairwise logistic loss with a small weight penalty. The reward model is a linear scoring model over the *frozen reference transformer's candidate probabilities*, followed by sigmoid for bounded rewards. Its representation remains fixed during subsequent policy optimization. This small model deliberately makes approximation errors visible; it is not a general language reward model.

Require comparison loss ≤ **0.68** and agreement with recorded preferences ≥ **60%**. If the fit is poor, check contradictory judgments and broaden the comparisons. Fit again after any edits. Save `preferences.json` with the transformer parameters still unchanged from the newly captured reference.

## Optimize using the learned reward

Use **Tensor engine · minibatches**, **Adam**, **Exact learning rate → 0.0003**, **32** examples per update, **2** epochs per run, reporting every **1** epoch, **Backend → Tensor CPU**, and **Gradient norm limit (0 = off) → 1**. Click **Optimize policy with rewards**. Keep the comparisons, fitted reward model, and captured reference fixed throughout this stage.

The policy now maximizes expected *learned* reward while remaining close to its reference. The exact correctness checker supplies a separate diagnostic; it does not supply this policy's training reward. Compare expected learned reward with expected correctness, then run ordinary **Test → Run inference** for full-vocabulary validation accuracy. Increasing predicted reward need not increase correctness.

Save `rlhf.json`. If necessary, return to `preferences.json` and use a smaller rate or shorter policy run to preserve QA quality. Additional training is optional. This is a bounded RLHF exercise with human-entered preferences and exact finite-action policy optimization; it does not implement PPO.

## Grading and final checkpoint

| File | Points | Requirements |
| --- | --- | --- |
| rlvr.json | 20 | The preceding completed verifiable-reward policy, with independently checked reward improvement and QA quality. |
| preferences.json | 40 | Valid canonical transformer and human-comparison records; captured reference exactly matches submitted rlvr.json; original reference parameters unchanged; comparison loss ≤ 0.68 and agreement ≥ 60%; reference validation accuracy ≥ 65%. |
| rlhf.json | 40 | Identical reference, comparisons, and fitted reward model; changed transformer parameters; training objective gain ≥ 0.00001 on the fixed 96-example benchmark; full-vocabulary accuracy ≥ 65% on all 240 validation questions. |

Every score is independently recomputed from submitted operations, parameters, and comparison records. Cached reports and claimed training history are ignored. No written responses are submitted.

Preserve `rlhf.json` unchanged. The next assignment freezes this exact transformer and trains a sparse autoencoder on its activations. Any later policy training would change the activation distribution and require renewed SAE validation.
