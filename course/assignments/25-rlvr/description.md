<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 25: Improve an answer policy with verifiable rewards

Your instruction-tuned transformer learned from demonstrations. Now adapt the same transformer using a reward for a correct answer. Keep its architecture and dataset; change the objective used to update its parameters.

**Submit:** `before.json` and `rlvr.json`. Total: 100 points. Keep both files for later comparisons.

## Prepare the reference policy

Import your trained `after.json` from *Instruction tuning*. Keep the original two-block, two-head transformer, vocabulary, and answer-position logits. Do not randomize it.

Verify **Dataset → Details → Browse datasets… → Text and sequences → Football stories → Answer-token fine-tuning**. Keep the embedded 25-token vocabulary, 23-token contexts, and original training/validation split. The app calls the held-out partition a test set; here it is validation because you use it during model selection.

Open **Train → Reward learning**. Select **Reward source for new reference → Verifiable answer correctness**, then **Capture current model as reward reference**. Set **Reference KL strength** to **0.1**. Click **Evaluate reward policy** and save `before.json`. Capturing stores a frozen copy of the current transformer parameters; it does not replace the trainable model.

## Understand the new objective

The candidate actions are the nine answer tokens observed in training. The policy renormalizes their logits into candidate probabilities. A checker gives reward 1 to the correct answer and 0 to the others. The app minimizes:

```text
J = −Σa p(a | context) r(context, a) + β KL(p || p_reference)
```

The first term favors rewarding answers; the second discourages large changes from the frozen reference. This is exact expected-reward optimization over a small, one-step action space. It enumerates all candidates rather than sampling actions, and it is not PPO or a delayed-reward problem. Contrast this objective with the cross-entropy objective used for instruction tuning.

## Optimize the same transformer

First select **Training execution → Tensor engine · minibatches** to reveal the controls. Follow the Train panel in this order:

| Train setting | Value |
| --- | --- |
| Epochs per run | 2 |
| Report loss every | 1 |
| Examples per update | 32 |
| Reshuffle training examples each epoch | Checked |
| Exact learning rate | 0.0003 |
| Training execution | Tensor engine · minibatches |
| Backend | Tensor CPU |
| Optimizer | Adam |
| Gradient norm limit (0 = off) | 1 |
| Early stopping patience (0 = off) | 0 |

Click **Optimize policy with rewards**. The ordinary **Run epochs** button still trains the canvas cross-entropy Loss; use the reward button for this investigation. Keep the captured reference fixed. Inspect the reward objective, expected correctness, candidate accuracy, and reference KL before and after the updates.

Use **Test → Run inference** to evaluate the full vocabulary as well. Candidate accuracy restricts the answer set, while ordinary Test allows every vocabulary token; these are different measurements. Save `rlvr.json`.

If the final policy falls below the validation quality requirement, return to your saved baseline and try a smaller learning rate or a shorter run. Keep the same captured reference. Training longer is optional and does not guarantee improvement. Neither training history nor browser speed is graded.

## Grading

| File | Points | Requirements |
| --- | --- | --- |
| before.json | 40 | Canonical data and transformer; verifiable reward mode; policy parameters exactly equal the captured reference; reference full-vocabulary validation accuracy ≥ 65%. |
| rlvr.json | 60 | Same captured reference; actual transformer parameter changes; recomputed training objective gain ≥ 0.00001; full-vocabulary validation accuracy ≥ 65%. |

The grader recomputes the objective on 96 deterministically spaced training examples and evaluates all 240 validation questions. It does not trust stored scores or epoch counters. It awards separate credit for valid saved graphs, predecessor quality, checkpoint continuity, and the optimized policy. No written answers are required.

**Handoff:** preserve `rlvr.json` for *Learning from human preferences*.
