<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 18: Give each block two attention patterns

**Your starting point and handoff**

| Step | What to use |
| --- | --- |
| Start with | A copy of your saved **stacked.json** |
| Modify | Replace each block’s single attention head with two independent heads while keeping the total feature width at 16. Randomize the modified architecture. |
| Save and keep | **multihead.json**; keep this trained generator for the next assignment. |

A single attention head produces one probability distribution over positions per query. Give each transformer block **two independently learned heads**, each producing an **8-feature** result. Keep the total width at 16 and retain both successive blocks.

Use the narrative configuration selected below in your Dataset block. It keeps the same football narratives, **25-token vocabulary**, **32-token input windows** and training/held-out split. The targets contain one next-token ID for each input position. Keep the supplied data and split unchanged. Held-out data is validation during development.

Continue using model width **16** and MLP hidden width **32** unless a step explicitly changes a head’s width.

Select your Dataset block and open **Details → Browse datasets…**. Choose **Text and sequences** → **Football stories**. Choose the **Narrative generation** preset. Set **Training targets** to **Every input position**. Click **Use this dataset**. Keep the supplied data and split unchanged.

**Prepared dataset properties**

| Dataset property | Supplied value |
| --- | --- |
| Vocabulary entries | 25, including unknown and padding |
| Input token IDs | 32 |
| Position IDs | 32 |
| Next-token target IDs | 32 |
| Training windows | 1534 |
| Held-out windows | 765 |

## Two patterns within each block

Start from a copy of your two-block model. Replace each block’s Q, K and V projections with independent projections for each head. Determine their shapes from the 16-feature input and 8-feature head width. Each head computes its own QK<sup>T</sup>, divides by **√8** (use `x1 / 2.8284271247461903`), applies a causal mask and rowwise Softmax, and combines its own V.

Concatenate the two results across features, in head order, before the block’s existing output projection. Keep the residual routes, MLPs, normalizations and vocabulary projection. Work out the concatenation axis. Do this in both blocks, with separate Params throughout.

Randomize and train the completed model. Save **multihead.json**. The grader verifies both heads in both blocks and the fixed total width. Parameter count stays the same as the one-head two-block architecture, while attention-score storage and execution cost can increase.

## Parameter-count hint

Your completed models should have the following counts. Count individual learned numbers, including embedding tables and biases, and count each shared Param only once.

- **multihead.json:** **5,689** trainable parameters.

## Train and save

First set **Training execution** to **Tensor engine · minibatches** to show the Tensor controls. Then follow the Train tab from top to bottom:

**Training settings — Train tab**

| Control | Value |
| --- | --- |
| Epochs per run | 10 |
| Report loss every | 1 |
| Examples per update | 32 |
| Reshuffle training examples each epoch | Checked |
| Exact learning rate | 0.003 |
| Training execution | Tensor engine · minibatches |
| Backend | Tensor CPU |
| Optimizer | Adam |
| Gradient norm limit (0 = off) | 1 |
| Early stopping patience (0 = off) | 0 |

**Loss block → Details → Parameter regularization**

| Control | Value |
| --- | --- |
| Penalty | None |

Tensor CPU is the validated backend. Auto or WebGL may be used; speed depends on the computer and browser. Use the same backend for timed comparisons.

**Param → Initialize choices**

| Param block | Initialize |
| --- | --- |
| Word and position tables | uniform |
| Other matrices | Xavier · matrices |
| Bias vectors | zeros |
| Layer norm scales | ones |
| Layer norm offsets | zeros |

Apply changed shapes, check the saved Initialize choices, then use **Train → Randomize parameters**. Remove unused Params after an architecture change. Each changed architecture starts fresh; evaluation copies retain their trained values.

## Keep your pretrained generator

This next-token training is **pretraining** for the next assignment’s question-answer task. Keep **multihead.json** after the recommended ten-epoch run. You may continue narrative training in additional **5-epoch** runs with the same settings, saving a checkpoint first. Choose a checkpoint meeting training loss **≤ 0.60** and held-out loss **≤ 0.65**. More pretraining does not guarantee better question answering. Complete optional narrative training before preparing the QA baseline.

## Optional: generate a continuation

In **Dataset → Details → Continue the text**, paste this prompt and try **Most likely** or **Sample → Generate 40 tokens**:

> nora entered bedroom . liam entered kitchen . nora took football . nora entered hall . nora dropped football . now the football is in the hall . end nora entered bedroom . liam entered kitchen . nora took football . nora dropped football . nora entered hall . now

The prompt is longer than 32 tokens, so the app uses its last 32. Short prompts need padding and may behave differently. The temperature and sampling controls introduced earlier apply to every generator in this sequence. Use the same prompt to compare Most likely with Sample at temperatures 0.5, 1 and 2, keeping Top-k at 25. Sampling changes token selection without training. A fluent continuation can still misstate a person’s location, the football’s location or who holds it. This inspection is optional; the submitted model itself is the graded artifact.

## Submit and grading

Points below are dataset / model form / training loss / held-out loss. Loss limits are upper bounds. Performance credit requires the assigned architecture. The grader recomputes results from saved parameters; cached scores and declared epoch counts are ignored.

| File | Points | Training loss ≤ | Held-out loss ≤ |
| --- | --- | --- | --- |
| multihead.json | 10 / 50 / 20 / 20 | 0.6 | 0.65 |

Accuracy, parameter counts, common final-position loss and generated continuations are informational. No required written answers, screenshots or unsubmitted investigations.

Assignment 18-multiple-heads.
