<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 28: Test features through residual interventions

When does a feature activate, and which predictions does its direction support? Investigate both questions in a common pretrained transformer and sparse autoencoder, then build interventions that test its effect on next-word probabilities.

Download [**feature-steering.json**](https://davbachman.github.io/NeuralCanvas/models/feature-steering.json) and use **File → Import**. It contains a transformer after instruction tuning, verifiable-reward adaptation and preference-reward adaptation, evaluated on narrative contexts, plus a newly trained 32-feature SAE, with every Param frozen and no discovery or steering branches. Use this exact checkpoint for every submitted file; keep your own sparse.json for comparison. Feature numbers and decoder directions are specific to these weights: do not transplant the supplied SAE onto your own independently trained transformer. Do not train, randomize or change learned values.

The checkpoint embeds the canonical data. Verify its configuration in **Dataset → Details → Browse datasets…**: **Text and sequences** → **Football stories** → **Narrative generation**, with **Training targets → Every input position**. Preserve its 25-word vocabulary, 32-token windows, and 1,534 training / 765 validation windows.

## Discover what activates feature 26

Investigate **zero-based feature index 26** of the encoder ReLU output h, labelled **sae features**. This index has meaning only in this checkpoint. Run **Test → Run inference** on training data, select h, and open **Data → Inspect feature contexts**. Set **Feature index** to 26, choose **Examples to inspect**, then **Find high activations**. Brackets mark the selected token. Compare surrounding words as well as that token.

Look for a recurring word at highly activating positions. Check smaller or zero activations in **Data → Dataset row**. Develop a hypothesis on training examples, then check additional examples. For the numerical comparison use the **final input row, position 31**, of each window, holding absolute position fixed; the context inspector can also show other positions.

Record your hypothesis as connected operations in **discovery.json**. Keep the supplied vocabulary Loss and original residual path. Construct:

```text
h_j = one-column Slice of h for feature 26
X = One-hot encoding of Dataset input token IDs, vocabulary size 25
m = one-column Slice of X for your hypothesized word
on_word  = h_j * m
off_word = h_j * (1 - m)
```

Determine Slice axes and exclusive bounds from the shapes. Use Arithmetic for the elementwise products. h\_j, m and the two products have shape **32 × 1**. Use the Dataset's input IDs, not next-token targets. Find your word's vocabulary index in the Dataset's Data view. Keep both activation branches visible, with no new Params or changes to Loss.

Compare the two branches on the same examples. The grader verifies their construction and scores the association on final positions of all canonical training and validation windows. **AUROC** is the fraction of matching-word / other-word pairs in which activation is larger on the matching word, with ties counting one-half. Require AUROC ≥ **0.9619** and a mean activation gap ≥ **1.4272** on both splits, with at least 10 examples per group. No manually selected examples or written interpretation are submitted.

<details>
<summary>Input-context hint — open after inspecting examples</summary>

This checkpoint’s feature 26 is associated with final-position occurrences of **entered**. Compare surrounding contexts as well as the selected word. This is an input-context association, not proof of a unique concept or an output effect.

</details>

## Build three interventions on feature 26

Preserve discovery.json and make three copies. The supplied Loss already uses **Cross entropy (logits)**, original vocabulary logits and **Dataset target**, with **Penalized activation → None**, **Valid-token mask → All valid positions** and no parameter penalty. Keep those settings and every Param **Frozen**.

Use the second block's final feed-forward residual r, immediately before final output Layer norm. Keep **the original r feeding the encoder**. Only the path from r to output Layer norm receives r′. Preserve the complete SAE reconstruction and discovery branches.

Let v\_j be row 26 of the actual learned decoder Wd, selected by a Slice connected directly to Wd. Use Input, Slice, Arithmetic and Matrix product to construct:

| File | Construction at every input position | Purpose |
| --- | --- | --- |
| zero.json | r′ = r + 0 v\_j | Restore original predictions at zero strength. |
| amplified.json | r′ = r + α v\_j, 0 &lt; α ≤ 1 | Add the bounded direction. Start with α = 1. |
| ablated.json | r′ = r − h\_j v\_j | Remove the feature contribution, retaining reconstruction error. |

Use feature 26 in all three files. Determine Slice and Matrix product shapes. Strength must be a fixed scalar, with no new Params. Derive ablation from the actual encoder activation; a fixed negative strength does not implement ablation. Apply the intervention at every position, without a position mask. Do not replace r with the reconstruction, copy decoder numbers into Input, or modify a vocabulary bias.

## Discover which continuation the direction supports

Compare the same contexts in all three models. Use **Test → Held-out test set → Run inference**, then inspect the final row of the separate vocabulary Softmax in **Data → Dataset row**. Begin with windows ending in your discovered word, and include low-activation controls. The input word associated with a feature need not be the next word it supports.

For a quick comparison select **Dataset → Details → Inspect next-token probabilities** and enter exactly the same prompt in each copy. This display shows the top 12 tokens and excludes padding; the grader uses all 25 entries. Which next word increases under amplification and decreases under ablation? Compare probability changes as well as changes to the most likely token.

The grader uses **all validation windows ending in the discovered input word**, without choosing only successful examples. There are 80 for this checkpoint. Amplification must raise the supported output word's mean probability by at least **0.05**, and ablation must reduce the same word's probability by at least **0.171**, relative to zero. These are absolute probability differences. Start with α = 1; a much smaller strength can fail the effect requirement.

<details>
<summary>Output self-check — open after comparing distributions</summary>

In contexts ending in **entered**, this direction supports **office**. Original mean probability is 0.7171; amplification at α = 1 gives 0.7885, and ablation gives 0.4729. These measurements describe this supplied final-model checkpoint and do not establish improved factual accuracy.

</details>

## Check controls and submit — four models, 100 points

The grader evaluates all 765 canonical validation windows and ranks final-position activations to select the highest and lowest 32, with canonical-order tie breaking. These groups differ from the word-matched contexts above. **Total variation = ½ Σ |p − q|** over the vocabulary measures any distribution change; the word-specific check measures its direction.

| Check | Requirement |
| --- | --- |
| Checkpoint / feature | Unchanged supplied transformer and SAE values; frozen Params; feature 26. |
| Discovery | Connected on-word / off-word activation branches; both-split AUROC ≥ 0.9619 and mean activation gap ≥ 1.4272. |
| Zero restoration | Maximum absolute probability difference ≤ 0.0000001. |
| Amplification | 0 &lt; α ≤ 1; all-window mean TV ≥ 0.005; supported-word mean probability increase ≥ 0.05 on word-matched contexts. |
| Ablation | High-32 mean TV ≥ 0.1611; low-32 mean TV ≤ 0.01; supported-word probability decrease ≥ 0.171 on word-matched contexts. |
| Bounds / collateral change | Max h\_j ≤ 5.4505 over all validation positions. Both nonzero interventions: mean TV ≤ 0.1 and final-position cross-entropy increase ≤ 0.3957. |

The unchanged checkpoint retains its final preference-trained transformer values. Narrative cross entropy is reported diagnostically; SAE MSE must be ≤ 0.20 on both splits, with validation activity ≤ 16 and ≤ 16 inactive features. No retraining is needed.

| Submitted file / component | Points |
| --- | --- |
| discovery.json: checkpoint and connected branches / verified association | 10 / 10 |
| zero.json: construction / restoration | 10 / 10 |
| amplified.json: construction / distribution and word-specific effects | 10 / 10 |
| ablated.json: construction / effects and controls | 15 / 15 |
| Supplied values and assigned feature preserved across all four files | 10 |

Valid components earn independent partial credit. The grader checks saved operations and values, ignoring cached outputs, labels and inspection history. No written responses, screenshots or quiz are required. The existing validation split was used to choose the checkpoint and feature; it is not an untouched test set. A causal output effect does not establish unique meaning or correct factual tracking.

## Optional generation and clamping

Use **Dataset → Details → Continue the text** with **Most likely** to compare continuations of the same prompt. Generation reevaluates the graph at every token; free-running text is not the acceptance test. Optional unsubmitted extensions: try a different feature, clamp to c with r′ = r + (c − h\_j)v\_j, or build a token-position mask. Keep the original residual feeding the encoder.
