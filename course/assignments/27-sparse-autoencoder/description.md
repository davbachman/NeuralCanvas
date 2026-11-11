<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 27: Interpret a transformer with sparse features

A residual coordinate can mix information about many tokens and contexts. Build a new representation of your transformer's residual stream, then compare useful reconstruction with a preference for fewer active features. A feature need not have a simple semantic interpretation.

Start from your own final **rlhf.json** from the preference-learning lesson. Preserve this original file for submission. It contains the transformer after instruction tuning, verifiable-reward adaptation and preference-reward adaptation. This investigation freezes that final model; it does not return to the earlier pretrained generator.

Import rlhf.json and make a copy for narrative inspection. After preserving the original file, choose **Train → Reward learning → Clear reward setup**. This removes the previous reward reference and reports while preserving the current transformer weights. Select the narrative dataset below without training. The QA model selects only the final row of vocabulary logits: reconnect the **full vocabulary logits** directly to both the vocabulary Loss and its separate Softmax branch, then remove the unused final-position Slice. Keep the vocabulary, both transformer blocks, position embeddings and every learned value unchanged. Save this copy as **baseline.json**. Narrative cross entropy is a diagnostic of the adapted model, not a requirement to recover its pre-fine-tuning language-model score.

Select your Dataset block and open **Details → Browse datasets…**. Choose **Text and sequences** → **Football stories**. Choose the **Narrative generation** preset. Set **Training targets** to **Every input position**. Click **Use this dataset**. Keep the supplied data and split unchanged.

**Prepared dataset properties**

| Property | Value |
| --- | --- |
| Vocabulary entries | 25, including unknown and padding |
| Input positions / residual width | 32 / 16 |
| Training / validation windows | 1,534 / 765 |
| Autoencoder features | 32; independent of vocabulary size |

## Preserve the reference model

In a copy of baseline.json, choose **Train → Freeze existing parameters** before adding new Params. This also fixes existing Dropout blocks in evaluation mode. Frozen weights remain in the calculations, but training and parameter initialization do not change their values. Verify **Param → Details → Update setting → Frozen** on the original Params. The four new autoencoder Params must remain **Trainable**.

Use exactly the residual produced by the **second block's feed-forward residual addition, immediately before the final output Layer norm**. Call this tensor **r**; it has 32 rows and 16 coordinates per row. Keep the original residual connected to the vocabulary path. Keep the vocabulary logits and separate Softmax for later use.

## Construct a reconstruction baseline

Using Param, Matrix product, Arithmetic and ReLU, construct:

```text
h = ReLU(r We + be)
r_hat = h Wd + bd
```

Use **32 hidden features**. Determine the shapes of the two matrices and two bias vectors from these equations. Each row of Wd is a 16-coordinate decoder direction. Initialize the two matrices with **Xavier · matrices** and both biases with **zeros**. For Wd select **Norm constraint → Unit row norm · matrix directions**, then initialize it. Each direction is normalized after initialization and every update; a zero direction safely becomes the first coordinate direction. The constraint prevents shrinking the encoder and growing the decoder to evade the sparsity penalty.

Reuse the single Loss block: disconnect its two old inputs, connect r\_hat to prediction and the original r to target, then choose **Mean squared error**. In **Activation objective**, choose **Target source → Frozen internal activation**, **Valid-token mask → All valid positions**, and your ReLU h as **Penalized activation**. Set **Activation L1 strength λ** to **0**. Keep **Parameter regularization → Penalty → None**.

The canonical windows have 32 real input tokens. The engines also exclude padding automatically for internal token reconstruction. An optional binary valid-token mask can exclude additional positions in other datasets, but this comparison uses every canonical input position. The target branch is detached, and its upstream parameters must be frozen and deterministic.

After freezing the transformer and setting the four new Params’ initialization choices, use **Train → Randomize parameters**. Verify that the transformer values remain identical to baseline.json, then save this untrained autoencoder as **sae-start.json**. Reopen this exact file before each of the two training runs so both begin with the same encoder and decoder values. Do not randomize again between those runs. Keep sae-start.json locally; it is not submitted.

**Training settings — Train tab, with Tensor engine selected**

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

Reopen **sae-start.json**, keep λ = 0, train the autoencoder and save **reconstruction.json**. Fit only on the training split. Runtime depends on the device and is not graded.

## Add a sparse comparison

Reopen **sae-start.json** and change only **Activation L1 strength λ** to **0.03**. Keep the transformer, dataset, residual location, 32-feature width, decoder constraint and other training settings fixed. Train and save **sparse.json**. The objective is:

```text
mean over valid tokens and coordinates of (r - r_hat)^2
  + λ × mean over valid tokens of sum over features of |h|
```

Activation L1 is different from the existing parameter penalty. Reconstruction error, activation penalty and total objective are reported separately. A feature is active if its ReLU output is **strictly greater than zero**; the tolerance is zero. An inactive feature never activates over the evaluated split. Small positive values still count as active, and the L1 derivative at zero is zero.

Use **Test → Run inference** on the training split. Select h and open **Data → Inspect feature contexts**. Choose a zero-based **Feature index** and **Examples to inspect**, then **Find high activations**. The scan ranks positive activations and brackets the selected token in its context. Increase the example limit when needed; results describe only the scanned examples. Use this inspection to choose a feature for the construction below. You submit the construction, not a claim that you inspected examples.

## Look for a feature family in your own model

Use your trained model, without a supplied pretrained SAE, to choose a candidate for the submitted contribution below. Possible families include a particular verb or person's name; a word class such as rooms; periods ending events versus consequence clauses; or positions near a story's beginning or end. These are examples to investigate, not patterns guaranteed in every model. Feature indices are local to your encoder.

Inspect several candidates. Compare high activations with low or zero activations in **Data → Dataset row**. Form a tentative description using training examples, then check additional contexts and validation examples. Where possible hold the current word fixed and vary its context. A feature can combine patterns; one striking example does not establish a clean concept. Use the most informative candidate you find, even if its interpretation remains uncertain.

Distinguish **the context that activates a feature** from **the output its direction might support**. A feature active on a name need not promote that name next. The next lesson tests that distinction using a supplied checkpoint. Sparse activity alone does not establish greater interpretability than a dense representation; optionally inspect matching examples in reconstruction.json as a control.

## Independent construction: isolate your candidate's contribution

In sparse.json, choose any feature j from 0 through 31. Use two **Slice** operations and a **Matrix product** to construct the contribution **h<sub>j</sub> v<sub>j</sub>**, where h<sub>j</sub> contains that feature's activation at every token and v<sub>j</sub> is row j of your actual learned decoder Wd. Determine the Slice axes and exclusive bounds yourself. Keep the dimensions needed for a matrix product. The result must have the same 32 × 16 shape as r.

Keep this visible branch in sparse.json, with no new Params and no connection that changes the autoencoder Loss. Do not type copied decoder numbers into an Input. The grader accepts any valid j and independently verifies the connected contribution.

## Submit and grading

Submit **rlhf.json**, **baseline.json**, **reconstruction.json** and **sparse.json**. The unchanged rlhf.json is the handoff artifact; it has no separate points, but establishes the policy preserved in the three graded constructions. The grader recomputes all results from canonical examples and saved graph operations. Cached values, labels, losses and epoch counts are ignored.

| File / component | Points | Requirements |
| --- | --- | --- |
| baseline.json | 5 data + 5 form + 10 quality | Exact final policy values preserved from rlhf.json; canonical narrative evaluation; original QA policy passes the preference-learning handoff checks. |
| reconstruction.json | 5 data + 10 form + 10 quality | Unpenalized, 32-feature autoencoder; MSE ≤ 0.10 on both training and validation. |
| sparse.json | 5 data + 10 form + 10 quality + 10 sparsity + 10 contribution | Same construction; MSE ≤ 0.20 on both splits; ≤ 16 active features/token and ≤ 16 inactive features over validation; valid h<sub>j</sub>v<sub>j</sub> branch. |
| Controlled comparison | 10 | Identical adequate final-policy values in rlhf.json and all three constructions; sparse validation activity ≤ 75% of reconstruction.json activity. |

All autoencoder quality credit requires an adequate original QA policy and exact preservation of its learned values. The grader checks the separately submitted rlhf.json and recomputes its policy and reward-model results; a claimed training stage or cached score is insufficient. Use λ = 0.03 as the tested recipe; bounded values from **0.01 through 0.10** are accepted if they meet the same targets. Every saved decoder row must have norm one (numerical tolerance 0.00005). These bounds leave room for different initializations; they do not award points for a claimed number of epochs. Valid components receive partial credit.

The held-out split is **validation** when used to select settings or features. Preserve all three files for comparison; the intervention lesson supplies a common pretrained checkpoint. Your candidate's contribution is the submitted artifact of this investigation. The grader checks its construction, not a semantic label or claimed inspection history. No written answers, screenshots or separate quiz are required. Optional exploration: inspect other features or compare free-running text later; do not assume every feature is monosemantic.

Background: [Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) and [Scaling Monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/). This small classroom model tests the mechanism; it does not reproduce large-model results.
