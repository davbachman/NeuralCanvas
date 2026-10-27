<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 8: Learn a compact representation of words

**Your starting point and handoff**

| Step | What to use |
| --- | --- |
| Start with | A copy of your saved **stories.json model** |
| Modify | Keep the Dataset and classification loss/probability pattern; switch counts to ordered token IDs and add an MLP. Then copy Part A to build Part B. |
| Save and keep | **onehot.json** and **embedding.json**; keep embedding.json for the next assignment. |

Return to the football questions from the previous assignment. This time preserve the order of every word. First build an MLP from one-hot vectors. Then replace those vectors with learned embeddings and use the saved space to make the hidden layer wider. Both models get ten epochs and approximately the same number of learned parameters.

## The data

Open a copy of your saved **stories.json**, select the Dataset, and use the browser choices below.

Use the football-location dataset with **ordered token IDs**, containing the same 480 training and 240 held-out questions as the previous assignment. Each story contains five events involving Nora, Liam, and a football. Entering a room moves a person and any football they carry; taking picks up the football; dropping leaves it in the person's current room. The question asks **where is football**. Keep the supplied split.

Select your Dataset block and open **Details → Browse datasets…**. Choose **Text and sequences** → **Football stories**. Choose the **Football location · word counts** preset. Set **Model inputs and prediction** to **One ordered token sequence → answer class**. Click **Use this dataset**. Keep the supplied data and split unchanged.

**Prepared dataset properties**

| Dataset property | Supplied value |
| --- | --- |
| Vocabulary entries | 17, including unknown and padding |
| Token IDs per example | 23 |
| Answer classes | 6 rooms |
| Training examples | 480 |
| Held-out examples | 240 |

This configuration changes the representation from the previous assignment’s counts to ordered token IDs. Check the resulting **23 token IDs** and **17 vocabulary entries**, including unknown and padding. The six answer classes are bathroom, bedroom, garden, hall, kitchen and office.

## Part A — Ordered one-hot model (40 points)

1. Connect the Dataset's token IDs to **One-hot** and set its number of classes to **17**. The output for one story is **23 × 17**.
2. Use a Tensor transform set to **Reshape** to make one row of **391** numbers: shape `1, 391`. This keeps all token vectors in their original order. Apply the operation.
3. Build a dense layer with **8 ReLU hidden units**, followed by a dense layer producing **6 logits**. Include a learned bias vector in each layer. Determine the matrix and bias shapes.
4. Use **Cross entropy (logits)** with the Dataset's answer target. Keep a separate **Softmax** branch on the logits.
5. Train from fresh parameters using the recipe below. Save **onehot.json**.

## Part B — Learned embeddings and a wider hidden layer (60 points)

Preserve Part A, then make a copy. Replace its one-hot representation with a learned **four-dimensional embedding** for each token and increase the ReLU hidden layer to **32 units**.

To create the new representation, add **Embedding** and a Param table with shape **17, 4**. Set that Param's Initialize choice to **uniform** and initialize it. The table connects to the Embedding block's first input; the Dataset token IDs connect to its second. Its output is one four-number vector per token.

Complete the adaptation independently. Keep all **23 embedding vectors in their original order as a single row**. Work out the new reshape, matrix, and bias shapes, and wire the representation into the rest of the model. Remove the old one-hot block and any unused Params. Retain six output logits, the original target, and the probability branch.

After applying the shape changes, initialize *all* Params afresh, then train for the same ten epochs. Aim for **at least 70% held-out accuracy**, training loss at most **0.35**, and held-out loss at most **1.10**. Save **embedding.json**. Do not deliberately weaken your one-hot model; the grader uses independent targets for both files.

## Training recipe for both models

First set **Training execution** to **Tensor engine · minibatches** to show the Tensor controls. Then follow the Train tab from top to bottom:

**Training settings — Train tab**

| Control | Value |
| --- | --- |
| Epochs per run | 10 |
| Report loss every | 1 |
| Examples per update | 32 |
| Reshuffle training examples each epoch | Checked |
| Exact learning rate | 0.02 |
| Training execution | Tensor engine · minibatches |
| Backend | Tensor CPU |
| Optimizer | Adam |
| Gradient norm limit (0 = off) | 0 |
| Early stopping patience (0 = off) | 0 |

**Loss block → Details → Parameter regularization**

| Control | Value |
| --- | --- |
| Penalty | None |

**Param → Initialize choices**

| Param block | Initialize |
| --- | --- |
| Word embedding table | uniform |
| Other matrices | Xavier · matrices |
| Bias vectors | zeros |

Apply changed shapes, check saved Initialize choices, then use **Train → Randomize parameters**. Train each model from fresh parameters; preserve your Part A file before adapting a copy for Part B.

If a run misses the targets, check the graph and initialization choices first. Ten epochs is the intended comparison budget; grading measures the saved model's structure and results, not a claimed epoch count. The held-out set serves as validation during development.

## Parameter-count hint

- **onehot.json:** **3,190** trainable parameters.
- **embedding.json:** **3,242** trainable parameters.

These counts differ by less than 2%. Similar parameter counts and update budgets do not imply identical runtime: embedding lookup and larger hidden layers involve different operations.

## Inspect your submitted models

Use **Test → Held-out test set → Run inference** to compare their results. Softmax → Data shows the probabilities; match the original Dataset row numbers to the report. These views support your submitted models. No written explanation or separate investigation is required.

## Submit — two files, 100 points

Submit **onehot.json** and **embedding.json**. Performance credit requires the assigned architecture. The grader recomputes predictions on the original data and ignores cached losses and claimed training history.

| File | Data | Model form | Training loss | Held-out loss | Accuracy | Total |
| --- | --- | --- | --- | --- | --- | --- |
| onehot.json | 5 | 20 | 10 (≤ 1.50) | 5 (≤ 1.55) | Informational | 40 |
| embedding.json | 5 | 25 | 10 (≤ 0.35) | 10 (≤ 1.10) | 10 (≥ 70%) | 60 |

Assignment embeddings-08. Course-generated football stories; the data and split match the earlier word-count dataset.
