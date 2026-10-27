<!-- Generated from description.html; run course/scripts/sync-handouts.py after editing. -->

# Assignment 16: Normalize the block and learn at every position

**Your starting point and handoff**

| Step | What to use |
| --- | --- |
| Start with | A copy of your saved **residual.json** |
| Modify | Part A adds normalization. Copy A for independently trained all-position Part B. Copy trained B for fixed-weight final-position evaluation in C. |
| Save and keep | **transformer.json**, **all\_positions.json**, **last\_position.json**; keep all\_positions.json for the next assignment. |

Successive additions and learned transformations can change feature scales. **Layer norm** computes a mean and variance across the features of each token, then applies a learned scale and offset. This differs from fitting Dataset standardization across training examples.

Use the narrative configuration selected below in your Dataset block. It keeps the same football narratives, **25-token vocabulary**, **32-token input windows** and training/held-out split. There is one target after the final input token. Keep the supplied data and split unchanged. Held-out data is validation during development.

Continue using model width **16** and MLP hidden width **32** unless a step explicitly changes a head’s width.

Use **Narrative generation** throughout. Choose **Final position only** targets in Parts A/C and **Every input position** in Part B.

For Part A and evaluation copy C: Select your Dataset block and open **Details → Browse datasets…**. Choose **Text and sequences** → **Football stories**. Choose the **Narrative generation** preset. Set **Training targets** to **Final position only**. Click **Use this dataset**. Keep the supplied data and split unchanged.

For Part B: Select your Dataset block and open **Details → Browse datasets…**. Choose **Text and sequences** → **Football stories**. Choose the **Narrative generation** preset. Set **Training targets** to **Every input position**. Click **Use this dataset**. Keep the supplied data and split unchanged.

**Prepared dataset properties**

| Dataset property | Parts A and C: last | Part B: all |
| --- | --- | --- |
| Vocabulary entries | 25, including unknown and padding | 25, including unknown and padding |
| Input token IDs | 32 | 32 |
| Position IDs | 32 | 32 |
| Next-token target IDs | 1 | 32 |
| Training windows | 1534 | 1534 |
| Held-out windows | 765 | 765 |

## Part A — Normalize features within each token

Add Layer norm before attention, before the shared MLP, and immediately before the vocabulary projection. Each normalization has independent learned scale and offset Params of shape **(16)**; initialize scale to **ones** and offset to **zeros**. Keep epsilon at **0.00001**.

Wire the graph to implement:

- **R = X + Attention(LN<sub>1</sub>(X))W<sub>O</sub>**
- **H = R + MLP(LN<sub>2</sub>(R))**
- **logits = LN<sub>final</sub>(H)W<sub>vocab</sub> + b<sub>vocab</sub>**

Start with a copy of your saved **residual.json**. In the dataset browser, choose **Football stories → Narrative generation**, set **Training targets** to **Final position only**, and click **Use this dataset**. Keep the final-logits Slice for this part. The shortcut inputs remain X and R. Determine which wires must move; putting normalization on a shortcut changes the assigned model. Train from fresh parameters for **10 epochs in one run** using the settings below, then save **transformer.json**. Keep this trained file before making the next change.

## Part B — Learn from every position

Your transformer produces vocabulary logits at every position but has learned from only the last one. With causal masking, every row can instead predict its own next token in the same forward pass.

In a separate copy, open the dataset browser, choose **Football stories → Narrative generation**, set **Training targets** to **Every input position**, and click **Use this dataset**. It contains exactly the same documents, split and input windows as Part A; the targets now contain **32 shifted token IDs**. For an input t<sub>0</sub>…t<sub>31</sub>, the targets are t<sub>1</sub>…t<sub>32</sub>.

Remove the final-logits Slice and connect all logits to the loss and output Softmax. Retain every causal mask: the target for an earlier row also appears later in the input, so allowing that row to attend forward would leak the answer. Keep the single transformer block and width 16. Randomize and train from fresh parameters for **10 epochs in one run**, then save **all\_positions.json**. Keep this file for the next assignment.

## Part C — Make a comparable evaluation copy

Save a separate copy of the trained Part B model, use the dataset browser to select **Narrative generation** with **Training targets: Final position only**, click **Use this dataset**, and restore the final-logits Slice. **Do not randomize or train this copy.** Save **last\_position.json**. Only all\_positions.json and last\_position.json must have identical learned parameters; transformer.json is a separately trained baseline.

Use held-out inference on **transformer.json** and **last\_position.json** to compare accuracy and loss on the same final-position task. The all-position metrics average different predictions and cannot be compared directly with those numbers. Both models train for 10 epochs with the same architecture, optimizer settings and input windows. Ask whether learning at every position improves held-out final-position prediction at the same number of passes and optimizer updates. Training times can differ, and fresh random initialization can affect the result. The grader checks all three models and their final-position losses; no written comparison is required.

Training positions can be processed together; autoregressive generation still chooses one token after another. Thirty-two targets per window does not promise a 32-fold speedup.

## Parameter-count hint

Your completed models should have the following counts. Count individual learned numbers, including embedding tables and biases, and count each shared Param only once.

- **transformer.json:** **3,529** trainable parameters.
- **all\_positions.json:** **3,529** trainable parameters.
- **last\_position.json:** **3,529** trainable parameters.

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

## Optional: generate a continuation

In **Dataset → Details → Continue the text**, paste this prompt and try **Most likely** or **Sample → Generate 40 tokens**:

> nora entered bedroom . liam entered kitchen . nora took football . nora entered hall . nora dropped football . now the football is in the hall . end nora entered bedroom . liam entered kitchen . nora took football . nora dropped football . nora entered hall . now

The prompt is longer than 32 tokens, so the app uses its last 32. Short prompts need padding and may behave differently. The temperature and sampling controls introduced earlier apply to every generator in this sequence. Use the same prompt to compare Most likely with Sample at temperatures 0.5, 1 and 2, keeping Top-k at 25. Sampling changes token selection without training. A fluent continuation can still misstate a person’s location, the football’s location or who holds it. This inspection is optional; the submitted model itself is the graded artifact.

## Submit and grading

Points below are dataset / model form / training loss / held-out loss. Loss limits are upper bounds. Performance credit requires the assigned architecture. The grader recomputes results from saved parameters; cached scores and declared epoch counts are ignored.

| File | Points | Training loss ≤ | Held-out loss ≤ |
| --- | --- | --- | --- |
| transformer.json | 5 / 25 / 5 / 5 | 0.65 | 0.7 |
| all\_positions.json | 5 / 20 / 5 / 10 | 0.65 | 0.7 |
| last\_position.json | 2 / 4 / 2 / 2 | 0.65 | 0.7 |

An additional **10 points** require identical learned parameters in all\_positions.json, last\_position.json.

Accuracy, parameter counts, common final-position loss and generated continuations are informational. No required written answers, screenshots or unsubmitted investigations.

Assignment 16-normalization.
